"""Decisions with more options than the answer-slot alphabet, via batched runoffs.

`direct` and `shared` read one logit per declared option, so a single decision can offer at
most `len(LETTERS)` options. Callers that must rank more than that -- an agent choosing among
every control it can see, for instance -- partition the options into groups, score the groups
against the same state, and run the group winners off against each other.

Every group in a round is one row of a single `score_shared` batch, so a round costs one
prefill regardless of how many decisions or groups it contains. Rounds after the first are
small: only the winners advance.

The returned distribution is a runoff product, `P(option) = P(its group) * P(option | group)`.
It is normalised over all supplied options, but it is not the distribution a single forward
pass over all of them would have produced, and the two disagree whenever a group's internal
ordering depends on which other options shared its prompt.
"""

from __future__ import annotations

import math
import time

from .core import LETTERS
from .shared import score_shared

MAX_SLOTS = len(LETTERS)
SEPARATOR = "\x1f"


def partition(option_ids: list[str], max_size: int = MAX_SLOTS) -> list[list[str]]:
    """Split options into balanced groups, never leaving a group the scorer would reject."""
    if max_size < 2:
        raise ValueError("max_size must allow at least two options")
    count = len(option_ids)
    if count < 2:
        raise ValueError("Need at least two options to partition")
    if count <= max_size:
        return [list(option_ids)]
    groups_count = math.ceil(count / max_size)
    base, extra = divmod(count, groups_count)
    if base < 2:
        raise ValueError("Cannot partition without creating a single-option group")
    groups, start = [], 0
    for index in range(groups_count):
        size = base + (1 if index < extra else 0)
        groups.append(list(option_ids[start : start + size]))
        start += size
    return groups


def marginals(rounds: list[list[dict]]) -> dict[str, float]:
    """Collapse one decision's rounds into a single distribution over its original options.

    `rounds` is earliest first. Each round is a list of groups carrying `option_ids`,
    `probabilities`, and the `winner` that advanced. The final round holds exactly one group.
    Walking backwards, a group's members inherit the marginal already established for their
    winner, which is exactly the probability that their group won.
    """
    if not rounds or len(rounds[-1]) != 1:
        raise ValueError("Rounds must end with a single decisive group")
    result: dict[str, float] = {}
    for group in rounds[-1]:
        for option_id, probability in zip(group["option_ids"], group["probabilities"]):
            result[option_id] = probability
    for groups in reversed(rounds[:-1]):
        for group in groups:
            # Read before writing: the winner's own marginal is rescaled by its group score too.
            base = result[group["winner"]]
            for option_id, probability in zip(group["option_ids"], group["probabilities"]):
                result[option_id] = base * probability
    return result


def _winner(option_ids: list[str], probabilities: list[float]) -> str:
    best = max(range(len(option_ids)), key=lambda index: probabilities[index])
    return option_ids[best]


def score_options(
    model,
    tokenizer,
    state,
    decisions: list[dict],
    metadata: dict,
    max_tokens: int = 4096,
    max_slots: int = MAX_SLOTS,
    max_batch: int = 0,
    scorer=score_shared,
) -> tuple[list[dict], dict]:
    """Score every decision against one shared state, with no cap on options per decision.

    `decisions` are `{"id", "question", "options": [{"id", "description"}]}`. `max_batch`
    bounds the rows sent to one `score_shared` call: that call replicates the state prefix
    across the batch, so memory grows with batch size and a large round can exhaust a GPU.
    Zero means no bound. `scorer` is injectable so the batching logic can be exercised
    without a model.
    """
    if not decisions:
        raise ValueError("Need at least one decision")
    if len({decision["id"] for decision in decisions}) != len(decisions):
        raise ValueError("Decision IDs must be unique")
    if max_slots < 2 or max_batch < 0:
        raise ValueError("max_slots must be at least two and max_batch cannot be negative")

    started = time.perf_counter()
    questions, catalog, order = {}, {}, {}
    pools, history, settled = {}, {}, {}
    for decision in decisions:
        identifier, options = decision["id"], decision["options"]
        if SEPARATOR in identifier:
            raise ValueError("Decision IDs cannot contain the batch row separator")
        if not options:
            raise ValueError(f"Decision {identifier!r} has no options")
        ids = [option["id"] for option in options]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Decision {identifier!r} has duplicate option IDs")
        questions[identifier] = decision["question"]
        catalog[identifier] = {option["id"]: option["description"] for option in options}
        order[identifier] = ids
        history[identifier] = []
        if len(ids) == 1:
            # One candidate cannot lose a comparison the scorer is not allowed to run.
            settled[identifier] = {ids[0]: 1.0}
        else:
            pools[identifier] = ids

    rounds_run, batches_run, prefill_tokens, suffix_tokens = 0, 0, 0, 0
    while pools:
        planned = []
        for identifier, pool in pools.items():
            for group in partition(pool, max_slots):
                planned.append((identifier, group))
        rows = [
            {
                "id": f"{identifier}{SEPARATOR}{rounds_run}{SEPARATOR}{index}",
                "state": state,
                "question": questions[identifier],
                "options": [{"id": option, "description": catalog[identifier][option]} for option in group],
            }
            for index, (identifier, group) in enumerate(planned)
        ]
        width = max_batch or len(rows)
        scored = {}
        for start in range(0, len(rows), width):
            chunk = rows[start : start + width]
            results, timing = scorer(model, tokenizer, chunk, metadata, max_tokens)
            batches_run += 1
            prefill_tokens += timing.get("prefix_tokens", 0)
            suffix_tokens += timing.get("true_suffix_tokens", 0)
            scored.update({result["id"]: result for result in results})

        advancing = {}
        for row, (identifier, group) in zip(rows, planned):
            result = scored[row["id"]]
            probabilities = list(result["probabilities"])
            if len(probabilities) != len(group):
                raise RuntimeError(f"Scorer returned {len(probabilities)} scores for {len(group)} options")
            advancing.setdefault(identifier, []).append(
                {"option_ids": list(group), "probabilities": probabilities, "winner": _winner(group, probabilities)}
            )

        pools = {}
        for identifier, groups in advancing.items():
            history[identifier].append(groups)
            if len(groups) == 1:
                settled[identifier] = marginals(history[identifier])
            else:
                pools[identifier] = [group["winner"] for group in groups]
        rounds_run += 1

    results = []
    for decision in decisions:
        identifier = decision["id"]
        distribution = settled[identifier]
        ids = order[identifier]
        probabilities = [distribution[option] for option in ids]
        total = sum(probabilities)
        if not math.isfinite(total) or total <= 0:
            raise RuntimeError(f"Decision {identifier!r} produced no usable distribution")
        # Guard against drift accumulated by multiplying round probabilities together.
        probabilities = [value / total for value in probabilities]
        results.append(
            {
                "id": identifier,
                "option_ids": ids,
                "probabilities": probabilities,
                "rounds": len(history[identifier]),
                "model": metadata,
                "readout": "runoff over native option-logit groups",
                "probability_status": (
                    "conditional option score combined across runoff rounds; "
                    "uncalibrated as decision confidence and not equal to a single-pass distribution"
                ),
            }
        )
    timing = {
        "total_seconds": time.perf_counter() - started,
        "rounds": rounds_run,
        "batches": batches_run,
        "prefill_tokens": prefill_tokens,
        "suffix_tokens": suffix_tokens,
        "decisions": len(decisions),
        "max_slots": max_slots,
        "max_batch": max_batch,
    }
    return results, timing
