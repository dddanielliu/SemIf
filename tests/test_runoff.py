import pytest

from semif_phase1.runoff import marginals, partition, score_options


def fake_scorer(scores):
    """Return a score_shared-compatible scorer driven by a fixed option -> score table."""
    calls = []

    def scorer(_model, _tokenizer, rows, _metadata, _max_tokens):
        calls.append([row["id"] for row in rows])
        results = []
        for row in rows:
            raw = [scores[option["id"]] for option in row["options"]]
            total = sum(raw)
            results.append({"id": row["id"], "probabilities": [value / total for value in raw]})
        return results, {"prefix_tokens": 7, "true_suffix_tokens": 3 * len(rows)}

    scorer.calls = calls
    return scorer


def decision(identifier, count, offset=0):
    return {
        "id": identifier,
        "question": "Which one?",
        "options": [{"id": f"o{offset + index}", "description": f"option {index}"} for index in range(count)],
    }


def test_small_option_set_stays_one_group():
    assert partition(["a", "b", "c"], 16) == [["a", "b", "c"]]


def test_partition_balances_and_never_leaves_a_singleton():
    groups = partition([f"o{index}" for index in range(17)], 16)
    assert [len(group) for group in groups] == [9, 8]
    groups = partition([f"o{index}" for index in range(250)], 16)
    assert len(groups) == 16
    assert sorted(len(group) for group in groups) == [15] * 6 + [16] * 10
    assert sum(len(group) for group in groups) == 250
    assert min(len(group) for group in groups) >= 2


def test_partition_rejects_degenerate_input():
    with pytest.raises(ValueError):
        partition(["a"], 16)
    with pytest.raises(ValueError):
        partition(["a", "b"], 1)


def test_marginals_multiply_group_and_within_group_probabilities():
    rounds = [
        [
            {"option_ids": ["a", "b"], "probabilities": [0.75, 0.25], "winner": "a"},
            {"option_ids": ["c", "d"], "probabilities": [0.4, 0.6], "winner": "d"},
        ],
        [{"option_ids": ["a", "d"], "probabilities": [0.2, 0.8], "winner": "d"}],
    ]
    result = marginals(rounds)
    assert result == pytest.approx({"a": 0.15, "b": 0.05, "c": 0.32, "d": 0.48})
    assert sum(result.values()) == pytest.approx(1.0)


def test_marginals_require_a_decisive_final_round():
    with pytest.raises(ValueError):
        marginals([[{"option_ids": ["a"], "probabilities": [1.0], "winner": "a"}, {}]])


def test_options_beyond_one_alphabet_resolve_through_rounds():
    scores = {f"o{index}": 1.0 for index in range(40)}
    scores["o37"] = 100.0
    scorer = fake_scorer(scores)
    results, timing = score_options(
        None, None, "state", [decision("d", 40)], {}, max_slots=16, scorer=scorer
    )
    (result,) = results
    assert result["option_ids"] == [f"o{index}" for index in range(40)]
    assert sum(result["probabilities"]) == pytest.approx(1.0)
    assert result["option_ids"][result["probabilities"].index(max(result["probabilities"]))] == "o37"
    assert result["rounds"] == timing["rounds"] == 2


def test_every_decision_and_group_shares_one_round_batch():
    scorer = fake_scorer({f"o{index}": index + 1.0 for index in range(60)})
    score_options(
        None,
        None,
        "state",
        [decision("a", 20), decision("b", 20, offset=20), decision("c", 20, offset=40)],
        {},
        max_slots=16,
        scorer=scorer,
    )
    # Round one: two groups per decision. Round two: the two winners of each decision.
    assert [len(call) for call in scorer.calls] == [6, 3]


def test_max_batch_splits_a_round_without_changing_the_result():
    scores = {f"o{index}": index + 1.0 for index in range(40)}
    unbounded = fake_scorer(scores)
    bounded = fake_scorer(scores)
    expected, _ = score_options(None, None, "s", [decision("d", 40)], {}, max_slots=16, scorer=unbounded)
    actual, timing = score_options(
        None, None, "s", [decision("d", 40)], {}, max_slots=16, max_batch=2, scorer=bounded
    )
    assert actual[0]["probabilities"] == pytest.approx(expected[0]["probabilities"])
    assert max(len(call) for call in bounded.calls) == 2
    assert timing["batches"] > timing["rounds"]


def test_single_option_decision_never_reaches_the_scorer():
    scorer = fake_scorer({"o0": 1.0})
    results, timing = score_options(None, None, "s", [decision("d", 1)], {}, scorer=scorer)
    assert results[0]["probabilities"] == [1.0]
    assert scorer.calls == [] and timing["rounds"] == 0


def test_duplicate_identifiers_are_rejected():
    with pytest.raises(ValueError):
        score_options(None, None, "s", [decision("d", 3), decision("d", 3)], {}, scorer=fake_scorer({}))


def test_a_warm_prefix_is_offered_to_scorers_that_accept_one():
    """The default scorer reuses one prefill across rounds; a plain scorer never sees it."""
    seen = []

    def warm_aware(_model, _tokenizer, rows, _metadata, _max_tokens, warm=None):
        seen.append(warm)
        return (
            [{"id": row["id"], "probabilities": [1 / len(row["options"])] * len(row["options"])} for row in rows],
            {"prefix_tokens": 5, "true_suffix_tokens": 1},
        )

    score_options(None, None, "s", [decision("d", 40)], {}, max_slots=16, scorer=warm_aware)
    assert len(seen) == 2, "two rounds"
    assert seen[0] is seen[1] is not None, "the same warm prefix spans both rounds"
    assert seen[0].prefix is None, "released once the decision is done"


def test_a_plain_scorer_is_called_without_a_warm_prefix():
    scorer = fake_scorer({f"o{index}": 1.0 for index in range(40)})
    score_options(None, None, "s", [decision("d", 40)], {}, max_slots=16, scorer=scorer)
    assert len(scorer.calls) == 2
