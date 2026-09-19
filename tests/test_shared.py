import json

import pytest

from semif_phase1.shared import _suffix_layout


def test_suffix_padding_follows_real_tokens():
    layout, ends = _suffix_layout([[3, 4], [5]], 7, 0)
    assert layout["input_ids"] == [[3, 4], [5, 0]]
    assert layout["attention_mask"] == [[1] * 9, [1] * 8 + [0]]
    assert layout["position_ids"] == [[7, 8], [7, 0]]
    assert ends == [1, 0]


def test_empty_suffix_is_rejected():
    with pytest.raises(ValueError):
        _suffix_layout([[1], []], 7, 0)


class _MergingTokenizer:
    """A tokenizer whose vocabulary merges `]}` into one token, like MiniCPM's does.

    Characters map to their own ids; the pair `]}` becomes a single fused id. That is enough
    to reproduce the boundary a single-token trim cannot clear.
    """

    FUSED = 10_000

    def encode(self, text, add_special_tokens=False):
        ids, index = [], 0
        while index < len(text):
            if text.startswith("]}", index):
                ids.append(self.FUSED)
                index += 2
            else:
                ids.append(ord(text[index]))
                index += 1
        return ids

    def apply_chat_template(self, turns, **kwargs):
        return "<s>" + turns[-1]["content"] + "</s>"


def test_prefix_trims_past_a_multi_token_boundary_merge():
    """A state ending in `[]` merges with the following `}`; one trim is not enough."""
    from semif_phase1.shared import _state_prefix

    tokenizer = _MergingTokenizer()
    state = {"recent_actions": []}
    prefix = _state_prefix(tokenizer, state)

    full = tokenizer.encode("<s>" + json.dumps({"evidence": state, "criterion": "x"}, ensure_ascii=False))
    assert prefix, "a prefix must survive the trim"
    assert full[: len(prefix)] == prefix, "the trimmed prefix must be a real prefix of the prompt"


def test_prefix_keeps_the_single_token_trim_when_that_already_works():
    """A vocabulary with no boundary merge must be unaffected, so published runs do not move."""
    from semif_phase1.shared import _state_prefix

    class Plain(_MergingTokenizer):
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    tokenizer = Plain()
    state = {"page": "hello"}
    evidence = json.dumps({"evidence": state}, ensure_ascii=False)[:-1]
    assert len(_state_prefix(tokenizer, state)) == len("<s>" + evidence) - 1
