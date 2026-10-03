"""No-slot decision canvas preprocessing."""

import pytest

from axolotl.integrations.decision.preprocessing import build_decision_canvas


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(text.encode("utf-8"))


def record():
    return {
        "source": "synthetic",
        "group": "test",
        "state": "observed",
        "questions": {"q": {"type": "choice", "options": ["no", "yes"]}},
        "labels": {"q": {"kind": "hard", "gold_idx": 1}},
    }


def test_no_slot_canvas_preserves_targets_and_kstep_pinning():
    first = build_decision_canvas(Tokenizer(), record(), [9], scaffold_ids=(), turn_close_id=4, pad_id=0, vocab_size=256, seed=3)
    later = build_decision_canvas(Tokenizer(), record(), [9], scaffold_ids=(), turn_close_id=4, pad_id=0, vocab_size=256, seed=3, steps=2)
    assert first.targets == ({"kind": "hard", "gold_idx": 1},)
    assert first.question_ids == ("q",)
    assert first.allowed_ids and first.template_length > 0
    assert not any(first.slot_mask)
    assert all(first.semantic_mask)
    assert all(later.pinned_mask[i] == (i not in first.label_positions) for i in range(128))


def test_slot_plan_is_rejected():
    with pytest.raises(ValueError, match="latent slots were removed"):
        build_decision_canvas(Tokenizer(), record(), [9], scaffold_ids=(), turn_close_id=4, pad_id=0, vocab_size=256, slot_plan=object())
