"""
Unit tests for RL data utility functions (excess_length_strategy support).
"""

import unittest

from axolotl.utils.data.rl import (
    _drop_long_sequences,
    _raise_on_long_sequences,
    _truncate_long_sequences_rl,
)
from axolotl.utils.schemas.enums import RLType


class _FakeTokenizer:
    """Simple whitespace tokenizer for testing length calculations."""

    def __call__(self, text, add_special_tokens=True):  # noqa: ARG002
        tokens = text.split()
        return {"input_ids": list(range(len(tokens)))}

    def decode(self, token_ids, skip_special_tokens=True):  # noqa: ARG002
        # Each token id maps to a placeholder word; length is what matters.
        return " ".join(f"w{i}" for i in range(len(token_ids)))


def _make_dpo_sample(prompt_len: int, chosen_len: int, rejected_len: int):
    """Create a DPO sample with specified word counts."""
    return {
        "prompt": " ".join(f"p{i}" for i in range(prompt_len)),
        "chosen": " ".join(f"c{i}" for i in range(chosen_len)),
        "rejected": " ".join(f"r{i}" for i in range(rejected_len)),
    }


def _make_kto_sample(prompt_len: int, completion_len: int):
    """Create a KTO sample with specified word counts."""
    return {
        "prompt": " ".join(f"p{i}" for i in range(prompt_len)),
        "completion": " ".join(f"c{i}" for i in range(completion_len)),
    }


class TestDropLongSequences(unittest.TestCase):
    """Tests for the existing _drop_long_sequences filter function."""

    def setUp(self):
        self.tokenizer = _FakeTokenizer()

    def test_dpo_keeps_short_samples(self):
        sample = _make_dpo_sample(prompt_len=3, chosen_len=2, rejected_len=2)
        result = _drop_long_sequences(
            sample, RLType.DPO, self.tokenizer, sequence_len=10
        )
        self.assertTrue(result)

    def test_dpo_drops_long_chosen(self):
        sample = _make_dpo_sample(prompt_len=5, chosen_len=10, rejected_len=2)
        result = _drop_long_sequences(
            sample, RLType.DPO, self.tokenizer, sequence_len=10
        )
        self.assertFalse(result)

    def test_dpo_drops_long_rejected(self):
        sample = _make_dpo_sample(prompt_len=5, chosen_len=2, rejected_len=10)
        result = _drop_long_sequences(
            sample, RLType.DPO, self.tokenizer, sequence_len=10
        )
        self.assertFalse(result)

    def test_kto_keeps_short_samples(self):
        sample = _make_kto_sample(prompt_len=3, completion_len=2)
        result = _drop_long_sequences(
            sample, RLType.KTO, self.tokenizer, sequence_len=10
        )
        self.assertTrue(result)

    def test_kto_drops_long_completion(self):
        sample = _make_kto_sample(prompt_len=5, completion_len=10)
        result = _drop_long_sequences(
            sample, RLType.KTO, self.tokenizer, sequence_len=10
        )
        self.assertFalse(result)

    def test_grpo_always_keeps(self):
        sample = {"prompt": "a " * 100}
        result = _drop_long_sequences(
            sample, RLType.GRPO, self.tokenizer, sequence_len=5
        )
        self.assertTrue(result)

    def test_dpo_missing_keys_raises(self):
        with self.assertRaises(ValueError):
            _drop_long_sequences({"prompt": "hi"}, RLType.DPO, self.tokenizer, 10)

    def test_kto_missing_keys_raises(self):
        with self.assertRaises(ValueError):
            _drop_long_sequences({"prompt": "hi"}, RLType.KTO, self.tokenizer, 10)

    def test_ipo_uses_dpo_logic(self):
        sample = _make_dpo_sample(prompt_len=5, chosen_len=10, rejected_len=2)
        result = _drop_long_sequences(
            sample, RLType.IPO, self.tokenizer, sequence_len=10
        )
        self.assertFalse(result)

    def test_orpo_uses_dpo_logic(self):
        sample = _make_dpo_sample(prompt_len=3, chosen_len=2, rejected_len=2)
        result = _drop_long_sequences(
            sample, RLType.ORPO, self.tokenizer, sequence_len=10
        )
        self.assertTrue(result)

    def test_boundary_length_kept(self):
        """Samples exactly at sequence_len should be kept."""
        sample = _make_dpo_sample(prompt_len=5, chosen_len=5, rejected_len=5)
        result = _drop_long_sequences(
            sample, RLType.DPO, self.tokenizer, sequence_len=10
        )
        self.assertTrue(result)


class TestRaiseOnLongSequences(unittest.TestCase):
    """Tests for _raise_on_long_sequences (excess_length_strategy='raise')."""

    def setUp(self):
        self.tokenizer = _FakeTokenizer()

    def test_short_sample_passes(self):
        sample = _make_dpo_sample(prompt_len=3, chosen_len=2, rejected_len=2)
        result = _raise_on_long_sequences(
            sample, RLType.DPO, self.tokenizer, sequence_len=10
        )
        self.assertTrue(result)

    def test_long_sample_raises_valueerror(self):
        sample = _make_dpo_sample(prompt_len=5, chosen_len=10, rejected_len=2)
        with self.assertRaises(ValueError, msg="excess_length_strategy"):
            _raise_on_long_sequences(
                sample, RLType.DPO, self.tokenizer, sequence_len=10
            )

    def test_kto_long_raises(self):
        sample = _make_kto_sample(prompt_len=5, completion_len=10)
        with self.assertRaises(ValueError):
            _raise_on_long_sequences(
                sample, RLType.KTO, self.tokenizer, sequence_len=10
            )

    def test_grpo_never_raises(self):
        sample = {"prompt": "a " * 100}
        result = _raise_on_long_sequences(
            sample, RLType.GRPO, self.tokenizer, sequence_len=5
        )
        self.assertTrue(result)


class TestTruncateLongSequencesRL(unittest.TestCase):
    """Tests for _truncate_long_sequences_rl (excess_length_strategy='truncate')."""

    def setUp(self):
        self.tokenizer = _FakeTokenizer()

    def test_dpo_short_sample_unchanged(self):
        sample = _make_dpo_sample(prompt_len=3, chosen_len=2, rejected_len=2)
        result = _truncate_long_sequences_rl(
            sample, RLType.DPO, self.tokenizer, sequence_len=10
        )
        self.assertEqual(result["chosen"], sample["chosen"])
        self.assertEqual(result["rejected"], sample["rejected"])

    def test_dpo_truncates_chosen(self):
        sample = _make_dpo_sample(prompt_len=5, chosen_len=10, rejected_len=3)
        result = _truncate_long_sequences_rl(
            sample, RLType.DPO, self.tokenizer, sequence_len=10
        )
        # max_response_len = 10 - 5 = 5, chosen had 10 words -> truncated to 5
        chosen_tokens = self.tokenizer(result["chosen"], add_special_tokens=False)[
            "input_ids"
        ]
        self.assertEqual(len(chosen_tokens), 5)

    def test_dpo_truncates_rejected(self):
        sample = _make_dpo_sample(prompt_len=5, chosen_len=3, rejected_len=10)
        result = _truncate_long_sequences_rl(
            sample, RLType.DPO, self.tokenizer, sequence_len=10
        )
        rejected_tokens = self.tokenizer(result["rejected"], add_special_tokens=False)[
            "input_ids"
        ]
        self.assertEqual(len(rejected_tokens), 5)

    def test_dpo_truncates_both(self):
        sample = _make_dpo_sample(prompt_len=5, chosen_len=10, rejected_len=10)
        result = _truncate_long_sequences_rl(
            sample, RLType.DPO, self.tokenizer, sequence_len=10
        )
        chosen_len = len(
            self.tokenizer(result["chosen"], add_special_tokens=False)["input_ids"]
        )
        rejected_len = len(
            self.tokenizer(result["rejected"], add_special_tokens=False)["input_ids"]
        )
        self.assertEqual(chosen_len, 5)
        self.assertEqual(rejected_len, 5)

    def test_dpo_prompt_unchanged(self):
        """Prompt text should never be modified."""
        sample = _make_dpo_sample(prompt_len=5, chosen_len=10, rejected_len=10)
        result = _truncate_long_sequences_rl(
            sample, RLType.DPO, self.tokenizer, sequence_len=10
        )
        self.assertEqual(result["prompt"], sample["prompt"])

    def test_dpo_prompt_exceeds_limit_returns_unchanged(self):
        """When prompt alone exceeds sequence_len, sample is returned as-is."""
        sample = _make_dpo_sample(prompt_len=15, chosen_len=3, rejected_len=3)
        result = _truncate_long_sequences_rl(
            sample, RLType.DPO, self.tokenizer, sequence_len=10
        )
        self.assertEqual(result, sample)

    def test_kto_truncates_completion(self):
        sample = _make_kto_sample(prompt_len=5, completion_len=10)
        result = _truncate_long_sequences_rl(
            sample, RLType.KTO, self.tokenizer, sequence_len=10
        )
        completion_len = len(
            self.tokenizer(result["completion"], add_special_tokens=False)["input_ids"]
        )
        self.assertEqual(completion_len, 5)

    def test_kto_short_sample_unchanged(self):
        sample = _make_kto_sample(prompt_len=3, completion_len=2)
        result = _truncate_long_sequences_rl(
            sample, RLType.KTO, self.tokenizer, sequence_len=10
        )
        self.assertEqual(result["completion"], sample["completion"])

    def test_kto_prompt_exceeds_limit_returns_unchanged(self):
        sample = _make_kto_sample(prompt_len=15, completion_len=3)
        result = _truncate_long_sequences_rl(
            sample, RLType.KTO, self.tokenizer, sequence_len=10
        )
        self.assertEqual(result, sample)

    def test_grpo_unchanged(self):
        sample = {"prompt": "a " * 100}
        result = _truncate_long_sequences_rl(
            sample, RLType.GRPO, self.tokenizer, sequence_len=5
        )
        self.assertEqual(result, sample)

    def test_ipo_uses_dpo_logic(self):
        sample = _make_dpo_sample(prompt_len=5, chosen_len=10, rejected_len=3)
        result = _truncate_long_sequences_rl(
            sample, RLType.IPO, self.tokenizer, sequence_len=10
        )
        chosen_len = len(
            self.tokenizer(result["chosen"], add_special_tokens=False)["input_ids"]
        )
        self.assertEqual(chosen_len, 5)

    def test_does_not_mutate_original(self):
        """Verify immutability — original sample dict is not modified."""
        sample = _make_dpo_sample(prompt_len=5, chosen_len=10, rejected_len=10)
        original_chosen = sample["chosen"]
        original_rejected = sample["rejected"]
        _truncate_long_sequences_rl(sample, RLType.DPO, self.tokenizer, sequence_len=10)
        self.assertEqual(sample["chosen"], original_chosen)
        self.assertEqual(sample["rejected"], original_rejected)

    def test_dpo_missing_keys_raises(self):
        with self.assertRaises(ValueError):
            _truncate_long_sequences_rl(
                {"prompt": "hi"}, RLType.DPO, self.tokenizer, 10
            )

    def test_kto_missing_keys_raises(self):
        with self.assertRaises(ValueError):
            _truncate_long_sequences_rl(
                {"prompt": "hi"}, RLType.KTO, self.tokenizer, 10
            )

    def test_boundary_no_truncation_needed(self):
        """Samples exactly at sequence_len should not be modified."""
        sample = _make_dpo_sample(prompt_len=5, chosen_len=5, rejected_len=5)
        result = _truncate_long_sequences_rl(
            sample, RLType.DPO, self.tokenizer, sequence_len=10
        )
        self.assertEqual(result["chosen"], sample["chosen"])
        self.assertEqual(result["rejected"], sample["rejected"])
