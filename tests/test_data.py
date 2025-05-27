"""
test module for the axolotl.utils.data module
"""

import unittest
from unittest.mock import MagicMock

from transformers import LlamaTokenizer

from axolotl.utils.data import encode_pretraining, md5
from axolotl.utils.data.rl import drop_long_rl_seq

from tests.hf_offline_utils import enable_hf_offline


class TestEncodePretraining(unittest.TestCase):
    """
    test class for encode pretraining and md5 helper
    """

    @enable_hf_offline
    def setUp(self):
        self.tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
        self.tokenizer.add_special_tokens(
            {
                "eos_token": "</s>",
                "bos_token": "<s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        )
        self.max_tokens = 15  # set a small number for easy inspection

    def test_encode_pretraining(self):
        examples = {
            "text": [
                "Hello, world!",
                "Nice to meet you.",
                "lorem ipsum dolor sit amet.",
                "Nice to meet you again!.",
                "hello, hello",
            ]
        }
        result = encode_pretraining(self.tokenizer, self.max_tokens, examples)

        self.assertEqual(len(result["input_ids"]), 3)

        # Assert the length of input_ids and attention_mask is correct
        self.assertEqual(len(result["input_ids"][0]), self.max_tokens)
        self.assertEqual(len(result["attention_mask"][0]), self.max_tokens)

        # Assert EOS and PAD tokens are correctly added
        # hello world! is 4 tokens
        self.assertEqual(result["input_ids"][0][0], self.tokenizer.bos_token_id)
        self.assertEqual(result["input_ids"][0][5], self.tokenizer.eos_token_id)
        self.assertEqual(result["input_ids"][0][6], self.tokenizer.pad_token_id)
        # second part, 5 tokens
        self.assertEqual(result["input_ids"][0][7], self.tokenizer.bos_token_id)
        self.assertEqual(result["input_ids"][0][13], self.tokenizer.eos_token_id)
        self.assertEqual(result["input_ids"][0][14], self.tokenizer.pad_token_id)

    def test_md5(self):
        self.assertEqual(md5("hello world"), "5eb63bbbe01eeed093cb22bb8f5acdc3")
        self.assertEqual(
            md5("hello world", "utf-8"), "5eb63bbbe01eeed093cb22bb8f5acdc3"
        )


class TestDropLongRLSeq(unittest.TestCase):
    """
    Tests for the drop_long_rl_seq function.
    """

    def setUp(self):
        # Mock tokenizer that returns length based on input string length
        self.tokenizer = MagicMock()

        def side_effect_func(
            text, add_special_tokens=False
        ):  # pylint: disable=unused-argument
            return {"input_ids": list(range(len(text)))}

        self.tokenizer.side_effect = side_effect_func
        self.tokenizer.decode = lambda tokens, skip_special_tokens: "".join(
            ["x"] * len(tokens)
        )  # pylint: disable=unused-argument

        self.sequence_len = 20

    def test_dpo_drop_mode_valid(self):
        """Test DPO drop mode with a valid sample."""
        sample = {
            "prompt": "p" * 5,
            "chosen": "c" * 7,
            "rejected": "r" * 6,
        }  # 5+7=12 <= 20, 5+6=11 <= 20
        result = drop_long_rl_seq(
            sample, "dpo", self.tokenizer, self.sequence_len, handling="drop"
        )
        self.assertTrue(result)

    def test_dpo_drop_mode_invalid_chosen(self):
        """Test DPO drop mode with chosen too long."""
        sample = {
            "prompt": "p" * 5,
            "chosen": "c" * 16,
            "rejected": "r" * 6,
        }  # 5+16=21 > 20
        result = drop_long_rl_seq(
            sample, "dpo", self.tokenizer, self.sequence_len, handling="drop"
        )
        self.assertFalse(result)

    def test_dpo_drop_mode_invalid_rejected(self):
        """Test DPO drop mode with rejected too long."""
        sample = {
            "prompt": "p" * 5,
            "chosen": "c" * 7,
            "rejected": "r" * 16,
        }  # 5+16=21 > 20
        result = drop_long_rl_seq(
            sample, "dpo", self.tokenizer, self.sequence_len, handling="drop"
        )
        self.assertFalse(result)

    def test_dpo_truncate_mode_no_truncation_needed(self):
        """Test DPO truncate mode when no truncation is needed."""
        sample = {
            "prompt": "p" * 5,
            "chosen": "c" * 7,
            "rejected": "r" * 6,
        }  # 5+7=12 <= 20, 5+6=11 <= 20
        original_sample = sample.copy()
        result = drop_long_rl_seq(
            sample, "dpo", self.tokenizer, self.sequence_len, handling="truncate"
        )
        self.assertEqual(
            result, original_sample
        )  # Should return the original sample unchanged

    def test_dpo_truncate_mode_prompt_too_long(self):
        """Test DPO truncate mode when the prompt itself is too long."""
        sample = {"prompt": "p" * 25, "chosen": "c" * 7, "rejected": "r" * 6}
        original_sample = sample.copy()
        result = drop_long_rl_seq(
            sample, "dpo", self.tokenizer, self.sequence_len, handling="truncate"
        )
        # Even though truncation isn't possible, the function should return the original sample
        # for the map operation, assuming downstream filtering will catch it.
        self.assertEqual(result, original_sample)

    def test_dpo_truncate_mode_chosen_truncated(self):
        """Test DPO truncate mode when only 'chosen' needs truncation."""
        prompt_len = 5
        max_resp_len = self.sequence_len - prompt_len  # 20 - 5 = 15
        sample = {
            "prompt": "p" * prompt_len,
            "chosen": "c" * 18,
            "rejected": "r" * 10,
        }  # 5+18=23 > 20, 5+10=15 <= 20
        result = drop_long_rl_seq(
            sample, "dpo", self.tokenizer, self.sequence_len, handling="truncate"
        )
        self.assertEqual(len(result["prompt"]), prompt_len)
        self.assertEqual(len(result["chosen"]), max_resp_len)  # Truncated to 15
        self.assertEqual(
            result["chosen"], "x" * max_resp_len
        )  # Check decoded truncated value
        self.assertEqual(len(result["rejected"]), 10)  # Unchanged

    def test_dpo_truncate_mode_rejected_truncated(self):
        """Test DPO truncate mode when only 'rejected' needs truncation."""
        prompt_len = 5
        max_resp_len = self.sequence_len - prompt_len  # 15
        sample = {
            "prompt": "p" * prompt_len,
            "chosen": "c" * 10,
            "rejected": "r" * 18,
        }  # 5+10=15 <= 20, 5+18=23 > 20
        result = drop_long_rl_seq(
            sample, "dpo", self.tokenizer, self.sequence_len, handling="truncate"
        )
        self.assertEqual(len(result["prompt"]), prompt_len)
        self.assertEqual(len(result["chosen"]), 10)  # Unchanged
        self.assertEqual(len(result["rejected"]), max_resp_len)  # Truncated to 15
        self.assertEqual(
            result["rejected"], "x" * max_resp_len
        )  # Check decoded truncated value

    def test_dpo_truncate_mode_both_truncated(self):
        """Test DPO truncate mode when both 'chosen' and 'rejected' need truncation."""
        prompt_len = 8
        max_resp_len = self.sequence_len - prompt_len  # 20 - 8 = 12
        sample = {
            "prompt": "p" * prompt_len,
            "chosen": "c" * 15,
            "rejected": "r" * 14,
        }  # 8+15=23 > 20, 8+14=22 > 20
        result = drop_long_rl_seq(
            sample, "dpo", self.tokenizer, self.sequence_len, handling="truncate"
        )
        self.assertEqual(len(result["prompt"]), prompt_len)
        self.assertEqual(len(result["chosen"]), max_resp_len)  # Truncated to 12
        self.assertEqual(result["chosen"], "x" * max_resp_len)
        self.assertEqual(len(result["rejected"]), max_resp_len)  # Truncated to 12
        self.assertEqual(result["rejected"], "x" * max_resp_len)

    def test_dpo_truncate_mode_no_truncation_needed_but_long(self):
        """Test DPO truncate mode where individual parts fit but combined don't, but no truncation happens."""
        # This tests the case where len(chosen) <= max_resp_len and len(rejected) <= max_resp_len
        # but the initial check failed because e.g. prompt + chosen > sequence_len
        # The current logic *will* truncate if len(chosen) > max_resp_len.
        # Let's test a case where one is slightly too long causing the initial fail,
        # but the other fits *within* the max_response_len, so only one gets truncated.
        prompt_len = 10
        max_resp_len = self.sequence_len - prompt_len  # 10
        sample = {
            "prompt": "p" * prompt_len,
            "chosen": "c" * 11,
            "rejected": "r" * 9,
        }  # 10+11=21 > 20, 10+9=19 <= 20
        result = drop_long_rl_seq(
            sample, "dpo", self.tokenizer, self.sequence_len, handling="truncate"
        )
        self.assertEqual(len(result["prompt"]), prompt_len)
        self.assertEqual(len(result["chosen"]), max_resp_len)  # Truncated to 10
        self.assertEqual(result["chosen"], "x" * max_resp_len)
        self.assertEqual(len(result["rejected"]), 9)  # Unchanged, as 9 <= 10

    # Add similar tests for KTO if needed, checking prompt + completion length

    def test_kto_drop_mode_valid(self):
        """Test KTO drop mode with a valid sample."""
        sample = {"prompt": "p" * 5, "completion": "c" * 14}  # 5+14=19 <= 20
        result = drop_long_rl_seq(
            sample, "kto", self.tokenizer, self.sequence_len, handling="drop"
        )
        self.assertTrue(result)

    def test_kto_drop_mode_invalid(self):
        """Test KTO drop mode with an invalid sample."""
        sample = {"prompt": "p" * 5, "completion": "c" * 16}  # 5+16=21 > 20
        result = drop_long_rl_seq(
            sample, "kto", self.tokenizer, self.sequence_len, handling="drop"
        )
        self.assertFalse(result)

    def test_kto_truncate_mode_no_truncation_needed(self):
        """Test KTO truncate mode when no truncation is needed."""
        sample = {"prompt": "p" * 5, "completion": "c" * 14}  # 5+14=19 <= 20
        original_sample = sample.copy()
        result = drop_long_rl_seq(
            sample, "kto", self.tokenizer, self.sequence_len, handling="truncate"
        )
        self.assertEqual(result, original_sample)

    def test_kto_truncate_mode_prompt_too_long(self):
        """Test KTO truncate mode when the prompt itself is too long."""
        sample = {"prompt": "p" * 25, "completion": "c" * 7}
        original_sample = sample.copy()
        result = drop_long_rl_seq(
            sample, "kto", self.tokenizer, self.sequence_len, handling="truncate"
        )
        self.assertEqual(result, original_sample)  # Returns original sample

    def test_kto_truncate_mode_completion_truncated(self):
        """Test KTO truncate mode when completion needs truncation."""
        prompt_len = 8
        max_comp_len = self.sequence_len - prompt_len  # 20 - 8 = 12
        sample = {"prompt": "p" * prompt_len, "completion": "c" * 15}  # 8+15=23 > 20
        result = drop_long_rl_seq(
            sample, "kto", self.tokenizer, self.sequence_len, handling="truncate"
        )
        self.assertEqual(len(result["prompt"]), prompt_len)
        self.assertEqual(len(result["completion"]), max_comp_len)  # Truncated to 12
        self.assertEqual(result["completion"], "x" * max_comp_len)

    def test_missing_keys_dpo(self):
        """Test ValueError raised if keys missing for DPO."""
        sample = {"prompt": "p"}
        with self.assertRaisesRegex(
            ValueError, "Prompt, chosen and rejected keys are required"
        ):
            drop_long_rl_seq(sample, "dpo", self.tokenizer, self.sequence_len)

    def test_missing_keys_kto(self):
        """Test ValueError raised if keys missing for KTO."""
        sample = {"prompt": "p"}
        with self.assertRaisesRegex(
            ValueError, "Prompt and completion keys are required"
        ):
            drop_long_rl_seq(sample, "kto", self.tokenizer, self.sequence_len)

    def test_unknown_rl_type(self):
        """Test ValueError raised for unknown RL type."""
        sample = {}
        with self.assertRaisesRegex(ValueError, "Unknown RL type"):
            drop_long_rl_seq(sample, "xyz", self.tokenizer, self.sequence_len)

    # GRPO test - current implementation always passes
    def test_grpo_drop(self):
        """Test GRPO drop mode (currently always True)."""
        sample = {}
        result = drop_long_rl_seq(
            sample, "grpo", self.tokenizer, self.sequence_len, handling="drop"
        )
        self.assertTrue(result)

    def test_grpo_truncate(self):
        """Test GRPO truncate mode (currently returns original sample)."""
        sample = {"a": 1}
        result = drop_long_rl_seq(
            sample, "grpo", self.tokenizer, self.sequence_len, handling="truncate"
        )
        self.assertEqual(result, sample)


if __name__ == "__main__":
    unittest.main()
