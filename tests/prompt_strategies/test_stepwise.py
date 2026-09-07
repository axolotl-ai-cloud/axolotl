"""
tests for chat_template prompt strategy
"""

import pytest
from datasets import Dataset

from axolotl.prompt_strategies.stepwise_supervised import (
    StepwiseSupervisedPromptTokenizingStrategy,
)


class TestStepWiseSupervisedPromptTokenizingStrategy:
    """
    Test class for stepwise supervised prompt strategy
    """

    @pytest.fixture()
    def stepwise_supervised_dataset(self):
        return Dataset.from_list(
            [
                {
                    "prompt": "Which number is larger, 9.8 or 9.11?",
                    "completions": [
                        "The fractional part of 9.8 is 0.8, while the fractional part of 9.11 is 0.11.",
                        "Since 0.11 is greater than 0.8, the number 9.11 is larger than 9.8.",
                        "Actually, this is incorrect. In decimal numbers, 0.8 is equal to 0.80, which is larger than 0.11. Therefore, 9.8 is larger than 9.11.",
                    ],
                    "labels": [True, False, False],
                }
            ]
        )

    def test_stepwise_supervised_dataset(
        self, qwen3_tokenizer, stepwise_supervised_dataset
    ):
        strategy = StepwiseSupervisedPromptTokenizingStrategy(
            qwen3_tokenizer,
            sequence_len=2048,
            step_separator="\n",
        )
        sample = stepwise_supervised_dataset[0]
        labels = strategy.tokenize_prompt(sample)["labels"]
        prompt_len = len(
            qwen3_tokenizer(sample["prompt"], add_special_tokens=False)["input_ids"]
        )
        if qwen3_tokenizer.bos_token_id is not None:
            prompt_len += 1

        separator_len = len(qwen3_tokenizer.encode("\n", add_special_tokens=False))
        completion_lengths = [
            len(qwen3_tokenizer(completion, add_special_tokens=False)["input_ids"])
            + separator_len
            for completion in sample["completions"]
        ]
        expected = [-100] * prompt_len
        for completion_len, label in zip(
            completion_lengths, sample["labels"], strict=False
        ):
            expected.extend([-100] * (completion_len - 1))
            expected.append(int(label))

        assert labels == expected
