"""
tests for chat_template prompt strategy
"""

import datasets
import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from axolotl.datasets import TokenizedPromptDataset
from axolotl.prompt_strategies.stepwise_supervised import (
    StepwiseSupervisedPromptTokenizingStrategy,
)


class TestStepWiseSupervisedPromptTokenizingStrategy:
    """
    Test class for stepwise supervised prompt strategy
    """

    @pytest.fixture()
    def stepwise_supervised_dataset(self):
        # pylint: disable=duplicate-code
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

    @pytest.fixture()
    def tokenizer(self):
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    def test_stepwise_supervised_dataset(self, tokenizer, stepwise_supervised_dataset):
        strategy = StepwiseSupervisedPromptTokenizingStrategy(
            tokenizer,
            sequence_len=2048,
            step_separator="\n",
        )
        stepwise_supervised_dataset = stepwise_supervised_dataset.cast_column(
            "labels", datasets.Sequence(datasets.Value("int64"))
        )
        dataset_wrapper = TokenizedPromptDataset(
            strategy,
            stepwise_supervised_dataset,
            process_count=1,
        )
        labels = dataset_wrapper[0]["labels"]
        # expected labels is:
        # the prompt + first step are ignored, followed by the label for step 1 (True)
        # the second step, and its label (False)
        # the third step, and its label (False)
        expected = [-100] * 47 + [1] + [-100] * 29 + [0] + [-100] * 48 + [0]

        assert labels == expected
