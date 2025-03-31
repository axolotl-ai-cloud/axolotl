"""
Test module for alpaca integration w chatml
"""

import pytest
from datasets import Dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer

from axolotl.datasets import TokenizedPromptDataset
from axolotl.prompt_tokenizers import AlpacaPromptTokenizingStrategy
from axolotl.prompters import AlpacaPrompter, PromptStyle

from tests.hf_offline_utils import enable_hf_offline


@pytest.fixture(name="alpaca_dataset")
def fixture_alpaca_dataset():
    return Dataset.from_list(
        [
            {
                "instruction": "Evaluate this sentence for spelling and grammar mistakes",
                "input": "He finnished his meal and left the resturant",
                "output": "He finished his meal and left the restaurant.",
            }
        ]
    )


@pytest.fixture(name="tokenizer")
@enable_hf_offline
def fixture_tokenizer():
    # pylint: disable=all
    tokenizer = AutoTokenizer.from_pretrained(
        "casperhansen/mistral-7b-instruct-v0.1-awq"
    )
    tokenizer.add_special_tokens(
        {
            "eos_token": AddedToken(
                "<|im_end|>", rstrip=False, lstrip=False, normalized=False
            )
        }
    )
    tokenizer.add_tokens(
        [
            AddedToken("<|im_start|>", rstrip=False, lstrip=False, normalized=False),
        ]
    )

    return tokenizer


class TestAlpacaChatml:
    """
    Test class for alpaca prompter
    """

    def test_no_double_im_end(self, alpaca_dataset, tokenizer):
        strategy = AlpacaPromptTokenizingStrategy(
            AlpacaPrompter(prompt_style=PromptStyle.CHATML.value),
            tokenizer,
            False,  # train_on_inputs
            2048,  # sequence_len
        )

        dataset_wrapper = TokenizedPromptDataset(
            strategy, alpaca_dataset, process_count=1
        )

        input_ids = dataset_wrapper[0]["input_ids"]
        # fmt: off
        assert input_ids == [
            1,  # Bos
            32001, 1587, 13, 20548, 336, 349, 396, 13126, 369, 13966, 264, 3638, 28725, 5881, 1360, 395, 396, 2787, 369, 5312, 3629, 2758, 28723, 12018, 264, 2899, 369, 6582, 1999, 2691, 274, 272, 2159, 28723, 32000, 28705, 13,  # instruction
            32001, 2188, 13, 16627, 11931, 456, 12271, 354, 668, 3572, 304, 18756, 3479, 17179, 13, 2428, 854, 28711, 1497, 516, 11314, 304, 1749, 272, 1846, 324, 440, 32000, 28705, 13,  # input
            32001, 13892, 13, 650, 5967, 516, 11314, 304, 1749, 272, 9926, 28723, 32000,  # output
        ]
        # fmt: on

    def test_no_train_on_input(self, alpaca_dataset, tokenizer):
        strategy = AlpacaPromptTokenizingStrategy(
            AlpacaPrompter(prompt_style=PromptStyle.CHATML.value),
            tokenizer,
            False,  # train_on_inputs
            2048,  # sequence_len
        )

        dataset_wrapper = TokenizedPromptDataset(
            strategy, alpaca_dataset, process_count=1
        )

        labels = dataset_wrapper[0]["labels"]
        # fmt: off
        assert labels == [
            -100,  # bos
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,  # instruction
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,  # input
            -100, -100, -100, 650, 5967, 516, 11314, 304, 1749, 272, 9926, 28723, 32000,  # Output
        ]
        # fmt: on

    def test_w_train_on_input(self, alpaca_dataset, tokenizer):
        strategy = AlpacaPromptTokenizingStrategy(
            AlpacaPrompter(prompt_style=PromptStyle.CHATML.value),
            tokenizer,
            True,  # train_on_inputs
            2048,  # sequence_len
        )

        dataset_wrapper = TokenizedPromptDataset(
            strategy, alpaca_dataset, process_count=1
        )

        labels = dataset_wrapper[0]["labels"]
        # fmt: off
        assert labels == [
            1,  # Bos
            32001, 1587, 13, 20548, 336, 349, 396, 13126, 369, 13966, 264, 3638, 28725, 5881, 1360, 395, 396, 2787, 369, 5312, 3629, 2758, 28723, 12018, 264, 2899, 369, 6582, 1999, 2691, 274, 272, 2159, 28723, 32000, 28705, 13,  # instruction
            32001, 2188, 13, 16627, 11931, 456, 12271, 354, 668, 3572, 304, 18756, 3479, 17179, 13, 2428, 854, 28711, 1497, 516, 11314, 304, 1749, 272, 1846, 324, 440, 32000, 28705, 13,  # input
            32001, 13892, 13, 650, 5967, 516, 11314, 304, 1749, 272, 9926, 28723, 32000,  # output
        ]
        # fmt: on
