"""
Test module for raw i/o data for prompts
"""

import pytest
from datasets import Dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer

from axolotl.datasets import TokenizedPromptDataset
from axolotl.prompt_strategies.input_output import (
    RawInputOutputPrompter,
    RawInputOutputStrategy,
)


@pytest.fixture(name="segments_dataset")
def fixture_sharegpt_dataset():
    return Dataset.from_list(
        [
            {
                "segments": [
                    {
                        "label": False,
                        "text": "<s>hello ",
                    },
                    {
                        "label": True,
                        "text": "hi there.<eot>",
                    },
                    {
                        "label": False,
                        "text": "goodbye ",
                    },
                    {
                        "label": True,
                        "text": "farewell<eot>",
                    },
                ]
            }
        ]
    )


@pytest.fixture(name="tokenizer")
def fixture_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "casperhansen/mistral-7b-instruct-v0.1-awq"
    )
    tokenizer.add_tokens(
        [
            AddedToken("<eot>", rstrip=False, lstrip=False, normalized=False),
        ]
    )

    return tokenizer


class TestRawInputOutputPrompts:
    """
    Test class for raw i/o prompter
    """

    def test_segment_prompts(self, segments_dataset, tokenizer):
        strategy = RawInputOutputStrategy(
            RawInputOutputPrompter(),
            tokenizer,
            False,  # train_on_inputs
            2048,  # sequence_len
        )

        dataset_wrapper = TokenizedPromptDataset(
            strategy, segments_dataset, process_count=1
        )

        input_ids = dataset_wrapper[0]["input_ids"]
        labels = dataset_wrapper[0]["labels"]

        assert (
            tokenizer.decode(input_ids)
            == "<s> hello  hi there.<eot> goodbye  farewell<eot>"
        )
        # fmt: off
        assert input_ids == [
            1,  # <s>
            6312,  # hell
            28709,  # o
            28705,  #
            12014,  # hi
            736,  # there
            28723,  # .
            32000,  # <eot>
            1179,  # good
            17664,  # bye
            28705,  #
            19111,  # fare
            5458,  # well
            32000,  # <eot>
        ]
        # fmt: on

        # fmt: off
        assert labels == [
            -100,  # <s>
            -100,  # hell
            -100,  # o
            -100,  #
            12014,  # hi
            736,  # there
            28723,  # .
            32000,  # <eot>
            -100,  # good
            -100,  # bye
            -100,  #
            19111,  # fare
            5458,  # well
            32000,  # <eot>
        ]
        # fmt: on
