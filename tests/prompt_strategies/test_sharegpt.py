"""
Test module for sharegpt integration w chatml
"""

import pytest
from datasets import Dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer

from axolotl.datasets import TokenizedPromptDataset
from axolotl.prompt_strategies.sharegpt import (
    GlaiveShareGPTPromptTokenizingStrategy,
    SimpleShareGPTPromptTokenizingStrategy,
    register_chatml_template,
)
from axolotl.prompters import ShareGPTPrompterV2

register_chatml_template()


@pytest.fixture(name="sharegpt_dataset")
def fixture_sharegpt_dataset():
    return Dataset.from_list(
        [
            {
                "conversations": [
                    {
                        "from": "system",
                        "value": "repeat",
                    },
                    {
                        "from": "human",
                        "value": "hello",
                    },
                    {
                        "from": "gpt",
                        "value": "hello",
                    },
                    {
                        "from": "human",
                        "value": "goodbye",
                    },
                    {
                        "from": "gpt",
                        "value": "goodbye",
                    },
                ]
            }
        ]
    )


@pytest.fixture(name="glaive_dataset")
def fixture_sharegpt_glaive_dataset():
    return Dataset.from_list(
        [
            {
                "system": "SYSTEM: This is a system prompt",
                "chat": "USER: Can you book a flight for me from New York to London? ASSISTANT: I'm sorry, but I don't have the capability to book flights.  <|endoftext|>",
            }
        ]
    )


@pytest.fixture(name="multi_role_dataset")
def fixture_multi_role_dataset():
    return Dataset.from_list(
        [
            {
                "conversations": [
                    {
                        "from": "system",
                        "value": "use get_weather(city) to get the weather for a city",
                    },
                    {
                        "from": "human",
                        "value": "hello, what's the weather in New York?",
                    },
                    {
                        "from": "gpt",
                        "value": "let me get that for you",
                    },
                    {
                        "from": "tool",
                        "value": "get_weather(New York)",
                    },
                    {
                        "from": "gpt",
                        "value": "the weather in New York is 70 degrees and sunny",
                    },
                ]
            }
        ]
    )


@pytest.fixture(name="tokenizer")
def fixture_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
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


class TestSharegpt:
    """
    Test class for sharegpt prompter
    """

    def test_no_double_im_end(self, sharegpt_dataset, tokenizer):
        strategy = SimpleShareGPTPromptTokenizingStrategy(
            ShareGPTPrompterV2(
                conversation="chatml",
                role_key_model=None,
                role_key_human=None,
            ),
            tokenizer,
            False,  # train_on_inputs
            2048,  # sequence_len
        )

        dataset_wrapper = TokenizedPromptDataset(
            strategy, sharegpt_dataset, process_count=1
        )

        input_ids = dataset_wrapper[0]["input_ids"]
        # fmt: off
        assert input_ids == [
            #  28705, 13, is " \n"
            1,   # bos
            32001, 1587, 13, 25997, 32000, 28705, 13,  # system
            32001, 2188, 13, 21558, 32000, 28705, 13,  # human
            32001, 13892, 13, 21558, 32000, 28705, 13,  # gpt
            32001, 2188, 13, 12684, 17664, 32000, 28705, 13,   # human
            32001, 13892, 13, 12684, 17664, 32000, 28705, 13,  # gpt
        ]
        # fmt: on

    def test_w_train_on_input(self, sharegpt_dataset, tokenizer):
        strategy = SimpleShareGPTPromptTokenizingStrategy(
            ShareGPTPrompterV2(
                conversation="chatml",
                role_key_model=None,
                role_key_human=None,
            ),
            tokenizer,
            False,  # train_on_inputs
            2048,  # sequence_len
        )

        dataset_wrapper = TokenizedPromptDataset(
            strategy, sharegpt_dataset, process_count=1
        )

        labels = dataset_wrapper[0]["labels"]
        # fmt: off
        assert labels == [
            -100,   # bos
            -100, -100, -100, -100, -100, -100, -100,  # system
            -100, -100, -100, -100, -100, -100, -100,  # human
            -100, -100, 13, 21558, 32000, 28705, 13,  # gpt
            -100, -100, -100, -100, -100, -100, -100, -100,   # human
            -100, -100, 13, 12684, 17664, 32000, 28705, 13,  # gpt
        ]
        # fmt: on

    def test_no_train_on_input(self, sharegpt_dataset, tokenizer):
        strategy = SimpleShareGPTPromptTokenizingStrategy(
            ShareGPTPrompterV2(
                conversation="chatml",
                role_key_model=None,
                role_key_human=None,
            ),
            tokenizer,
            True,  # train_on_inputs
            2048,  # sequence_len
        )

        dataset_wrapper = TokenizedPromptDataset(
            strategy, sharegpt_dataset, process_count=1
        )

        labels = dataset_wrapper[0]["labels"]
        # fmt: off
        assert labels == [
            1,   # bos
            32001, 1587, 13, 25997, 32000, 28705, 13,  # system
            32001, 2188, 13, 21558, 32000, 28705, 13,  # human
            32001, 13892, 13, 21558, 32000, 28705, 13,  # gpt
            32001, 2188, 13, 12684, 17664, 32000, 28705, 13,   # human
            32001, 13892, 13, 12684, 17664, 32000, 28705, 13,  # gpt
        ]
        # fmt: on

    def test_chatml_glaive(self, glaive_dataset, tokenizer):
        strategy = GlaiveShareGPTPromptTokenizingStrategy(
            ShareGPTPrompterV2(
                conversation="chatml",
                role_key_model=None,
                role_key_human=None,
            ),
            tokenizer,
            True,  # train_on_inputs
            2048,  # sequence_len
        )

        dataset_wrapper = TokenizedPromptDataset(
            strategy, glaive_dataset, process_count=1
        )

        labels = dataset_wrapper[0]["labels"]
        # fmt: off
        assert labels == [
            1,  # bos
            32001, 1587, 13, 3260, 349, 264, 1587, 11510, 32000, 28705, 13,  # system
            32001, 2188, 13, 6325, 368, 1820, 264, 9314, 354, 528, 477, 1450, 2726, 298, 4222, 28804, 32000, 28705, 13,  # human
            32001, 13892, 13, 28737, 28742, 28719, 7371, 28725, 562, 315, 949, 28742, 28707, 506, 272, 21368, 298, 1820, 22447, 28723, 28705, 523, 28766, 416, 1009, 772, 28766, 28767, 32000, 28705, 13  # gpt
        ]
        # fmt: on

    def test_multi_role_dataset(self, multi_role_dataset, tokenizer):
        strategy = SimpleShareGPTPromptTokenizingStrategy(
            ShareGPTPrompterV2(conversation="chatml", roles={"input": ["tool"]}),
            tokenizer,
            False,  # train_on_inputs
            2048,  # sequence_len
        )

        dataset_wrapper = TokenizedPromptDataset(
            strategy, multi_role_dataset, process_count=1
        )

        input_ids = dataset_wrapper[0]["input_ids"]
        # fmt: off
        assert input_ids == [
            1,   # bos
            32001, 1587, 13, 1730, 625, 28730, 769, 1223, 28732, 18373, 28731, 298, 625, 272, 8086, 354, 264, 2990, 32000, 28705, 13,  # system
            32001, 2188, 13, 21558, 28725, 767, 28742, 28713, 272, 8086, 297, 1450, 2726, 28804, 32000, 28705, 13,  # human
            32001, 13892, 13, 895, 528, 625, 369, 354, 368, 32000, 28705, 13,  # gpt
            32001, 3921, 13, 527, 28730, 769, 1223, 28732, 2972, 2726, 28731, 32000, 28705, 13,  # tool
            32001, 13892, 13, 1237, 8086, 297, 1450, 2726, 349, 28705, 28787, 28734, 11182, 304, 4376, 1780, 32000, 28705, 13  # gpt
        ]
        # fmt: on

        labels = dataset_wrapper[0]["labels"]
        # fmt: off
        assert labels == [
            -100,  # bos
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,  # system
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,  # human
            -100, -100, 13, 895, 528, 625, 369, 354, 368, 32000, 28705, 13,  # gpt
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,  # tool
            -100, -100, 13, 1237, 8086, 297, 1450, 2726, 349, 28705, 28787, 28734, 11182, 304, 4376, 1780, 32000, 28705, 13  # gpt
        ]
        # fmt: on
