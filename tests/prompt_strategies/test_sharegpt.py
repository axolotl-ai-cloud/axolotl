"""
Test module for sharegpt integration w chatml
"""
import pytest
from datasets import Dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer

from axolotl.datasets import TokenizedPromptDataset
from axolotl.prompt_strategies.sharegpt import SimpleShareGPTPromptTokenizingStrategy
from axolotl.prompters import ShareGPTPrompterV2


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


@pytest.fixture(name="tokenizer")
def fixture_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.add_special_tokens(
        {
            "eos_token": AddedToken(
                "<|im_end|>", rstrip=True, lstrip=False, normalized=False
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

    def test_something(self, sharegpt_dataset, tokenizer):
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
            1,   # bos
            32001, 1587, 13, 25997, 32000,  # system
            32001, 2188, 13, 21558, 32000,  # human
            32001, 13892, 13, 21558, 32000,  # gpt
            32001, 2188, 13, 12684, 17664, 32000,   # human
            32001, 13892, 13, 12684, 17664, 32000  # gpt
        ]
        # fmt: on
