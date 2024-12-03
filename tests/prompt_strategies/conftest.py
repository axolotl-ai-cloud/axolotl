"""
shared fixtures for prompt strategies tests
"""

import pytest
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


@pytest.fixture(name="assistant_dataset")
def fixture_assistant_dataset():
    return Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "goodbye"},
                    {"role": "assistant", "content": "goodbye"},
                ]
            }
        ]
    )


@pytest.fixture(name="sharegpt_dataset")
def fixture_sharegpt_dataset():
    # pylint: disable=duplicate-code
    return Dataset.from_list(
        [
            {
                "conversations": [
                    {"from": "human", "value": "hello"},
                    {"from": "gpt", "value": "hello"},
                    {"from": "human", "value": "goodbye"},
                    {"from": "gpt", "value": "goodbye"},
                ]
            }
        ]
    )


@pytest.fixture(name="basic_dataset")
def fixture_basic_dataset():
    # pylint: disable=duplicate-code
    return Dataset.from_list(
        [
            {
                "conversations": [
                    {"from": "system", "value": "You are an AI assistant."},
                    {"from": "human", "value": "Hello"},
                    {"from": "assistant", "value": "Hi there!"},
                    {"from": "human", "value": "How are you?"},
                    {"from": "assistant", "value": "I'm doing well, thank you!"},
                ]
            }
        ]
    )


@pytest.fixture(name="llama3_tokenizer")
def fixture_llama3_tokenizer():
    hf_hub_download(
        repo_id="NousResearch/Meta-Llama-3-8B-Instruct",
        filename="special_tokens_map.json",
    )
    hf_hub_download(
        repo_id="NousResearch/Meta-Llama-3-8B-Instruct",
        filename="tokenizer_config.json",
    )
    hf_hub_download(
        repo_id="NousResearch/Meta-Llama-3-8B-Instruct", filename="tokenizer.json"
    )
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")

    return tokenizer


@pytest.fixture(name="phi35_tokenizer")
def fixture_phi35_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    return tokenizer
