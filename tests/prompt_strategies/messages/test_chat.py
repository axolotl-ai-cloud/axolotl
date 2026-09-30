"""
tests for chat_template prompt strategy
"""

import unittest

import pytest

from axolotl.prompt_strategies.messages.chat import load
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__, log_level="DEBUG")


class TestMessagesChatLlama3:
    """
    Test class for assistant style datasets with llama-3 prompts using the messages chat llama3 strategy.
    """

    def test_llama3_load(self, llama3_tokenizer, assistant_dataset):
        LOG.info("Loading llama-3 tokenizer with assistant dataset")
        strategy = load(
            llama3_tokenizer,
            DictDefault(
                {
                    "train_on_inputs": False,
                    "sequence_len": 512,
                }
            ),
            DictDefault(
                {
                    "chat_template": "llama3",
                    "message_field_role": "role",
                    "message_field_content": "content",
                    "field_messages": "messages",
                }
            ),
        )
        res = strategy.wrap_dataset(assistant_dataset)
        input_ids = res[0]["input_ids"]
        # fmt: off
        expected_input_ids = [
            128000,  # bos
            128006, 882, 128007,  # user header
            271, 15339, 128009,  # user prompt eot
            128006, 78191, 128007,  # assistant header
            271, 15339, 128009,  # assistant response eot
            128006, 882, 128007,
            271, 19045, 29474, 128009,
            128006, 78191, 128007,
            271, 19045, 29474, 128009,
        ]
        # fmt: on
        LOG.debug(f"Expected input_ids: {expected_input_ids}")
        LOG.debug(f"Actual input_ids: {input_ids}")
        assert input_ids == expected_input_ids, (
            f"Input IDs mismatch: {input_ids} != {expected_input_ids}"
        )


class TestMessagesChatTrainingField:
    """
    `message_field_training` from the dataset config must select the per-message weight.
    """

    @pytest.mark.parametrize("field", ["weight", "train", "training"])
    def test_configured_training_field_sets_weights(self, field):
        strategy = load(
            None,
            DictDefault({"train_on_inputs": False, "sequence_len": 512}),
            DictDefault(
                {
                    "chat_template": "chatml",
                    "field_messages": "messages",
                    "message_field_training": field,
                }
            ),
        )
        sample = {
            "messages": [
                {"role": "user", "content": "hello", field: 1},
                {"role": "assistant", "content": "bad answer", field: 0},
            ]
        }

        conversation = strategy.message_transform(sample)["conversation"]

        assert [msg["weight"] for msg in conversation] == [1, 0]


if __name__ == "__main__":
    unittest.main()
