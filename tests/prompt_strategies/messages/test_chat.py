"""
tests for chat_template prompt strategy
"""

# pylint: disable=duplicate-code
import logging
import unittest

from axolotl.prompt_strategies.messages.chat import load
from axolotl.utils.dict import DictDefault

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger("axolotl")


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
        assert (
            input_ids == expected_input_ids
        ), f"Input IDs mismatch: {input_ids} != {expected_input_ids}"


if __name__ == "__main__":
    unittest.main()
