"""
Tests for KTO dataset transform strategies with chatml formatting
"""

from axolotl.prompt_strategies.kto.chatml import argilla_chat


class TestKTOChatml:
    """
    Test kto.chatml transforms
    """

    def test_argilla_chat_reads_completion_messages(self):
        """argilla_chat builds the prompt from the conversation stored in the
        `completion` column. argilla/kto-mix-15k has no `chosen` column, so
        reading one raised KeyError during preprocessing; the llama-3 sibling
        already reads `completion`."""
        transform_fn = argilla_chat(cfg=None)
        sample = transform_fn(
            {
                "prompt": "What is 2 + 2?",
                "completion": [
                    {"role": "user", "content": "What is 2 + 2?"},
                    {"role": "assistant", "content": "4"},
                ],
                "label": True,
            }
        )
        assert (
            sample["prompt"]
            == "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n"
        )
        assert sample["completion"] == "4<|im_end|>"

    def test_argilla_chat_keeps_multi_turn_history(self):
        """argilla/kto-mix-15k is not uniformly single-turn, so earlier turns
        must land in the prompt rather than being dropped."""
        transform_fn = argilla_chat(cfg=None)
        sample = transform_fn(
            {
                "prompt": "What is 2 + 2?",
                "completion": [
                    {"role": "user", "content": "What is 2 + 2?"},
                    {"role": "assistant", "content": "4"},
                    {"role": "user", "content": "And times 3?"},
                    {"role": "assistant", "content": "12"},
                ],
                "label": True,
            }
        )
        assert sample["prompt"] == (
            "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n"
            "<|im_start|>assistant\n4<|im_end|>\n"
            "<|im_start|>user\nAnd times 3?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        assert sample["completion"] == "12<|im_end|>"
