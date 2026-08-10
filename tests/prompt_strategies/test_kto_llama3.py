"""
Tests for KTO dataset transform strategies with llama-3 formatting
"""

from axolotl.prompt_strategies.kto.llama3 import argilla_chat


class TestKTOLlama3:
    """
    Test kto.llama3 transforms
    """

    def test_argilla_chat_single_turn(self):
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
        assert sample["prompt"] == (
            "<|start_header_id|>user<|end_header_id|>\n\nWhat is 2 + 2?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        assert sample["completion"] == "4<|eot_id|>"

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
            "<|start_header_id|>user<|end_header_id|>\n\nWhat is 2 + 2?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n4<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\nAnd times 3?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        assert sample["completion"] == "12<|eot_id|>"
