"""
tests for the ORPO chat_template.argilla transform
"""

from axolotl.prompt_strategies.orpo.chat_template import argilla
from axolotl.utils.dict import DictDefault

SAMPLE = {
    "chosen": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "goodbye"},
    ],
    "rejected": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "party on"},
    ],
}


class TestORPOArgillaChatTemplate:
    """
    The dataset-level chat_template takes precedence over the top-level one, as it
    does for the DPO chat_template strategies.
    """

    def test_dataset_level_chat_template(self, llama3_tokenizer):
        cfg = DictDefault(
            {
                "chat_template": "chatml",
                "datasets": [
                    {"type": "chat_template.argilla", "chat_template": "llama3"}
                ],
            }
        )
        transform_fn = argilla(cfg, dataset_idx=0)
        result = transform_fn(dict(SAMPLE), tokenizer=llama3_tokenizer)
        assert result["prompt"] == (
            "<|begin_of_text|>"
            + "<|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        assert result["chosen"] == "goodbye<|eot_id|>"
        assert result["rejected"] == "party on<|eot_id|>"

    def test_top_level_chat_template(self, llama3_tokenizer):
        cfg = DictDefault(
            {
                "chat_template": "chatml",
                "datasets": [{"type": "chat_template.argilla"}],
            }
        )
        transform_fn = argilla(cfg, dataset_idx=0)
        result = transform_fn(dict(SAMPLE), tokenizer=llama3_tokenizer)
        assert (
            result["prompt"]
            == "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        )
        assert result["chosen"] == "goodbye<|im_end|>\n"
        assert result["rejected"] == "party on<|im_end|>\n"
