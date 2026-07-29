"""Tests for the multimodal pre-tokenization cache path."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from datasets import Dataset

from axolotl.datasets import TokenizedPromptDataset
from axolotl.prompt_strategies.chat_template import (
    ChatTemplatePrompter,
    ChatTemplateStrategy,
)
from axolotl.prompt_tokenizers import DatasetWrappingStrategy, PromptTokenizingStrategy
from axolotl.utils.data.wrappers import get_dataset_wrapper
from axolotl.utils.dict import DictDefault

CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['content'] }}{{ eos_token }}{% endfor %}"
)


class FakeTokenizer:
    """Minimal tokenizer stub for ChatTemplateStrategy construction."""

    eos_token = "</s>"
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [2]

    def apply_chat_template(self, conversation, **kwargs):  # noqa: ARG002
        return [1, 2, 3]


class FakeProcessor:
    """Processor stub returning tensors shaped like a Qwen-VL style output."""

    def apply_chat_template(self, conversation, tokenize=False, **kwargs):  # noqa: ARG002
        return "user: hello"

    def __call__(self, text=None, images=None, return_tensors="pt"):  # noqa: ARG002
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }
        if images:
            batch["pixel_values"] = torch.zeros(6, 8, dtype=torch.float32)
            batch["image_grid_thw"] = torch.tensor([[1, 2, 3]])
        return batch


def make_strategy(processor=None):
    tokenizer = FakeTokenizer()
    prompter = ChatTemplatePrompter(
        tokenizer,
        chat_template=CHAT_TEMPLATE,
        processor=processor,
    )
    return ChatTemplateStrategy(
        prompter,
        tokenizer,
        train_on_inputs=True,
        sequence_len=512,
    )


class TestBuildPromptPixelValues:
    """build_prompt must not box pixel_values into Python floats."""

    def test_pixel_values_stay_numpy_float32(self):
        prompter = make_strategy(processor=FakeProcessor()).prompter
        out = prompter.build_prompt(
            [{"role": "user", "content": "hello"}], images=["img"]
        )

        assert isinstance(out["pixel_values"], np.ndarray)
        assert out["pixel_values"].dtype == np.float32
        assert out["pixel_values"].shape == (6, 8)
        assert out["input_ids"] == [1, 2, 3, 4]
        assert out["image_grid_thw"] == [1, 2, 3]

    def test_arrow_cache_stores_float32(self, tmp_path):
        strategy = make_strategy(processor=FakeProcessor())
        ds = Dataset.from_dict(
            {"messages": [[{"role": "user", "content": "hello"}]], "images": [["x"]]}
        )
        tokenized = ds.map(
            strategy.tokenize_prompt,
            remove_columns=ds.column_names,
            cache_file_name=str(tmp_path / "cache.arrow"),
        )
        assert "float32" in str(tokenized.features["pixel_values"])


class TestBatchedRaggedColumns:
    """Batched tokenize_prompt must emit equal-length columns for Arrow."""

    def make_mixed_strategy(self):
        strategy = make_strategy()

        def fake_single(prompt):
            row = {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]}
            if prompt.get("images"):
                row["pixel_values"] = [[0.5, 0.5]]
            return row

        strategy._tokenize_single_prompt = fake_single  # pylint: disable=protected-access
        return strategy

    def test_missing_keys_padded_with_none(self):
        strategy = self.make_mixed_strategy()
        out = strategy.tokenize_prompt(
            {
                "messages": [["a"], ["b"], ["c"]],
                "images": [["img"], [], ["img"]],
            }
        )
        assert out["pixel_values"] == [[[0.5, 0.5]], None, [[0.5, 0.5]]]
        assert len(out["input_ids"]) == 3

    def test_mixed_batch_maps_without_arrow_error(self):
        strategy = self.make_mixed_strategy()
        ds = Dataset.from_dict(
            {
                "messages": [["a"], ["b"], ["c"], ["d"]],
                "images": [["img"], [], [], ["img"]],
            }
        )
        tokenized = ds.map(
            strategy.tokenize_prompt,
            batched=True,
            batch_size=4,
            remove_columns=ds.column_names,
        )
        assert len(tokenized) == 4
        assert tokenized[1]["pixel_values"] is None
        assert tokenized[3]["pixel_values"] == [[0.5, 0.5]]

    def test_empty_rows_still_dropped(self):
        strategy = make_strategy()
        strategy._tokenize_single_prompt = (  # pylint: disable=protected-access
            lambda prompt: {"input_ids": [1]} if prompt.get("images") else {}
        )
        out = strategy.tokenize_prompt(
            {"messages": [["a"], ["b"]], "images": [["img"], []]}
        )
        assert out == {"input_ids": [[1]]}


class SimpleStrategy(PromptTokenizingStrategy):
    """Batched no-op strategy for buffer-size plumbing tests."""

    @property
    def supports_batched(self):
        return True

    def tokenize_prompt(self, prompt):
        return {"input_ids": [[1, 2]] * len(prompt["text"])}


class TestMapBufferPlumbing:
    """batch_size / writer_batch_size must reach Dataset.map."""

    def capture_map_kwargs(self, **dataset_kwargs):
        strategy = SimpleStrategy(None, FakeTokenizer(), False, 512)
        ds = Dataset.from_dict({"text": ["a", "b"]})
        captured = {}
        original_map = Dataset.map

        def spy(self, *args, **kwargs):
            captured.update(kwargs)
            return original_map(self, *args, **kwargs)

        with patch.object(Dataset, "map", spy):
            TokenizedPromptDataset(strategy, ds, **dataset_kwargs)
        return captured

    def test_defaults_unchanged(self):
        captured = self.capture_map_kwargs()
        assert captured["batch_size"] == 1000
        assert "writer_batch_size" not in captured

    def test_overrides_forwarded(self):
        captured = self.capture_map_kwargs(batch_size=4, writer_batch_size=8)
        assert captured["batch_size"] == 4
        assert captured["writer_batch_size"] == 8


class TestWrapperBufferDefaults:
    """get_dataset_wrapper picks small buffers for multimodal runs."""

    def capture_dataset_kwargs(self, cfg_overrides=None, processor=None):
        cfg = DictDefault(
            {
                "dataset_num_proc": 1,
                "dataset_keep_in_memory": None,
                "skip_prepare_dataset": False,
                **(cfg_overrides or {}),
            }
        )
        dataset_config = DictDefault({"path": "dummy", "type": "dummy_type"})
        dataset = Dataset.from_dict({"messages": [["hi"]]})
        captured = {}

        class FakeLoadedStrategy(DatasetWrappingStrategy):
            def wrap_dataset(self, dataset, **kwargs):  # noqa: ARG002
                captured.update(kwargs)
                return dataset

        with patch(
            "axolotl.utils.data.wrappers.load", return_value=FakeLoadedStrategy()
        ):
            get_dataset_wrapper(
                dataset_config,
                None,
                cfg,
                "dummy_type",
                dataset,
                processor=processor,
            )
        return captured

    def test_text_defaults_untouched(self):
        captured = self.capture_dataset_kwargs()
        assert "batch_size" not in captured
        assert "writer_batch_size" not in captured

    def test_multimodal_defaults_shrink(self):
        captured = self.capture_dataset_kwargs(processor=object())
        assert captured["batch_size"] == 32
        assert captured["writer_batch_size"] == 32

    def test_config_overrides_win(self):
        captured = self.capture_dataset_kwargs(
            cfg_overrides={
                "dataset_map_batch_size": 8,
                "dataset_writer_batch_size": 64,
            },
            processor=object(),
        )
        assert captured["batch_size"] == 8
        assert captured["writer_batch_size"] == 64


if __name__ == "__main__":
    pytest.main([__file__])
