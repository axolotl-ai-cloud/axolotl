"""Tests for the multimodal pre-tokenization cache path."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from datasets import Dataset, IterableDataset

from axolotl.datasets import TokenizedPromptDataset, wrap_dataset_for_tokenized_prompt
from axolotl.prompt_strategies.chat_template import (
    ChatTemplatePrompter,
    ChatTemplateStrategy,
)
from axolotl.prompt_tokenizers import DatasetWrappingStrategy, PromptTokenizingStrategy
from axolotl.utils.data.wrappers import get_dataset_wrapper
from axolotl.utils.datasets import dataset_map_buffer_kwargs
from axolotl.utils.dict import DictDefault
from axolotl.utils.trainer import process_datasets_for_packing

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


class TestCrossBatchMixedModality:
    """Mixed media/text rows must tokenize even when a whole map batch is
    single-modality (the Arrow writer locks its schema on the first batch)."""

    def make_strategy(self):
        strategy = make_strategy(processor=FakeProcessor())

        def fake_single(prompt):
            row = {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]}
            if prompt.get("images"):
                row["pixel_values"] = [[0.5, 0.5]]
            return row

        strategy._tokenize_single_prompt = fake_single  # pylint: disable=protected-access
        return strategy

    def test_single_modality_batches_tokenize(self):
        # batch_size=2 -> batches [text, text] and [img, img]: schema drift repro
        ds = Dataset.from_dict(
            {
                "messages": [["a"], ["b"], ["c"], ["d"]],
                "images": [[], None, ["img"], ["img"]],
            }
        )
        tds = TokenizedPromptDataset(self.make_strategy(), ds, batch_size=2)
        assert len(tds) == 4
        assert tds[0]["pixel_values"] is None
        assert tds[2]["pixel_values"] == [[0.5, 0.5]]

    def test_row_order_preserved(self):
        ds = Dataset.from_dict(
            {
                "messages": [["a"], ["b"], ["c"], ["d"]],
                "images": [["img"], [], ["img"], []],
            }
        )
        tds = TokenizedPromptDataset(self.make_strategy(), ds, batch_size=1)
        has_media = [tds[i]["pixel_values"] is not None for i in range(4)]
        assert has_media == [True, False, True, False]

    def test_homogeneous_dataset_skips_split(self):
        ds = Dataset.from_dict(
            {"messages": [["a"], ["b"]], "images": [["img"], ["img"]]}
        )
        strategy = self.make_strategy()
        tds = TokenizedPromptDataset(strategy, ds, batch_size=2)
        assert tds._split_media_indices(ds) is None  # pylint: disable=protected-access
        assert all(tds[i]["pixel_values"] is not None for i in range(2))

    def test_text_only_strategy_skips_split(self):
        ds = Dataset.from_dict({"messages": [["a"]], "images": [["img"]]})
        strategy = make_strategy()  # no processor
        tds = TokenizedPromptDataset.__new__(TokenizedPromptDataset)
        tds.prompt_tokenizer = strategy
        assert tds._split_media_indices(ds) is None  # pylint: disable=protected-access

    def test_dropped_rows_dont_crash_order_restore(self):
        strategy = self.make_strategy()
        original = strategy._tokenize_single_prompt  # pylint: disable=protected-access

        def drop_page_two(prompt):
            if prompt.get("messages") == ["c"]:
                return {}
            return original(prompt)

        strategy._tokenize_single_prompt = drop_page_two  # pylint: disable=protected-access
        ds = Dataset.from_dict(
            {
                "messages": [["a"], ["b"], ["c"], ["d"]],
                "images": [["img"], [], ["img"], []],
            }
        )
        tds = TokenizedPromptDataset(strategy, ds, batch_size=2)
        assert len(tds) == 3
        media = sorted(tds[i]["pixel_values"] is not None for i in range(3))
        assert media == [False, False, True]

    def test_keep_in_memory_forwarded_to_flatten(self):
        ds = Dataset.from_dict(
            {
                "messages": [["a"], ["b"]],
                "images": [["img"], []],
            }
        )
        captured = {}
        original_flatten = Dataset.flatten_indices

        def spy(self, **kwargs):
            captured.update(kwargs)
            return original_flatten(self, **kwargs)

        with patch.object(Dataset, "flatten_indices", spy):
            TokenizedPromptDataset(
                self.make_strategy(), ds, keep_in_memory=True, writer_batch_size=2
            )
        assert captured["keep_in_memory"] is True
        assert captured["writer_batch_size"] == 2

    def test_iterable_batch_size_forwarded(self):
        strategy = self.make_mixed_iterable_strategy()
        ds = Dataset.from_dict({"messages": [["a"]], "images": [["img"]]})
        iterable = ds.to_iterable_dataset()
        captured = {}
        original_map = IterableDataset.map

        def spy(self, *args, **kwargs):
            captured.update(kwargs)
            return original_map(self, *args, **kwargs)

        with patch.object(IterableDataset, "map", spy):
            wrap_dataset_for_tokenized_prompt(strategy, iterable, batch_size=7)
        assert captured["batch_size"] == 7

    def make_mixed_iterable_strategy(self):
        return self.make_strategy()


class TestBufferKwargsHelper:
    """dataset_map_buffer_kwargs drives every map in the prepared pipeline."""

    def test_text_run_returns_empty(self):
        assert dataset_map_buffer_kwargs(DictDefault({}), batched=True) == {}

    def test_multimodal_cfg_defaults(self):
        cfg = DictDefault({"processor_type": "AutoProcessor"})
        assert dataset_map_buffer_kwargs(cfg, batched=True) == {
            "writer_batch_size": 32,
            "batch_size": 32,
        }

    def test_unbatched_omits_batch_size(self):
        cfg = DictDefault({"is_multimodal": True})
        assert dataset_map_buffer_kwargs(cfg) == {"writer_batch_size": 32}

    def test_overrides_apply_without_processor(self):
        cfg = DictDefault(
            {"dataset_map_batch_size": 8, "dataset_writer_batch_size": 64}
        )
        assert dataset_map_buffer_kwargs(cfg, batched=True) == {
            "writer_batch_size": 64,
            "batch_size": 8,
        }

    def test_packing_maps_receive_buffers(self):
        cfg = DictDefault(
            {
                "processor_type": "AutoProcessor",
                "dataset_num_proc": 1,
                "sample_packing": True,
                "sequence_len": 64,
            }
        )
        ds = Dataset.from_dict(
            {
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]],
                "labels": [[1, 2, 3]],
            }
        )
        captured = []
        filter_captured = []
        original_map = Dataset.map
        original_filter = Dataset.filter

        def spy(self, *args, **kwargs):
            captured.append(kwargs)
            return original_map(self, *args, **kwargs)

        def filter_spy(self, *args, **kwargs):
            filter_captured.append(kwargs)
            return original_filter(self, *args, **kwargs)

        with (
            patch.object(Dataset, "map", spy),
            patch.object(Dataset, "filter", filter_spy),
        ):
            process_datasets_for_packing(cfg, ds, None)
        position_ids_maps = [
            k for k in captured if k.get("desc", "").startswith("Add position_id")
        ]
        assert position_ids_maps
        assert all(k["writer_batch_size"] == 32 for k in position_ids_maps)
        trainable_filters = [
            k for k in filter_captured if "Trainable Tokens" in k.get("desc", "")
        ]
        assert trainable_filters
        assert all(k["batch_size"] == 32 for k in trainable_filters)


if __name__ == "__main__":
    pytest.main([__file__])
