"""Buffered multimodal sample packer for streaming/non-prepared datasets.

Accepted limitations: packs are non-deterministic (buffer-boundary ordering) and
single-process only (dataloader_num_workers is forced to 0; worker sharding is a
future enhancement).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator

import numpy as np
from torch.utils.data import IterableDataset

from axolotl.utils.collators.mm_pack_core import _TEXT_FIELDS, PackedSample, pack_group
from axolotl.utils.logging import get_logger
from axolotl.utils.samplers.balanced import balanced_greedy_pack_group

LOG = get_logger(__name__)

DEFAULT_MM_PACK_BUFFER_SIZE = 1000


class BufferedMultimodalPacker(IterableDataset):
    def __init__(
        self,
        source: Iterable[dict[str, Any]] | Callable[[], Iterable[dict[str, Any]]],
        batch_max_len: int,
        bin_size: int = 200,
        buffer_size: int = DEFAULT_MM_PACK_BUFFER_SIZE,
        length_key: str = "length",
        drop_attention_mask: bool = False,
    ):
        if batch_max_len <= 0:
            raise ValueError("batch_max_len must be positive")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self.source = source
        self.batch_max_len = batch_max_len
        self.bin_size = bin_size
        self.buffer_size = buffer_size
        self.length_key = length_key
        self.drop_attention_mask = drop_attention_mask
        self._overlong_warned = False

    # Trainer reads dataset.column_names; None = unknown until iterated.
    column_names = None

    def _iter_source(self) -> Iterable[dict[str, Any]]:
        # A callable source lets each epoch/worker start a fresh iterator.
        return self.source() if callable(self.source) else self.source

    def _row_length(self, row: dict[str, Any]) -> int:
        length = row.get(self.length_key)
        if length is None:
            return len(row["input_ids"])
        return int(length)

    def _pack_buffer(self, buffer: list[dict[str, Any]]) -> Iterator[PackedSample]:
        if not buffer:
            return
        lengths = np.array([self._row_length(row) for row in buffer], dtype=np.int64)
        bins = balanced_greedy_pack_group(lengths, 0, self.batch_max_len, self.bin_size)
        for bin_indices in bins:
            sample = pack_group([buffer[idx] for idx in bin_indices])
            if self.drop_attention_mask:
                # Mirror the prepared path's dataset-level mask drop (an
                # IterableDataset has no columns for the trainer to remove):
                # position_ids restarts drive packed-sequence isolation, while a
                # segment-id mask reaching an unpatched model would be misread.
                sample.text.pop("attention_mask", None)
            yield sample

    def __iter__(self) -> Iterator[PackedSample]:
        buffer: list[dict[str, Any]] = []
        for row in self._iter_source():
            # An over-long row can never fit a bin; drop it (no truncation to
            # keep image tokens aligned with pixel_values).
            length = self._row_length(row)
            if length > self.batch_max_len:
                if not self._overlong_warned:
                    self._overlong_warned = True
                    LOG.warning(
                        "Dropping multimodal row with length %d > sequence_len %d; "
                        "over-long multimodal rows are dropped (not truncated).",
                        length,
                        self.batch_max_len,
                    )
                continue
            buffer.append(row)
            if len(buffer) >= self.buffer_size:
                yield from self._pack_buffer(buffer)
                buffer = []
        yield from self._pack_buffer(buffer)


_MEDIA_PASSTHROUGH_PREFIX = "pixel_values"
_MEDIA_EXTRA_KEYS = {
    "image_grid_thw",
    "video_grid_thw",
    "image_sizes",
    "pixel_attention_mask",
    "image_attention_mask",
    "pixel_values_videos",
    "cross_attention_mask",
}


def _is_tensor_like(value: Any) -> bool:
    # A real (batched) model input has an array shape; scalar metadata does not.
    return getattr(value, "ndim", 0) >= 1


def iter_tokenized_mm_rows(
    examples: Iterable[dict[str, Any]],
    processing_strategy,
) -> Iterator[dict[str, Any]]:
    """Tokenize raw MM chat rows one at a time; mirrors MultiModalChatDataCollator per row plus `length`/`position_ids`."""
    import torch

    for example in examples:
        processed = processing_strategy([example])
        batch = processing_strategy.processor.apply_chat_template(
            [processed[0]["messages"]],
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            chat_template=processing_strategy.chat_template,
            processor_kwargs={"padding": True},
        )
        input_ids = batch["input_ids"][0]
        length = int(input_ids.shape[0])
        row: dict[str, Any] = {
            "input_ids": input_ids,
            "labels": processing_strategy.process_labels(batch["input_ids"])[0],
            "length": length,
            "position_ids": torch.arange(length),
        }
        if "attention_mask" in batch:
            row["attention_mask"] = batch["attention_mask"][0]
        else:
            row["attention_mask"] = torch.ones(length, dtype=torch.long)
        for key, value in batch.items():
            if key in {"input_ids", "labels", "attention_mask", "position_ids"}:
                continue
            if key in _TEXT_FIELDS:
                # Per-token fields (e.g. Gemma3 token_type_ids) are batched [1, L].
                row[key] = value[0]
            elif key.startswith(_MEDIA_PASSTHROUGH_PREFIX) or key in _MEDIA_EXTRA_KEYS:
                row[key] = value
            elif _is_tensor_like(value):
                # A tensor/array-valued processor output we don't know how to pack
                # (e.g. audio `input_features`) would be silently dropped otherwise.
                raise ValueError(
                    f"Unrecognized processor output {key!r} with a tensor/array value; "
                    "this modality/model is not supported by multimodal sample packing. "
                    "Disable `sample_packing`, or add the key to the media handling in "
                    "iter_tokenized_mm_rows."
                )
        yield row


def build_buffered_mm_packer(source, cfg) -> BufferedMultimodalPacker:
    # batch_max_len is one packed sequence (sequence_len); the collator stacks packs into a batch.
    bin_size = cfg.sample_packing_bin_size
    buffer_size = cfg.mm_pack_buffer_size
    return BufferedMultimodalPacker(
        source,
        batch_max_len=cfg.sequence_len,
        bin_size=bin_size if bin_size is not None else 200,
        buffer_size=(
            buffer_size if buffer_size is not None else DEFAULT_MM_PACK_BUFFER_SIZE
        ),
        drop_attention_mask=bool(cfg.attn_decontaminates_packing),
    )


def build_streaming_mm_dataset(
    raw_dataset, cfg, tokenizer, processor
) -> BufferedMultimodalPacker:
    from axolotl.processing_strategies import get_processing_strategy
    from axolotl.utils.chat_templates import get_chat_template_from_config

    chat_template = (
        get_chat_template_from_config(cfg=cfg, tokenizer=tokenizer)
        if cfg.chat_template
        else None
    )
    datasets_cfg = cfg.datasets or []
    # roles_to_train / train_on_eos are read from the first dataset only (mirrors
    # the single-dataset assumption in sft.py).
    ds_cfg = datasets_cfg[0] if datasets_cfg else None

    def _ds_get(key):
        if ds_cfg is None:
            return None
        if hasattr(ds_cfg, "get"):
            try:
                return ds_cfg.get(key)
            except (AttributeError, KeyError, TypeError):
                pass
        return getattr(ds_cfg, key, None)

    field_messages: list[str] = []
    for dataset_cfg in datasets_cfg:
        value = (
            dataset_cfg.get("field_messages")
            if hasattr(dataset_cfg, "get")
            else getattr(dataset_cfg, "field_messages", None)
        )
        if value and value not in field_messages:
            field_messages.append(value)

    strategy = get_processing_strategy(
        processor,
        chat_template,
        cfg.chat_template,
        image_size=cfg.image_size,
        image_resize_algorithm=cfg.image_resize_algorithm,
        train_on_inputs=bool(cfg.train_on_inputs),
        roles_to_train=_ds_get("roles_to_train"),
        train_on_eos=_ds_get("train_on_eos"),
        role_boundaries_override=(
            list(cfg.role_boundaries) if cfg.role_boundaries else None
        ),
        field_messages=field_messages or None,
    )

    def _source():
        return iter_tokenized_mm_rows(raw_dataset, strategy)

    return build_buffered_mm_packer(_source, cfg)
