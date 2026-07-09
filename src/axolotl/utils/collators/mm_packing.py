"""Packed multimodal SFT collator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from axolotl.utils.collators.batching import DataCollatorForSeq2Seq

_TEXT_FIELDS = {
    "input_ids",
    "labels",
    "attention_mask",
    "position_ids",
    "token_type_ids",
    "mm_token_type_ids",
}
_GRID_FIELDS = {
    "image_grid_thw",
    "video_grid_thw",
    # Pixtral/LLaVA-NeXT per-image (H, W); concat to (num_images, 2), no batch dim.
    "image_sizes",
}
# Per-image masks that must track pixel_values' batch layout.
_IMAGE_MASK_FIELDS = {
    "pixel_attention_mask",
    "image_attention_mask",
}


@dataclass
class MultiModalBatchSamplerDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    sequence_extra_fields: set[str] = field(default_factory=set, init=False)
    batched_media_fields: set[str] = field(default_factory=set, init=False)
    ragged_media_fields: set[str] = field(default_factory=set, init=False)

    def __call__(self, features, return_tensors=None):
        self.sequence_extra_fields = set()
        self.batched_media_fields = set()
        self.ragged_media_fields = set()
        groups = self._normalize_groups(features)
        text_features = []
        extras_by_key: dict[str, list[np.ndarray]] = {}

        for group in groups:
            text_row, extra_row = self._pack_group(group)
            text_features.append(text_row)
            for key, value in extra_row.items():
                extras_by_key.setdefault(key, []).append(value)

        batch = super().__call__(text_features, return_tensors=return_tensors)
        target_len = int(batch["input_ids"].shape[1])
        for key, values in extras_by_key.items():
            if key in self.sequence_extra_fields:
                batch[key] = self._stack_sequence_extra(values, target_len)
            elif key in self.batched_media_fields:
                batch[key] = self._stack_batched_media(values)
            elif key in self.ragged_media_fields or (
                key.startswith("pixel_values") and self._packs_are_ragged(values)
            ):
                batch[key] = self._concat_ragged_media(values)
            else:
                batch[key] = torch.as_tensor(np.concatenate(values, axis=0))
        return batch

    @staticmethod
    def _stack_batched_media(values: list[np.ndarray]) -> torch.Tensor:
        # Tiled VLMs (Idefics3) drop the all-zero padding images.
        max_imgs = max(int(value.shape[1]) for value in values)
        padded = []
        for value in values:
            pad = max_imgs - int(value.shape[1])
            if pad:
                pad_width = [(0, 0), (0, pad)] + [(0, 0)] * (value.ndim - 2)
                value = np.pad(value, pad_width)
            padded.append(value)
        return torch.as_tensor(np.concatenate(padded, axis=0))

    @staticmethod
    def _normalize_groups(features) -> list[list[dict[str, Any]]]:
        if not features:
            return []
        if isinstance(features[0], list):
            return features
        return [features]

    def _pack_group(
        self, group: list[dict[str, Any]]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        text_row: dict[str, np.ndarray] = {}
        extra_row: dict[str, np.ndarray] = {}
        keys = set().union(*(row.keys() for row in group))

        # Concatenating per-sample cross_attention_masks desyncs image count vs
        # pixel_values; block-diagonal layout not built yet, so fail loudly.
        if len(group) > 1 and any("cross_attention_mask" in row for row in group):
            raise ValueError(
                "Sample packing is not supported for cross-attention vision models "
                "(e.g. mllama / Llama-3.2-Vision) whose inputs carry "
                "`cross_attention_mask`. Set `sample_packing: false` for this model."
            )

        # Tiled VLMs (Idefics3/SmolVLM) ship 5D pixel_values with a leading batch dim
        # the model needs back; decide once so pixel_values and its masks stay consistent.
        batched_media = any(
            self._row_ndim(row["pixel_values"]) >= 5
            for row in group
            if row.get("pixel_values") is not None
        )

        for key in keys:
            if key == "length":
                continue
            rows = [row for row in group if key in row and row[key] is not None]
            values = [row[key] for row in rows]
            if not values:
                continue
            if key == "attention_mask":
                text_row[key] = np.concatenate(
                    [
                        (idx + 1) * self._as_1d_array(row[key])
                        for idx, row in enumerate(group)
                        if key in row and row[key] is not None
                    ]
                )
            elif key in _TEXT_FIELDS:
                text_row[key] = np.concatenate(
                    [self._as_1d_array(value) for value in values]
                )
            elif key in _GRID_FIELDS:
                extra_row[key] = np.concatenate(
                    [self._as_grid_array(value) for value in values], axis=0
                )
            elif key.startswith("pixel_values") or key in _IMAGE_MASK_FIELDS:
                if (
                    not batched_media
                    and key.startswith("pixel_values")
                    and self._is_ragged_pixel_values(values)
                ):
                    extra_row[key] = self._pack_ragged_pixel_values(values)
                    self.ragged_media_fields.add(key)
                else:
                    extra_row[key] = self._pack_media_group(values, batched_media)
                    if batched_media:
                        self.batched_media_fields.add(key)
            elif self._is_sequence_extra(key, values, rows):
                self.sequence_extra_fields.add(key)
                extra_row[key] = np.concatenate(
                    [np.asarray(value) for value in values], axis=0
                )
            else:
                extra_row[key] = np.concatenate(
                    [self._as_media_array(key, value) for value in values], axis=0
                )

        return text_row, extra_row

    @staticmethod
    def _row_ndim(value) -> int:
        # Ragged (list of differently-sized images) has no single ndim; report 0.
        try:
            return np.asarray(value).ndim
        except (ValueError, TypeError):
            return 0

    @classmethod
    def _iter_images(cls, value) -> list[np.ndarray] | None:
        # None for non-image layouts (e.g. Qwen2-VL 2D flat patches).
        if isinstance(value, (list, tuple)) or (
            isinstance(value, np.ndarray) and value.dtype == object
        ):
            images = []
            for item in value:
                arr = np.asarray(item)
                if arr.ndim != 3:
                    return None
                images.append(arr)
            return images or None
        arr = np.asarray(value)
        if arr.ndim == 4:
            return [arr[i] for i in range(arr.shape[0])]
        if arr.ndim == 3:
            return [arr]
        return None

    @classmethod
    def _is_ragged_pixel_values(cls, values) -> bool:
        # True when packed images differ in H/W, so axis-0 concat would fail.
        shapes: list[tuple[int, ...]] = []
        for value in values:
            images = cls._iter_images(value)
            if images is None:
                return False
            shapes.extend(img.shape for img in images)
        return len(shapes) > 1 and any(shape != shapes[0] for shape in shapes)

    @classmethod
    def _pack_ragged_pixel_values(cls, values) -> np.ndarray:
        # Model crops padding back via image_sizes, so pad to a common H×W and stack.
        images: list[np.ndarray] = []
        for value in values:
            images.extend(cls._iter_images(value))
        return cls._pad_and_stack_images(images)

    @staticmethod
    def _pad_and_stack_images(images: list[np.ndarray]) -> np.ndarray:
        h_max = max(img.shape[-2] for img in images)
        w_max = max(img.shape[-1] for img in images)
        padded = []
        for img in images:
            pad = [(0, 0)] * (img.ndim - 2) + [
                (0, h_max - img.shape[-2]),
                (0, w_max - img.shape[-1]),
            ]
            padded.append(np.pad(img, pad) if any(p != (0, 0) for p in pad) else img)
        return np.stack(padded, axis=0)

    @staticmethod
    def _packs_are_ragged(values) -> bool:
        if any(np.asarray(value).ndim < 3 for value in values):
            return False
        return len({np.asarray(value).shape[1:] for value in values}) > 1

    @classmethod
    def _concat_ragged_media(cls, values: list[np.ndarray]) -> torch.Tensor:
        images = [image for value in values for image in value]
        return torch.as_tensor(cls._pad_and_stack_images(images))

    @staticmethod
    def _pack_media_group(values, batched_media: bool) -> np.ndarray:
        arrays = [np.asarray(value) for value in values]
        if batched_media:
            arrays = [
                arr[0] if (arr.ndim >= 4 and arr.shape[0] == 1) else arr
                for arr in arrays
            ]
            return np.concatenate(arrays, axis=0)[None, ...]
        return np.concatenate(arrays, axis=0)

    @staticmethod
    def _as_1d_array(value) -> np.ndarray:
        return np.asarray(value, dtype=np.int64).reshape(-1)

    @staticmethod
    def _as_grid_array(value) -> np.ndarray:
        array = np.asarray(value, dtype=np.int64)
        if array.ndim == 1:
            return array.reshape(1, -1)
        return array.reshape(-1, array.shape[-1])

    @staticmethod
    def _as_media_array(key: str, value) -> np.ndarray:
        array = np.asarray(value)
        if key.startswith("pixel_values") and array.ndim > 3 and array.shape[0] == 1:
            array = array.reshape(array.shape[1:])
        if key.startswith("pixel_values") and array.ndim == 3:
            return array.reshape(1, *array.shape)
        if key.startswith("pixel_values"):
            return array
        if array.ndim == 0:
            return array.reshape(1)
        if array.ndim in (1, 2) and array.dtype.kind in {"i", "u", "f", "b"}:
            return array.reshape(1, *array.shape)
        return array

    @staticmethod
    def _is_sequence_extra(key: str, values, rows: list[dict[str, Any]]) -> bool:
        if key.startswith("pixel_values"):
            return False
        for value, row in zip(values, rows, strict=False):
            array = np.asarray(value)
            if array.ndim < 2 or array.shape[0] != len(row["input_ids"]):
                return False
        return True

    @staticmethod
    def _stack_sequence_extra(
        values: list[np.ndarray], target_len: int
    ) -> torch.Tensor:
        padded = []
        for value in values:
            pad_len = target_len - int(value.shape[0])
            if pad_len > 0:
                pad_shape = (pad_len, *value.shape[1:])
                value = np.concatenate(
                    [value, np.zeros(pad_shape, dtype=value.dtype)], axis=0
                )
            padded.append(value)
        return torch.as_tensor(np.stack(padded, axis=0))
