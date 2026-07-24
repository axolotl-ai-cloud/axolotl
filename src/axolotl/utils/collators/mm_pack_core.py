"""Group->packed-row media-merge logic shared by the materialized collator and the streaming packer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

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
class PackedSample:
    # The three field sets record how each extra media key must be stacked when packs are batched.
    text: dict[str, np.ndarray]
    extra: dict[str, np.ndarray]
    sequence_extra_fields: set[str] = field(default_factory=set)
    batched_media_fields: set[str] = field(default_factory=set)
    ragged_media_fields: set[str] = field(default_factory=set)

    def as_row(self) -> dict[str, np.ndarray]:
        return {**self.text, **self.extra}


def normalize_groups(features) -> list[list[dict[str, Any]]]:
    if not features:
        return []
    if isinstance(features[0], list):
        return features
    return [features]


def pack_group(group: list[dict[str, Any]]) -> PackedSample:
    text_row: dict[str, np.ndarray] = {}
    extra_row: dict[str, np.ndarray] = {}
    sequence_extra_fields: set[str] = set()
    batched_media_fields: set[str] = set()
    ragged_media_fields: set[str] = set()
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
        _row_ndim(row["pixel_values"]) >= 5
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
            # Enumerate the filtered values so segment ids stay contiguous even
            # if a row in the group lacks the mask.
            text_row[key] = np.concatenate(
                [(idx + 1) * _as_1d_array(value) for idx, value in enumerate(values)]
            )
        elif key in _TEXT_FIELDS:
            text_row[key] = np.concatenate([_as_1d_array(value) for value in values])
        elif key in _GRID_FIELDS:
            extra_row[key] = np.concatenate(
                [_as_grid_array(value) for value in values], axis=0
            )
        elif key.startswith("pixel_values") or key in _IMAGE_MASK_FIELDS:
            if (
                not batched_media
                and key.startswith("pixel_values")
                and _is_ragged_pixel_values(values)
            ):
                extra_row[key] = _pack_ragged_pixel_values(values)
                ragged_media_fields.add(key)
            else:
                extra_row[key] = _pack_media_group(values, batched_media)
                if batched_media:
                    batched_media_fields.add(key)
        elif _is_sequence_extra(key, values, rows):
            sequence_extra_fields.add(key)
            extra_row[key] = np.concatenate(
                [np.asarray(value) for value in values], axis=0
            )
        else:
            # Mirror iter_tokenized_mm_rows: an unrecognized >=2D float field is
            # a media-like model input (e.g. audio input_features) that the
            # generic concat below would silently mis-pack.
            if any(
                np.asarray(value).ndim >= 2 and np.asarray(value).dtype.kind == "f"
                for value in values
            ):
                raise ValueError(
                    f"Unrecognized media-like field {key!r} in a packed multimodal "
                    "sample; this modality/model is not supported by multimodal "
                    "sample packing. Disable `sample_packing`, or add the key to "
                    "the media handling in mm_pack_core."
                )
            extra_row[key] = np.concatenate(
                [_as_media_array(key, value) for value in values], axis=0
            )

    return PackedSample(
        text=text_row,
        extra=extra_row,
        sequence_extra_fields=sequence_extra_fields,
        batched_media_fields=batched_media_fields,
        ragged_media_fields=ragged_media_fields,
    )


def batch_packed_samples(
    samples: list[PackedSample],
    text_collate: Callable[..., dict[str, torch.Tensor]],
    return_tensors=None,
) -> dict[str, torch.Tensor]:
    # text_collate is a DataCollatorForSeq2Seq-style callable for the concatenated text fields.
    sequence_extra_fields: set[str] = set()
    batched_media_fields: set[str] = set()
    ragged_media_fields: set[str] = set()
    text_features: list[dict[str, np.ndarray]] = []
    extras_by_key: dict[str, list[np.ndarray]] = {}

    for sample in samples:
        sequence_extra_fields |= sample.sequence_extra_fields
        batched_media_fields |= sample.batched_media_fields
        ragged_media_fields |= sample.ragged_media_fields
        text_features.append(sample.text)
        for key, value in sample.extra.items():
            extras_by_key.setdefault(key, []).append(value)

    batch = text_collate(text_features, return_tensors=return_tensors)
    target_len = int(batch["input_ids"].shape[1])
    for key, values in extras_by_key.items():
        if key in sequence_extra_fields:
            batch[key] = _stack_sequence_extra(values, target_len)
        elif key in batched_media_fields:
            batch[key] = _stack_batched_media(values)
        elif key in ragged_media_fields or (
            key.startswith("pixel_values") and _packs_are_ragged(values)
        ):
            batch[key] = _concat_ragged_media(values)
        else:
            batch[key] = torch.as_tensor(np.concatenate(values, axis=0))
    return batch


def _stack_batched_media(values: list[np.ndarray]) -> torch.Tensor:
    # Pad image counts with all-zero images; tiled VLMs (Idefics3) skip them
    # via pixel_attention_mask.
    max_imgs = max(int(value.shape[1]) for value in values)
    padded = []
    for value in values:
        pad = max_imgs - int(value.shape[1])
        if pad:
            pad_width = [(0, 0), (0, pad)] + [(0, 0)] * (value.ndim - 2)
            value = np.pad(value, pad_width)
        padded.append(value)
    return torch.as_tensor(np.concatenate(padded, axis=0))


def _row_ndim(value) -> int:
    # Ragged (list of differently-sized images) has no single ndim; report 0.
    try:
        return np.asarray(value).ndim
    except (ValueError, TypeError):
        return 0


def _iter_images(value) -> list[np.ndarray] | None:
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


def _is_ragged_pixel_values(values) -> bool:
    # True when packed images differ in H/W, so axis-0 concat would fail.
    shapes: list[tuple[int, ...]] = []
    for value in values:
        images = _iter_images(value)
        if images is None:
            return False
        shapes.extend(img.shape for img in images)
    return len(shapes) > 1 and any(shape != shapes[0] for shape in shapes)


def _pack_ragged_pixel_values(values) -> np.ndarray:
    # Model crops padding back via image_sizes, so pad to a common H x W and stack.
    images: list[np.ndarray] = []
    for value in values:
        value_images = _iter_images(value)
        if value_images is not None:
            images.extend(value_images)
    return _pad_and_stack_images(images)


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


def _packs_are_ragged(values) -> bool:
    if any(np.asarray(value).ndim < 3 for value in values):
        return False
    return len({np.asarray(value).shape[1:] for value in values}) > 1


def _concat_ragged_media(values: list[np.ndarray]) -> torch.Tensor:
    images = [image for value in values for image in value]
    return torch.as_tensor(_pad_and_stack_images(images))


def _pack_media_group(values, batched_media: bool) -> np.ndarray:
    arrays = [np.asarray(value) for value in values]
    if batched_media:
        arrays = [
            arr[0] if (arr.ndim >= 4 and arr.shape[0] == 1) else arr for arr in arrays
        ]
        return np.concatenate(arrays, axis=0)[None, ...]
    return np.concatenate(arrays, axis=0)


def _as_1d_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.int64).reshape(-1)


def _as_grid_array(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array.reshape(-1, array.shape[-1])


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


def _is_sequence_extra(key: str, values, rows: list[dict[str, Any]]) -> bool:
    # Heuristic: 2D+ arrays whose leading dim equals the row's token count are
    # per-token; a non-token field with a coincidentally matching leading dim
    # would be misclassified and zero-padded.
    if key.startswith("pixel_values"):
        return False
    for value, row in zip(values, rows, strict=False):
        array = np.asarray(value)
        if array.ndim < 2 or array.shape[0] != len(row["input_ids"]):
            return False
    return True


def _stack_sequence_extra(values: list[np.ndarray], target_len: int) -> torch.Tensor:
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
