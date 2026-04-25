"""Collator for multimodal CPT."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

from PIL import Image
from torch import Tensor
from transformers import PreTrainedTokenizerBase, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from transformers.utils import PaddingStrategy

from axolotl.prompt_strategies.multimodal_pretrain import (
    ImageTokenSpec,
    check_processor_compatibility,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Decompression-bomb cap (~7070x7070).
_DEFAULT_MAX_IMAGE_PIXELS = 50_000_000
_DEFAULT_MAX_IMAGES_PER_ROW = 32


@dataclass
class MultiModalPretrainDataCollator(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    processor: ProcessorMixin
    image_token_spec: ImageTokenSpec
    image_base_dir: Optional[str] = None
    return_tensors: Literal["pt"] = "pt"
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None
    max_length: Optional[int] = None
    skip_bad_images: bool = False
    max_image_pixels: int = _DEFAULT_MAX_IMAGE_PIXELS
    max_images_per_row: int = _DEFAULT_MAX_IMAGES_PER_ROW

    _image_family_token_ids: set[int] = field(init=False, default_factory=set)
    _base_dir_real: Optional[str] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.return_tensors != "pt":
            raise ValueError(
                "MultiModalPretrainDataCollator only supports "
                "return_tensors='pt' (in-place torch ops are used downstream)."
            )
        check_processor_compatibility(self.processor)
        # All-text batches use self.tokenizer; image batches use self.processor.
        # If they don't share the same tokenizer instance, the two paths can
        # tokenize the same text differently.
        proc_tokenizer = getattr(self.processor, "tokenizer", None)
        if proc_tokenizer is not None and proc_tokenizer is not self.tokenizer:
            LOG.warning(
                "MultiModalPretrainDataCollator.tokenizer is not "
                "processor.tokenizer; all-text and image batches may "
                "tokenize inconsistently."
            )
        self._image_family_token_ids = set(self.image_token_spec.image_family_token_ids)
        if self.image_base_dir is not None:
            self._base_dir_real = os.path.realpath(self.image_base_dir)

    def _resolve_image_path(self, p: str) -> str:
        if not isinstance(p, str):
            raise ValueError(f"Image path must be str, got {type(p).__name__}.")
        if "\x00" in p:
            raise ValueError("Image path contains embedded NUL byte.")
        p_lower = p.lower()
        if p_lower.startswith(
            (
                "http://",
                "https://",
                "ftp://",
                "ftps://",
                "file://",
                "data:",
                "s3://",
                "gs://",
                "gcs://",
                "az://",
                "azure://",
                "hf://",
            )
        ) or p.startswith(("\\\\", "//")):
            raise ValueError(
                f"Non-local image path scheme is not supported in v1 "
                f"multimodal CPT (got {p!r})."
            )
        if self._base_dir_real is not None:
            if os.path.isabs(p):
                raise ValueError(
                    f"Absolute image path {p!r} is rejected when "
                    f"`image_base_dir` is configured. All image paths must be "
                    f"relative to the configured base directory."
                )
            resolved = os.path.realpath(os.path.join(self._base_dir_real, p))
            # commonpath (not startswith) so root-dir bases like "/" work.
            try:
                within_base = (
                    os.path.commonpath([self._base_dir_real, resolved])
                    == self._base_dir_real
                )
            except ValueError:
                within_base = False
            if not within_base:
                raise ValueError(
                    f"Image path {p!r} resolves outside `image_base_dir` "
                    f"after symlink resolution. Refusing to load."
                )
            return resolved
        return os.path.realpath(p) if os.path.isabs(p) else p

    def _open_image_hardened(self, resolved: str) -> Image.Image:
        # O_NOFOLLOW closes the realpath→open TOCTOU window for the final component.
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(resolved, os.O_RDONLY | nofollow)
        except OSError as exc:
            raise ValueError(
                f"Cannot open image (os.open failed: {type(exc).__name__})."
            ) from exc
        file_obj = os.fdopen(fd, "rb")
        try:
            with Image.open(file_obj) as src:
                w, h = src.size
                if w * h > self.max_image_pixels:
                    raise ValueError(
                        f"Image pixels ({w}x{h}) exceed "
                        f"max_image_pixels ({self.max_image_pixels})."
                    )
                # Multi-frame bomb guard (GIF/TIFF/WebP).
                n_frames = getattr(src, "n_frames", 1)
                if n_frames > 1:
                    raise ValueError(
                        f"Multi-frame images are not supported (got {n_frames} frames)."
                    )
                img = src.convert("RGB")
                img.load()
                return img
        finally:
            if not file_obj.closed:
                file_obj.close()

    def _load_images_for_row(
        self, paths: list[str], row_index: int
    ) -> list[Image.Image]:
        if len(paths) > self.max_images_per_row:
            raise ValueError(
                f"Row {row_index}: {len(paths)} images exceeds "
                f"`max_images_per_row={self.max_images_per_row}`. Split the "
                f"row or raise the cap if this is expected."
            )
        out: list[Image.Image] = []
        for raw in paths:
            try:
                resolved = self._resolve_image_path(raw)
                img = self._open_image_hardened(resolved)
            except Exception as exc:
                # Top-level log gets basename only; full path stays on DEBUG.
                basename = os.path.basename(str(raw))
                msg = (
                    f"Row {row_index}: failed to load image {basename!r} "
                    f"({type(exc).__name__})"
                )
                LOG.debug("failed image full path: %r; error: %s", raw, exc)
                if self.skip_bad_images:
                    LOG.warning("%s — skipping", msg)
                    continue
                raise RuntimeError(msg) from exc
            out.append(img)
        return out

    def torch_call(self, examples: list[dict]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Empty batch passed to MultiModalPretrainDataCollator.")

        texts: list[str] = []
        images: list[list[Image.Image]] = []
        for i, ex in enumerate(examples):
            if "_mm_text" not in ex or "images" not in ex:
                raise KeyError(
                    f"MultiModalPretrainDataCollator: row {i} is missing "
                    f"'_mm_text' or 'images'. Did you wire the multimodal CPT "
                    f"encoder (encode_streaming_multimodal or "
                    f"MultimodalPretrainTokenizationStrategy)?"
                )
            mm_text = ex["_mm_text"]
            if not isinstance(mm_text, str):
                raise TypeError(
                    f"Row {i}: `_mm_text` must be str, got "
                    f"{type(mm_text).__name__}. Check dataset encoding "
                    f"(Parquet BINARY columns may surface as bytes)."
                )
            raw = ex["images"]
            if raw is None:
                raw_paths: list[str] = []
            elif isinstance(raw, (list, tuple)):
                raw_paths = list(raw)
            else:
                raise TypeError(
                    f"Row {i}: `images` must be a list (or None), got "
                    f"{type(raw).__name__}."
                )
            for j, rp in enumerate(raw_paths):
                if not isinstance(rp, str):
                    raise TypeError(
                        f"Row {i}, image {j}: path must be str, got "
                        f"{type(rp).__name__}."
                    )
            texts.append(mm_text)
            loaded = self._load_images_for_row(raw_paths, row_index=i)
            if self.skip_bad_images and len(loaded) != len(raw_paths):
                # Drop the row to avoid silent placeholder/image count mismatch.
                LOG.warning(
                    "Row %d: %d/%d images failed to load; dropping row.",
                    i,
                    len(raw_paths) - len(loaded),
                    len(raw_paths),
                )
                texts.pop()
                continue
            images.append(loaded)

        if not texts:
            raise RuntimeError(
                "All rows in the batch were dropped due to image load "
                "failures. Check dataset integrity."
            )

        # All-text batch: bypass the processor and tokenize directly.
        if all(len(im) == 0 for im in images):
            LOG.debug(
                "MultiModalPretrainDataCollator: all-text batch (%d rows); "
                "using tokenizer-only fallback (no pixel_values).",
                len(texts),
            )
            tok_kwargs: dict[str, Any] = {
                "text": texts,
                "return_tensors": self.return_tensors,
                "padding": self.padding,
            }
            if self.pad_to_multiple_of is not None:
                tok_kwargs["pad_to_multiple_of"] = self.pad_to_multiple_of
            batch = self.tokenizer(**tok_kwargs)
            tok_input_ids: Tensor = batch["input_ids"]
            tok_labels = tok_input_ids.clone()
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is not None:
                tok_labels[tok_labels == pad_id] = -100
            for tid in self._image_family_token_ids:
                tok_labels[tok_labels == tid] = -100
            batch["labels"] = tok_labels
            return dict(batch)

        # No truncation: it chops input_ids mid-placeholder while pixel_values
        # keep every image — silent text/pixel mismatch. We warn post-hoc instead.
        proc_kwargs: dict[str, Any] = {
            "text": texts,
            "images": images,
            "return_tensors": self.return_tensors,
            "padding": self.padding,
        }
        if self.pad_to_multiple_of is not None:
            proc_kwargs["pad_to_multiple_of"] = self.pad_to_multiple_of
        try:
            batch = self.processor(**proc_kwargs)
        except Exception as exc:
            # Pinpoint the bad row; bail to "inconclusive" if retry raises a different class.
            LOG.warning(
                "MultiModalPretrainDataCollator: processor failed on a batch "
                "of %d rows (%s); retrying each row individually to locate "
                "the offender. This adds up to %d extra processor calls.",
                len(texts),
                type(exc).__name__,
                len(texts),
            )
            offender_idx: Optional[int] = None
            retry_ok = True
            retry_kwargs: dict[str, Any] = {
                "return_tensors": self.return_tensors,
                "padding": self.padding,
            }
            if self.pad_to_multiple_of is not None:
                retry_kwargs["pad_to_multiple_of"] = self.pad_to_multiple_of
            for i, (t, imgs) in enumerate(zip(texts, images, strict=True)):
                try:
                    self.processor(text=[t], images=[imgs], **retry_kwargs)
                except Exception as retry_exc:
                    if isinstance(retry_exc, type(exc)) or isinstance(
                        exc, type(retry_exc)
                    ):
                        offender_idx = i
                    else:
                        retry_ok = False
                    break
            if offender_idx is not None:
                location = f"row {offender_idx}"
            elif retry_ok:
                location = (
                    f"batch of {len(texts)} rows "
                    f"(individual rows all succeed; see __cause__ for details)"
                )
            else:
                location = f"batch of {len(texts)} rows (retry inconclusive)"
            raise RuntimeError(
                f"MultiModalPretrainDataCollator: processor call failed on "
                f"{location} ({type(exc).__name__}: {exc}). Common causes: "
                f"placeholder token absent from the row's text, image count "
                f"mismatch, or an unsupported processor class."
            ) from exc

        input_ids_len = batch["input_ids"].shape[-1]
        if self.max_length is not None and input_ids_len > self.max_length:
            LOG.warning(
                "Batch input_ids length %d exceeds configured sequence_len %d "
                "(image placeholder expansion). Reduce max_images_per_row or "
                "raise sequence_len if this fires repeatedly.",
                input_ids_len,
                self.max_length,
            )

        input_ids: Tensor = batch["input_ids"]
        labels = input_ids.clone()

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # Without this, image-family ids dominate loss and blow it up ~10x.
        for tid in self._image_family_token_ids:
            labels[labels == tid] = -100

        batch["labels"] = labels
        return batch
