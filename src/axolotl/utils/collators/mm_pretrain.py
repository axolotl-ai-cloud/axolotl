"""Collator for multimodal CPT."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

from PIL import Image
from torch import Tensor
from transformers import PreTrainedTokenizerBase, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from transformers.image_utils import load_image
from transformers.utils import PaddingStrategy

from axolotl.prompt_strategies.multimodal_pretrain import (
    ImageTokenSpec,
    check_processor_compatibility,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


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
    allow_remote_images: bool = False
    add_eos_token: bool = True

    _image_family_token_ids: set[int] = field(init=False, default_factory=set)

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

    def _resolve_image_source(self, src: Any) -> Any:
        # Only join base_dir for relative string paths; pass everything else
        # (PIL images, base64, absolute paths) through to load_image.
        if (
            self.image_base_dir
            and isinstance(src, str)
            and not os.path.isabs(src)
            and "://" not in src
        ):
            joined = os.path.join(self.image_base_dir, src)
            # Reject relative paths that escape image_base_dir (e.g. `../../etc`).
            base = os.path.realpath(self.image_base_dir)
            full = os.path.realpath(joined)
            if full != base and os.path.commonpath([base, full]) != base:
                raise ValueError(
                    f"image path {os.path.basename(src)!r} escapes image_base_dir"
                )
            return joined
        return src

    def _reject_remote(self, src: Any) -> Optional[str]:
        # Remote/URL sources are an SSRF vector when shards are untrusted; opt
        # in with allow_remote_images. Local paths never contain a `scheme://`.
        if not self.allow_remote_images and isinstance(src, str) and "://" in src:
            return src.split("://", 1)[0]
        return None

    def _load_images_for_row(self, sources: list, row_index: int) -> list[Image.Image]:
        out: list[Image.Image] = []
        for raw in sources:
            scheme = self._reject_remote(raw)
            if scheme is not None:
                msg = (
                    f"Row {row_index}: rejected remote image source "
                    f"(scheme {scheme!r}); set `allow_remote_images: true` to permit."
                )
                if self.skip_bad_images:
                    LOG.warning("%s — skipping", msg)
                    continue
                raise RuntimeError(msg)
            try:
                resolved = self._resolve_image_source(raw)
            except ValueError as exc:
                # Path traversal outside image_base_dir — surface clearly.
                msg = f"Row {row_index}: {exc}"
                if self.skip_bad_images:
                    LOG.warning("%s — skipping", msg)
                    continue
                raise RuntimeError(msg) from exc
            try:
                img = load_image(resolved)
            except Exception as exc:
                label = (
                    os.path.basename(raw)
                    if isinstance(raw, str)
                    else type(raw).__name__
                )
                msg = (
                    f"Row {row_index}: failed to load image {label!r} "
                    f"({type(exc).__name__})"
                )
                LOG.debug("failed image full source: %r; error: %s", raw, exc)
                if self.skip_bad_images:
                    LOG.warning("%s — skipping", msg)
                    continue
                raise RuntimeError(msg) from exc
            out.append(img)
        return out

    def _build_labels(self, batch: dict[str, Any]) -> Tensor:
        labels = batch["input_ids"].clone()
        # Mask padding by attention_mask, never by pad-token-id value: CPT
        # tokenizers routinely set pad_token == eos_token (and the processor may
        # pad with a different id than self.tokenizer), so value-masking would
        # delete the real trailing EOS and strip all end-of-sequence supervision.
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            labels[attention_mask == 0] = -100
        else:
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if pad_id is not None and pad_id != eos_id:
                labels[labels == pad_id] = -100
        # Without this, image-family ids dominate loss and blow it up ~10x.
        for tid in self._image_family_token_ids:
            labels[labels == tid] = -100
        return labels

    def torch_call(self, examples: list[dict]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Empty batch passed to MultiModalPretrainDataCollator.")

        # Skip our own EOS append when the tokenizer's post-processor already
        # appends one (add_eos_token=True) — otherwise the row gets a double EOS.
        tokenizer_appends_eos = bool(getattr(self.tokenizer, "add_eos_token", False))

        texts: list[str] = []
        images: list[list[Image.Image]] = []
        for i, ex in enumerate(examples):
            if "_mm_text" not in ex or "images" not in ex:
                raise KeyError(
                    f"MultiModalPretrainDataCollator: row {i} is missing "
                    f"'_mm_text' or 'images'. Did you wire the multimodal CPT "
                    f"encoder (encode_streaming_multimodal)?"
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
                raw_sources: list = []
            elif isinstance(raw, (list, tuple)):
                raw_sources = list(raw)
            else:
                raise TypeError(
                    f"Row {i}: `images` must be a list (or None), got "
                    f"{type(raw).__name__}."
                )
            # Processor re-tokenizes below, discarding the encoder's EOS — re-append.
            if (
                self.add_eos_token
                and self.tokenizer.eos_token
                and not tokenizer_appends_eos
                and not mm_text.endswith(self.tokenizer.eos_token)
            ):
                mm_text = mm_text + self.tokenizer.eos_token
            texts.append(mm_text)
            loaded = self._load_images_for_row(raw_sources, row_index=i)
            if self.skip_bad_images and len(loaded) != len(raw_sources):
                # Drop the row to avoid silent placeholder/image count mismatch.
                LOG.warning(
                    "Row %d: %d/%d images failed to load; dropping row.",
                    i,
                    len(raw_sources) - len(loaded),
                    len(raw_sources),
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
            batch["labels"] = self._build_labels(batch)
            return batch

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
                    if len(imgs) == 0:
                        # Some processors reject `images=[[]]` — would mislabel
                        # text-only rows as the offender.
                        self.tokenizer(text=[t], **retry_kwargs)
                    else:
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
                "(image placeholder expansion). Raise sequence_len if this "
                "fires repeatedly.",
                input_ids_len,
                self.max_length,
            )

        batch["labels"] = self._build_labels(batch)
        return batch
