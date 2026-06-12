"""Data collator that splits SFT examples into an encoder prefix and a decoder canvas.

DiffusionGemma trains block-by-block: the prompt prefix feeds the autoregressive
encoder (KV cache), and a fixed-length ``canvas`` block of target tokens is denoised
by the bidirectional decoder. This collator consumes standard SFT features
(``input_ids`` + ``labels`` with ``-100`` on the prompt) and emits, per example:

    input_ids        (B, P)   right-padded prompt prefix for the encoder
    attention_mask   (B, P)   1 on real prefix tokens
    canvas_labels    (B, L)   clean target block x0 (pad where empty)
    canvas_loss_mask (B, L)   1 on real answer tokens eligible for the loss

The trainer corrupts ``canvas_labels`` into ``decoder_input_ids`` at step time
(the diffusion timestep is random per step), so corruption is intentionally *not*
done here.
"""

from __future__ import annotations

import random

import torch

IGNORE_INDEX = -100


class CanvasCollator:
    """Collate SFT features into encoder-prefix / decoder-canvas tensors."""

    def __init__(
        self,
        tokenizer,
        canvas_length: int = 256,
        block_selection: str = "random",
        pad_to_multiple_of: int | None = None,
        seed: int | None = None,
        **_ignored,
    ):
        self.tokenizer = tokenizer
        self.canvas_length = canvas_length
        self.block_selection = block_selection
        self.pad_to_multiple_of = pad_to_multiple_of
        self._rng = random.Random(seed)  # nosec B311 - block selection, not crypto
        self.pad_token_id = (
            getattr(tokenizer, "pad_token_id", None)
            if getattr(tokenizer, "pad_token_id", None) is not None
            else 0
        )
        self.bos_token_id = getattr(tokenizer, "bos_token_id", None)

    def _split_example(self, input_ids: list[int], labels: list[int]):
        """Return (prefix_ids, canvas_ids, canvas_eligible, real_prefix).

        ``real_prefix`` is True when the prefix is a true slice of ``input_ids``
        (so ``mm_token_type_ids`` can be sliced to ``len(prefix)``); it is False
        for the synthetic-BOS fallback used when there is no masked prompt.
        """
        n = len(input_ids)
        answer_positions = [i for i in range(n) if labels[i] != IGNORE_INDEX]
        real_prefix = True
        if not answer_positions:
            # No masked prompt: treat the whole sequence as the answer with a BOS prefix.
            prefix = (
                [self.bos_token_id]
                if self.bos_token_id is not None
                else [self.pad_token_id]
            )
            span_ids = list(input_ids)
            span_eligible = [True] * n
            real_prefix = False
        else:
            first = answer_positions[0]
            last = answer_positions[-1]
            prefix = list(input_ids[:first])
            span_ids = list(input_ids[first : last + 1])
            span_eligible = [labels[i] != IGNORE_INDEX for i in range(first, last + 1)]
            if not prefix:
                prefix = (
                    [self.bos_token_id]
                    if self.bos_token_id is not None
                    else [self.pad_token_id]
                )
                real_prefix = False

        # Choose which canvas-length block of the answer span to train on.
        L = self.canvas_length
        if len(span_ids) <= L:
            start = 0
        else:
            num_blocks = (len(span_ids) + L - 1) // L
            block = (
                self._rng.randrange(num_blocks)
                if self.block_selection == "random"
                else 0
            )
            start = min(block * L, len(span_ids) - 1)
            # Preceding answer tokens become part of the prefix (growing context).
            # These carry mm_token_type_ids == 0 (images only appear in the prompt),
            # so slicing mm_token_type_ids to len(prefix) stays correct.
            prefix = prefix + span_ids[:start]

        canvas_ids = span_ids[start : start + L]
        canvas_eligible = span_eligible[start : start + L]
        return prefix, canvas_ids, canvas_eligible, real_prefix

    def _max_prefix_len(self, prefixes: list[list[int]]) -> int:
        max_len = max(len(p) for p in prefixes)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        return max_len

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        prefixes, canvases, eligibles = [], [], []
        # Scan all examples: a mixed batch may carry mm ids on only some rows.
        has_mm = any("mm_token_type_ids" in f for f in features)
        prefix_mms = []
        pixel_values, image_position_ids = [], []
        L = self.canvas_length

        for f in features:
            input_ids = list(f["input_ids"])
            labels = list(f.get("labels", input_ids))
            prefix, canvas_ids, canvas_eligible, real_prefix = self._split_example(
                input_ids, labels
            )
            pad = L - len(canvas_ids)
            canvases.append(canvas_ids + [self.pad_token_id] * pad)
            eligibles.append([1 if e else 0 for e in canvas_eligible] + [0] * pad)
            prefixes.append(prefix)

            if has_mm:
                mm = list(f.get("mm_token_type_ids", [0] * len(input_ids)))
                # Image tokens only occur in the prompt, so the prefix's mm ids are a
                # straight slice; the synthetic-BOS prefix carries no image tokens.
                prefix_mms.append(
                    mm[: len(prefix)] if real_prefix else [0] * len(prefix)
                )
            if "pixel_values" in f and f["pixel_values"] is not None:
                pixel_values.append(_as_tensor(f["pixel_values"]))
            if "image_position_ids" in f and f["image_position_ids"] is not None:
                image_position_ids.append(_as_tensor(f["image_position_ids"]))

        max_len = self._max_prefix_len(prefixes)
        input_ids = torch.tensor(
            [p + [self.pad_token_id] * (max_len - len(p)) for p in prefixes],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [[1] * len(p) + [0] * (max_len - len(p)) for p in prefixes],
            dtype=torch.long,
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "canvas_labels": torch.tensor(canvases, dtype=torch.long),
            "canvas_loss_mask": torch.tensor(eligibles, dtype=torch.long),
        }
        if has_mm:
            batch["mm_token_type_ids"] = torch.tensor(
                [m + [0] * (max_len - len(m)) for m in prefix_mms], dtype=torch.long
            )
        if pixel_values:
            # Image features are gathered in row-major token order across the batch.
            batch["pixel_values"] = torch.cat(pixel_values, dim=0)
        if image_position_ids:
            batch["image_position_ids"] = torch.cat(image_position_ids, dim=0)
        return batch


def _as_tensor(x) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
