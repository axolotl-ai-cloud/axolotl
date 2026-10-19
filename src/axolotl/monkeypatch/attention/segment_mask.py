"""Row-wise segment masks for packed attention (no Triton import; usable on CPU).

A :class:`SegmentAttentionMeta` describes a packed attention mask by row: query row ``i`` sees
keys in ``[a0, b0) u [a1, b1)`` that are also marked valid. The Triton kernel in
``flash_attn_d512_segments`` consumes it. The builders here derive it from the same inputs as
the FlexAttention mask_mods in ``core/trainers/diffusion_lm/attention.py`` and verify, by
counting, that the row description reproduces the mask exactly; they return ``None`` for any
layout it cannot express (the caller then keeps FlexAttention).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Attribute carrying the metadata on a FlexAttention BlockMask.
SEGMENTS_ATTR = "_axolotl_segment_meta"

_BIG = 2**30


@dataclass
class SegmentAttentionMeta:
    """Row-wise description of a packed attention mask.

    - ``q_ranges [B, M, 4]`` int32: row ``i`` may see keys in ``[a0, b0) u [a1, b1)``.
    - ``kv_valid [B, N]`` int8: key ``j`` is usable; a row sees ``ranges n kv_valid``.
    - ``kv_q_ranges [B, N, 2]`` int32: a superset ``[qa, qb)`` of the rows that see key ``j``.
    """

    q_ranges: torch.Tensor
    kv_valid: torch.Tensor
    kv_q_ranges: torch.Tensor

    @property
    def q_len(self) -> int:
        return self.q_ranges.shape[1]

    @property
    def kv_len(self) -> int:
        return self.kv_valid.shape[1]

    def to(self, *args, **kwargs) -> "SegmentAttentionMeta":
        """Move to a device; the integer dtypes are kept (a dtype argument is ignored)."""
        device = kwargs.get("device")
        if device is None:
            device = next((a for a in args if isinstance(a, (str, torch.device))), None)
        if device is None:
            return self
        return SegmentAttentionMeta(
            self.q_ranges.to(device),
            self.kv_valid.to(device),
            self.kv_q_ranges.to(device),
        )

    def dense(self) -> torch.Tensor:
        """Boolean ``[B, M, N]`` mask this metadata encodes (reference and tests only)."""
        keys = torch.arange(self.kv_len, device=self.q_ranges.device)[None, None, :]
        r = self.q_ranges.long()
        in0 = (keys >= r[..., 0:1]) & (keys < r[..., 1:2])
        in1 = (keys >= r[..., 2:3]) & (keys < r[..., 3:4])
        return (in0 | in1) & self.kv_valid.bool()[:, None, :]


def _slots(ids: torch.Tensor, members: torch.Tensor) -> torch.Tensor:
    """Index of each id in ``members`` (1-D), or ``len(members)`` when absent."""
    ordered, order = members.sort()
    where = torch.searchsorted(ordered, ids.contiguous()).clamp_max(len(members) - 1)
    found = ordered[where] == ids
    return torch.where(found, order[where], torch.full_like(where, len(members)))


def _spans(
    selected: torch.Tensor, slot: torch.Tensor, n_slots: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per (row, slot): first index, last index + 1 and count of ``selected`` positions.

    Empty slots get the span (0, 0). Slot ``n_slots`` is a discard bucket."""
    batch, length = selected.shape
    index = torch.arange(length, device=selected.device).expand(batch, length)
    target = torch.where(selected, slot, torch.full_like(slot, n_slots))
    lo = torch.full(
        (batch, n_slots + 1), _BIG, device=selected.device, dtype=torch.long
    )
    hi = torch.full((batch, n_slots + 1), -1, device=selected.device, dtype=torch.long)
    count = torch.zeros((batch, n_slots + 1), device=selected.device, dtype=torch.long)
    lo.scatter_reduce_(1, target, index, "amin")
    hi.scatter_reduce_(1, target, index, "amax")
    count.scatter_add_(1, target, torch.ones_like(index))
    empty = count == 0
    lo = torch.where(empty, 0, lo)[:, :n_slots]
    hi = torch.where(empty, 0, hi + 1)[:, :n_slots]
    return lo, hi, count[:, :n_slots]


def _count_in(prefix_sum: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
    return prefix_sum.gather(1, hi) - prefix_sum.gather(1, lo)


def _prefix_sum(mask: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.pad(mask.long().cumsum(1), (1, 0))


def decoder_segment_meta(
    encoder_document_ids: torch.Tensor,
    canvas_document_ids: torch.Tensor,
    encoder_validity: torch.Tensor,
    canvas_validity: torch.Tensor,
    decoder_prefix_lengths: torch.Tensor,
    encoder_position_ids: torch.Tensor,
    logical_ids: torch.Tensor | None = None,
) -> SegmentAttentionMeta | None:
    """Row metadata for the global (non-sliding) ``decoder_flex_block_mask``.

    Keys are ``[prompt (Lp); canvas (Lc)]``. A valid canvas query of document ``d`` sees the
    valid prompt keys of ``d`` whose position is below ``d``'s decoder prefix length, and the
    valid canvas keys of ``d``. Returns ``None`` unless each of those sets is one contiguous run
    once invalid keys are skipped."""
    if logical_ids is None:
        logical_ids = torch.arange(
            decoder_prefix_lengths.numel(), device=decoder_prefix_lengths.device
        )
    n_docs = logical_ids.numel()
    if n_docs == 0:
        return None
    prompt_length = encoder_document_ids.shape[1]
    prompt_slot = _slots(encoder_document_ids, logical_ids)
    canvas_slot = _slots(canvas_document_ids, logical_ids)
    # Prefix length per prompt token's document; 0 (nothing visible) for unknown documents.
    prefix_table = torch.cat(
        (decoder_prefix_lengths, decoder_prefix_lengths.new_zeros(1))
    ).to(encoder_position_ids.dtype)
    prefix = prefix_table[prompt_slot]
    prompt_valid = (
        encoder_validity.bool()
        & (prompt_slot < n_docs)
        & (encoder_position_ids < prefix)
    )
    canvas_valid = canvas_validity.bool() & (canvas_slot < n_docs)
    p_lo, p_hi, p_count = _spans(prompt_valid, prompt_slot, n_docs)
    c_lo, c_hi, c_count = _spans(canvas_valid, canvas_slot, n_docs)
    exact = torch.equal(
        _count_in(_prefix_sum(prompt_valid), p_lo, p_hi), p_count
    ) and torch.equal(_count_in(_prefix_sum(canvas_valid), c_lo, c_hi), c_count)
    if not exact:
        return None

    def per_token(table: torch.Tensor, slot: torch.Tensor, keep: torch.Tensor):
        padded = torch.nn.functional.pad(table, (0, 1))
        return torch.where(keep, padded.gather(1, slot), 0)

    q_ranges = torch.stack(
        (
            per_token(p_lo, canvas_slot, canvas_valid),
            per_token(p_hi, canvas_slot, canvas_valid),
            per_token(c_lo + prompt_length, canvas_slot, canvas_valid),
            per_token(c_hi + prompt_length, canvas_slot, canvas_valid),
        ),
        dim=-1,
    )
    kv_q_ranges = torch.cat(
        (
            torch.stack(
                (
                    per_token(c_lo, prompt_slot, prompt_valid),
                    per_token(c_hi, prompt_slot, prompt_valid),
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    per_token(c_lo, canvas_slot, canvas_valid),
                    per_token(c_hi, canvas_slot, canvas_valid),
                ),
                dim=-1,
            ),
        ),
        dim=1,
    )
    kv_valid = torch.cat((prompt_valid, canvas_valid), dim=1)
    return SegmentAttentionMeta(
        q_ranges.to(torch.int32).contiguous(),
        kv_valid.to(torch.int8).contiguous(),
        kv_q_ranges.to(torch.int32).contiguous(),
    )


def encoder_segment_meta(
    document_ids: torch.Tensor,
    validity: torch.Tensor,
    position_ids: torch.Tensor,
) -> SegmentAttentionMeta | None:
    """Row metadata for the global (non-sliding) ``encoder_flex_block_mask``.

    A valid query sees valid keys of its own document at positions no later than its own.
    Expressible when each document's valid tokens are one contiguous run with strictly
    increasing positions (then position order is index order)."""
    valid = validity.bool()
    members = torch.unique(document_ids[valid])
    n_docs = members.numel()
    if n_docs == 0:
        return None
    slot = _slots(document_ids, members)
    valid = valid & (slot < n_docs)
    lo, hi, count = _spans(valid, slot, n_docs)
    contiguous = torch.equal(hi - lo, count)
    same_run = valid[:, 1:] & valid[:, :-1] & (slot[:, 1:] == slot[:, :-1])
    increasing = bool((~same_run | (position_ids[:, 1:] > position_ids[:, :-1])).all())
    if not (contiguous and increasing):
        return None
    length = document_ids.shape[1]
    index = torch.arange(length, device=document_ids.device).expand_as(slot)
    padded_lo = torch.nn.functional.pad(lo, (0, 1))
    padded_hi = torch.nn.functional.pad(hi, (0, 1))
    start = torch.where(valid, padded_lo.gather(1, slot), 0)
    end = torch.where(valid, padded_hi.gather(1, slot), 0)
    zero = torch.zeros_like(start)
    q_ranges = torch.stack(
        (start, torch.where(valid, index + 1, 0), zero, zero), dim=-1
    )
    kv_q_ranges = torch.stack((torch.where(valid, index, 0), end), dim=-1)
    return SegmentAttentionMeta(
        q_ranges.to(torch.int32).contiguous(),
        valid.to(torch.int8).contiguous(),
        kv_q_ranges.to(torch.int32).contiguous(),
    )
