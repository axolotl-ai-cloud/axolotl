"""Prefix-shared packing for recurrent and linear-attention layers.

Async GRPO packs a rank's rollouts into a prefix trie (TRL's ``TreeRow``): every
segment of the trie is contiguous in the row, in depth-first order, and attention
reads each segment's root-to-segment path through ``tree_kv_index``. A recurrent
layer has to see the same path: a segment starts from the state its parent ends
in, and its short causal conv from the last inputs of its path, not from whatever
precedes it in the row.

This module holds the architecture-agnostic parts:

- ``TreePlan``: per segment its span, parent and depth level, and the conv history
  along its path, derived from TRL's tree kwargs and built once per forward.
- the driver: ``tree_conv`` runs every segment with its path history prepended
  (one call); ``tree_scan`` either forks, running the trie level by level and
  starting each segment from its parent's final state (the handoff stays in
  autograd), or runs every root-to-segment path in one call and keeps each
  segment's tail. Kernels that cannot take per-sequence initial states with
  varlen boundaries always take paths; the others fork when the tokens it saves
  outweigh one more kernel call per level.
- the registry of ``TreeFamily`` entries: which model types and mixer classes a
  component covers. The trainer reads it to choose a tree, unshared chains, or a
  refusal; a model patch registers its family next to the code it wraps.

A family component only says how to run its varlen scan with optional initial
states. TRL's own tree support for linear attention (huggingface/trl#7686) could
replace a family's entry later.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

# A varlen scan over the sequences of one call: (inputs [1, T, ...] each,
# cu_seqlens int32 [N + 1], initial_state [N, ...] or None) -> (output
# [1, T, ...], final_state [N, ...] or None).
ScanFn = Callable[
    [Sequence[torch.Tensor], torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor | None],
]
# A short causal conv: (x [1, T, D], cu_seqlens int32 [N + 1]) -> [1, T, D].
ConvFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# The kwargs a tree-following mixer reads; a layer that filters its mixer's
# kwargs passes these along.
TREE_KWARGS = ("tree_kv_index", "tree_cu_seqlens_q", "tree_cu_seqlens_k")

# Path tokens a fork must save per extra trie level. One more varlen call (fwd +
# bwd) of FLA's chunked kernels costs ~10 ms of host time per layer, rescanning
# a shared token ~0.2 us of GPU time per layer; on a launch-bound host a
# 24k-token saving still lost to rescanning.
FORK_MIN_SAVED_TOKENS = 65536


@dataclass(frozen=True)
class RecurrentComponent:
    """How one family's recurrence runs on a tree."""

    name: str
    # whether the scan takes per-sequence initial states together with cu_seqlens
    initial_state: bool = True


@dataclass(frozen=True)
class TreeFamily:
    model_types: tuple[str, ...]
    # every recurrent mixer class those model types use
    mixers: tuple[str, ...]
    component: RecurrentComponent


_FAMILIES: dict[str, TreeFamily] = {}


def register_tree_family(family: TreeFamily) -> None:
    for model_type in family.model_types:
        _FAMILIES[model_type] = family


def tree_family(model_types) -> TreeFamily | None:
    for model_type in model_types:
        if model_type in _FAMILIES:
            return _FAMILIES[model_type]
    return None


def mark_tree_forward(forward):
    """Mark a patched mixer forward that follows ``TreePlan``s."""
    forward._axolotl_recurrent_tree = True
    return forward


def follows_trees(module) -> bool:
    return getattr(type(module).forward, "_axolotl_recurrent_tree", False)


def tree_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {name: kwargs[name] for name in TREE_KWARGS if name in kwargs}


@dataclass
class _Level:
    start: int  # the level's span in the level-ordered row
    end: int
    cu_seqlens: torch.Tensor
    parents: torch.Tensor | None  # each segment's parent index in the level above


def _ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """``concat(arange(a, a + n) for a, n in zip(starts, counts))``."""
    offsets = np.cumsum(counts) - counts
    return np.repeat(starts - offsets, counts) + np.arange(counts.sum())


@dataclass
class TreePlan:
    """The segments of one packed row and the paths between them."""

    lengths: np.ndarray
    kv_index: np.ndarray  # root-to-segment paths end to end, as row positions
    cu_k: np.ndarray  # path boundaries in kv_index
    device: torch.device
    # row positions sorted by depth level (segments in row order within a level),
    # and each row position's index in that order
    order: torch.Tensor
    inverse: torch.Tensor
    levels: list[_Level] = field(default_factory=list)
    _conv: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    _paths: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def forks(self) -> bool:
        """Whether forking saves enough over running every path."""
        saved = int(self.cu_k[-1]) - int(self.lengths.sum())
        return saved >= FORK_MIN_SAVED_TOKENS * (len(self.levels) - 1)

    def conv_layout(self, width: int):
        """Gather index, cu_seqlens and kept positions for a conv of ``width``.

        Each segment is preceded by the last ``width - 1`` inputs of its path, or
        by zeros (gather index ``T``, past the row) where the path is shorter, so
        every kept output, the segments' own tokens in row order, reads what the
        full path would give it, even from a conv that ignores ``cu_seqlens``.
        """
        if width not in self._conv:
            history = width - 1
            counts = history + self.lengths
            slots = _ranges(self.cu_k[1:] - counts, counts)
            in_path = slots >= np.repeat(self.cu_k[:-1], counts)
            row_tokens = len(self.order)
            gather = np.where(in_path, self.kv_index[np.maximum(slots, 0)], row_tokens)
            cu = np.concatenate([[0], np.cumsum(counts)])
            keep = _ranges(cu[:-1] + history, self.lengths)
            self._conv[width] = self._tensors(gather, cu, keep)
        return self._conv[width]

    def path_layout(self):
        """Gather index, cu_seqlens and kept positions running every whole path."""
        if self._paths is None:
            keep = _ranges(self.cu_k[1:] - self.lengths, self.lengths)
            self._paths = self._tensors(self.kv_index, self.cu_k, keep)
        return self._paths

    def _tensors(self, gather, cu, keep):
        return (
            torch.as_tensor(gather, dtype=torch.long, device=self.device),
            torch.as_tensor(cu, dtype=torch.int32, device=self.device),
            torch.as_tensor(keep, dtype=torch.long, device=self.device),
        )


def tree_plan(kwargs: dict[str, Any]) -> TreePlan | None:
    """The plan of the packed row described by TRL's ``tree_*`` kwargs, if any.

    Built once per forward: it is kept on the ``tree_kv_index`` tensor, which
    every layer of that forward (and its checkpointed recompute) receives.
    """
    kv_index = kwargs.get("tree_kv_index")
    if kv_index is None:
        return None
    plan = getattr(kv_index, "_axolotl_tree_plan", None)
    if plan is None:
        plan = _build_plan(
            kv_index, kwargs["tree_cu_seqlens_q"], kwargs["tree_cu_seqlens_k"]
        )
        kv_index._axolotl_tree_plan = plan
    return plan


def _build_plan(kv_index, cu_q, cu_k) -> TreePlan:
    kv, q, k = (t.cpu().numpy().astype(np.int64) for t in (kv_index, cu_q, cu_k))
    lengths = np.diff(q)
    segment_of = np.repeat(np.arange(len(lengths)), lengths)
    parent_token = k[1:] - lengths - 1
    parent = np.where(
        parent_token >= k[:-1], segment_of[kv[np.maximum(parent_token, 0)]], -1
    )
    level = np.zeros(len(lengths), dtype=np.int64)
    for segment, above in enumerate(parent.tolist()):  # parents come first (DFS)
        if above >= 0:
            level[segment] = level[above] + 1
    order = np.argsort(level[segment_of], kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    device = kv_index.device
    plan = TreePlan(
        lengths,
        kv,
        k,
        device,
        torch.as_tensor(order, device=device),
        torch.as_tensor(inverse, device=device),
    )
    index_in_level = np.zeros(len(lengths), dtype=np.int64)
    start = 0
    for depth in range(int(level.max()) + 1):
        members = np.nonzero(level == depth)[0]
        index_in_level[members] = np.arange(len(members))
        sizes = lengths[members]
        cu = np.concatenate([[0], np.cumsum(sizes)])
        parents = None
        if depth:
            parents = torch.as_tensor(index_in_level[parent[members]], device=device)
        plan.levels.append(
            _Level(
                start,
                start + int(cu[-1]),
                torch.as_tensor(cu, dtype=torch.int32, device=device),
                parents,
            )
        )
        start += int(cu[-1])
    return plan


def segment_ids(cu_seqlens: torch.Tensor, total: int) -> torch.Tensor:
    """``seq_idx`` ([1, total] int32) for kernels that take boundaries that way."""
    ids = getattr(cu_seqlens, "_axolotl_segment_ids", None)
    if ids is None:
        positions = torch.arange(
            total, device=cu_seqlens.device, dtype=cu_seqlens.dtype
        )
        ids = torch.searchsorted(cu_seqlens[1:], positions, right=True)
        ids = ids.to(torch.int32)[None]
        cu_seqlens._axolotl_segment_ids = ids
    return ids


def tree_conv(
    conv: ConvFn, x: torch.Tensor, plan: TreePlan, width: int
) -> torch.Tensor:
    """A short causal conv of ``width`` over ``x`` ([1, T, D], row order) on a tree."""
    gather, cu_seqlens, keep = plan.conv_layout(width)
    padded = torch.nn.functional.pad(x, (0, 0, 0, 1))
    return conv(padded[:, gather], cu_seqlens)[:, keep]


def tree_scan(
    scan: ScanFn,
    inputs: Sequence[torch.Tensor],
    plan: TreePlan,
    component: RecurrentComponent,
) -> torch.Tensor:
    """Run a recurrence over ``inputs`` ([1, T, ...] each, row order) on a tree."""
    if len(plan.levels) == 1:  # nothing shared: the row is already in level order
        return scan(inputs, plan.levels[0].cu_seqlens, None)[0]
    if not (component.initial_state and plan.forks()):
        gather, cu_seqlens, keep = plan.path_layout()
        out, _ = scan([x[:, gather] for x in inputs], cu_seqlens, None)
        return out[:, keep]
    # One gather into level order; each level is then a contiguous slice.
    ordered = [x[:, plan.order] for x in inputs]
    outs, final = [], None
    for level in plan.levels:
        initial = None if level.parents is None else final[level.parents]
        out, final = scan(
            [x[:, level.start : level.end] for x in ordered], level.cu_seqlens, initial
        )
        outs.append(out)
    return torch.cat(outs, dim=1)[:, plan.inverse]
