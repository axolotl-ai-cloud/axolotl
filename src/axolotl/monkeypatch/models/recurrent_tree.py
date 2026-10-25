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
  (one varlen call); ``tree_scan`` runs the trie level by level, starting each
  segment from its parent's final state, with the handoff in autograd. Kernels
  that cannot take per-sequence initial states with varlen boundaries instead run
  every root-to-segment path in one call and keep each segment's tail.
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
# [1, T, ...], final_state [N, ...]).
ScanFn = Callable[
    [Sequence[torch.Tensor], torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor | None],
]
# A varlen short causal conv: (x [1, T, D], cu_seqlens int32 [N + 1]) -> [1, T, D].
ConvFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


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

    def conv_layout(self, width: int):
        """Gather index, cu_seqlens and kept positions for a conv of ``width``.

        Each segment is preceded by the last ``width - 1`` inputs of its path, so a
        conv that starts every sequence from zeros sees what the full path would
        give it; the kept positions are the segments' own tokens, in row order.
        """
        if width not in self._conv:
            ends = self.cu_k[1:]
            history = np.minimum(width - 1, ends - self.cu_k[:-1] - self.lengths)
            counts = history + self.lengths
            gather = self.kv_index[_ranges(ends - counts, counts)]
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


def tree_conv(
    conv: ConvFn, x: torch.Tensor, plan: TreePlan, width: int
) -> torch.Tensor:
    """A short causal conv of ``width`` over ``x`` ([1, T, D], row order) on a tree."""
    gather, cu_seqlens, keep = plan.conv_layout(width)
    return conv(x[:, gather], cu_seqlens)[:, keep]


def tree_scan(
    scan: ScanFn,
    inputs: Sequence[torch.Tensor],
    plan: TreePlan,
    component: RecurrentComponent,
) -> torch.Tensor:
    """Run a recurrence over ``inputs`` ([1, T, ...] each, row order) on a tree."""
    if not component.initial_state:
        gather, cu_seqlens, keep = plan.path_layout()
        out, _ = scan([x[:, gather] for x in inputs], cu_seqlens, None)
        return out[:, keep]
    if len(plan.levels) == 1:  # nothing shared: the row is already in level order
        return scan(inputs, plan.levels[0].cu_seqlens, None)[0]
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
