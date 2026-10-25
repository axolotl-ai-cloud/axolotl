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
    tokens: torch.Tensor  # row positions of the level's segments, end to end
    cu_seqlens: torch.Tensor
    parents: torch.Tensor | None  # each segment's parent index in the level above


@dataclass
class TreePlan:
    """The segments of one packed row and the paths between them."""

    starts: list[int]
    lengths: list[int]
    parent: list[int]
    paths: list[list[int]]  # root-to-segment row positions, ending with its own
    num_tokens: int
    device: torch.device
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
            gather, cu, keep = [], [0], []
            for path, length in zip(self.paths, self.lengths, strict=True):
                history = path[max(0, len(path) - length - (width - 1)) : -length]
                keep.extend(
                    range(
                        len(gather) + len(history), len(gather) + len(history) + length
                    )
                )
                gather.extend(history + path[-length:])
                cu.append(len(gather))
            self._conv[width] = self._tensors(gather, cu, keep)
        return self._conv[width]

    def path_layout(self):
        """Gather index, cu_seqlens and kept positions running every whole path."""
        if self._paths is None:
            gather, cu, keep = [], [0], []
            for path, length in zip(self.paths, self.lengths, strict=True):
                gather.extend(path)
                cu.append(len(gather))
                keep.extend(range(len(gather) - length, len(gather)))
            self._paths = self._tensors(gather, cu, keep)
        return self._paths

    def _tensors(self, gather, cu, keep):
        return (
            torch.tensor(gather, dtype=torch.long, device=self.device),
            torch.tensor(cu, dtype=torch.int32, device=self.device),
            torch.tensor(keep, dtype=torch.long, device=self.device),
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
    kv, q, k = (t.tolist() for t in (kv_index, cu_q, cu_k))
    starts = q[:-1]
    lengths = [b - a for a, b in zip(q[:-1], q[1:], strict=True)]
    paths = [kv[a:b] for a, b in zip(k[:-1], k[1:], strict=True)]
    owner = {}
    for segment, (start, length) in enumerate(zip(starts, lengths, strict=True)):
        owner.update(dict.fromkeys(range(start, start + length), segment))
    parent, level = [], []
    for path, length in zip(paths, lengths, strict=True):
        above = owner[path[-length - 1]] if len(path) > length else -1
        parent.append(above)
        level.append(0 if above < 0 else level[above] + 1)
    plan = TreePlan(starts, lengths, parent, paths, q[-1], kv_index.device)
    index_in_level: dict[int, int] = {}
    for depth in range(max(level) + 1):
        members = [s for s in range(len(starts)) if level[s] == depth]
        for i, segment in enumerate(members):
            index_in_level[segment] = i
        tokens = [p for s in members for p in range(starts[s], starts[s] + lengths[s])]
        cu = [0]
        for segment in members:
            cu.append(cu[-1] + lengths[segment])
        parents = (
            torch.tensor(
                [index_in_level[parent[s]] for s in members], device=plan.device
            )
            if depth
            else None
        )
        plan.levels.append(
            _Level(
                torch.tensor(tokens, dtype=torch.long, device=plan.device),
                torch.tensor(cu, dtype=torch.int32, device=plan.device),
                parents,
            )
        )
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
    result: torch.Tensor | None = None
    final = None
    for level in plan.levels:
        initial = None if level.parents is None else final[level.parents]
        out, final = scan(
            [x[:, level.tokens] for x in inputs], level.cu_seqlens, initial
        )
        if result is None:
            result = out.new_zeros((out.shape[0], plan.num_tokens, *out.shape[2:]))
        result = result.index_copy(1, level.tokens, out)
    assert result is not None
    return result
