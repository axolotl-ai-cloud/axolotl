# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The prefix trie of a tree-packed row: its segments and their root-to-segment paths.

A tree-packed row (TRL's ``TreeRow``) lays the segments of a prefix trie out
contiguously, depth first, so a prefix shared by several samples is forwarded once.
Attention reads each segment's root-to-segment path through ``tree_kv_index``. A
recurrent layer has to see the same path: a segment starts from the state its parent
ends in, and its short causal conv reads the last inputs of its path, not whatever
precedes it in the row.

``TreePlan`` holds what the driver needs, derived on the host from the ``tree_*``
kwargs the model's forward receives. Both TRL formats are accepted:

- TRL 1.16: ``tree_kv_index``, ``tree_cu_seqlens_q`` and ``tree_cu_seqlens_k``
  (collated with ``-1`` padding);
- huggingface/trl#7686: ``tree_segment_parents`` and ``tree_segment_depths`` besides
  those (``cu_seqlens`` collated with ``0`` padding), which are used as given.

A row of whole samples laid end to end is a trie without branches
(``TreePlan.from_cu_seqlens``), so chains and trees take the same code path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

# Forward kwargs that describe a tree-packed row, in either format; a layer that
# filters the kwargs it passes on keeps these.
TREE_KWARGS = (
    "tree_kv_index",
    "tree_cu_seqlens_q",
    "tree_cu_seqlens_k",
    "tree_max_seqlen_q",
    "tree_max_seqlen_k",
    "tree_segment_parents",
    "tree_segment_depths",
    "tree_level_sizes",
    "tree_n_chunks",
)

_PLAN_ATTR = "_tree_recurrence_plan"
_SEGMENT_IDS_ATTR = "_tree_recurrence_segment_ids"

Layout = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def tree_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The tree kwargs among ``kwargs``."""
    return {name: kwargs[name] for name in TREE_KWARGS if name in kwargs}


@dataclass
class Level:
    """One depth of the trie: a contiguous span of the level-ordered row."""

    start: int
    end: int
    cu_seqlens: torch.Tensor
    # each segment's parent, as an index into the level above (None at the roots)
    parents: torch.Tensor | None


def _ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """``concat(arange(a, a + n) for a, n in zip(starts, counts))``."""
    offsets = np.cumsum(counts) - counts
    return np.repeat(starts - offsets, counts) + np.arange(counts.sum())


def _host(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.int64).reshape(-1)


def _num_segments(cu_seqlens: np.ndarray) -> int:
    """How many segments ``cu_seqlens`` holds before its collation padding."""
    if len(cu_seqlens) < 2 or cu_seqlens[0] != 0:
        return 0
    steps = np.diff(cu_seqlens)
    stop = np.flatnonzero(steps <= 0)
    return int(stop[0]) if len(stop) else len(steps)


def _depths(parents: np.ndarray) -> np.ndarray:
    depths = np.zeros(len(parents), dtype=np.int64)
    for segment, parent in enumerate(parents.tolist()):
        if parent >= segment:
            raise ValueError("a tree segment must follow its parent")
        if parent >= 0:
            depths[segment] = depths[parent] + 1
    return depths


def _parents_from_paths(
    kv_index: np.ndarray, cu_q: np.ndarray, cu_k: np.ndarray
) -> np.ndarray:
    """Each segment's parent: the segment of the path token just before its own."""
    lengths = np.diff(cu_q)
    segment_of = np.repeat(np.arange(len(lengths)), lengths)
    before = cu_k[1:] - lengths - 1
    return np.where(
        before >= cu_k[:-1], segment_of[kv_index[np.maximum(before, 0)]], -1
    )


def _paths_from_parents(
    cu_q: np.ndarray, parents: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    paths: list[np.ndarray] = []
    for segment, parent in enumerate(parents.tolist()):
        own = np.arange(cu_q[segment], cu_q[segment + 1])
        paths.append(own if parent < 0 else np.concatenate([paths[parent], own]))
    cu_k = np.concatenate([[0], np.cumsum([len(path) for path in paths])])
    return np.concatenate(paths), cu_k


@dataclass(eq=False)
class TreePlan:
    """The segments of one tree-packed row and the paths between them.

    ``cu_seqlens`` are the segments' boundaries in the row; ``parents`` their parent
    segments (``-1`` for a root), which come before them; ``depths`` their depth in
    the trie; ``kv_index`` every root-to-segment path end to end, as row positions,
    with boundaries ``cu_seqlens_k``. Index tensors are built on first use, once per
    device, and kept.
    """

    cu_seqlens: np.ndarray
    parents: np.ndarray
    depths: np.ndarray
    kv_index: np.ndarray
    cu_seqlens_k: np.ndarray
    _tensors: dict[tuple, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_kwargs(cls, kwargs: Mapping[str, Any]) -> TreePlan:
        """The plan of the row described by ``tree_*`` kwargs, in either TRL format.

        Collation padding after the last segment is ignored.
        """
        cu_q = _host(kwargs["tree_cu_seqlens_q"])
        segments = _num_segments(cu_q)
        if not segments:
            raise ValueError("tree_cu_seqlens_q describes no segments")
        cu_q = cu_q[: segments + 1]
        kv_index, cu_k = kwargs.get("tree_kv_index"), kwargs.get("tree_cu_seqlens_k")
        if kv_index is not None and cu_k is not None:
            cu_k = _host(cu_k)[: segments + 1]
            kv_index = _host(kv_index)[: cu_k[-1]]
        else:
            kv_index = cu_k = None
        if kwargs.get("tree_segment_parents") is not None:
            parents = _host(kwargs["tree_segment_parents"])[:segments]
            if kwargs.get("tree_segment_depths") is not None:
                depths = _host(kwargs["tree_segment_depths"])[:segments]
            else:
                depths = _depths(parents)
            if kv_index is None:
                kv_index, cu_k = _paths_from_parents(cu_q, parents)
        elif kv_index is not None:
            parents = _parents_from_paths(kv_index, cu_q, cu_k)
            depths = _depths(parents)
        else:
            raise ValueError(
                "a tree needs tree_kv_index and tree_cu_seqlens_k, or "
                "tree_segment_parents"
            )
        return cls(cu_q, parents, depths, kv_index, cu_k)

    @classmethod
    def from_cu_seqlens(cls, cu_seqlens: Any) -> TreePlan:
        """Whole sequences laid end to end: a trie of unbranched roots."""
        cu = _host(cu_seqlens)
        cu = cu[: _num_segments(cu) + 1]
        roots = np.full(len(cu) - 1, -1, dtype=np.int64)
        return cls(cu, roots, np.zeros_like(roots), np.arange(cu[-1]), cu)

    @property
    def num_tokens(self) -> int:
        return int(self.cu_seqlens[-1])

    @property
    def num_segments(self) -> int:
        return len(self.cu_seqlens) - 1

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.cu_seqlens)

    @property
    def num_levels(self) -> int:
        return int(self.depths.max()) + 1

    @property
    def path_tokens(self) -> int:
        """Tokens of every root-to-leaf path together: what rescanning runs."""
        leaves = self._leaves()[1]
        return int(np.diff(self.cu_seqlens_k)[leaves].sum())

    def _leaves(self) -> tuple[np.ndarray, np.ndarray]:
        """Each segment's first leaf (itself for a leaf), and the leaves."""
        key = ("leaves",)
        if key not in self._tensors:
            leaf_of = np.full(self.num_segments, -1, dtype=np.int64)
            # children come after their parents, so each is done before its parent
            for segment in range(self.num_segments - 1, -1, -1):
                if leaf_of[segment] < 0:
                    leaf_of[segment] = segment
                parent = self.parents[segment]
                if parent >= 0:
                    leaf_of[parent] = leaf_of[segment]
            leaves = np.flatnonzero(leaf_of == np.arange(self.num_segments))
            self._tensors[key] = (leaf_of, leaves)
        return self._tensors[key]

    def segment_cu_seqlens(self, device: torch.device) -> torch.Tensor:
        key = ("cu_seqlens", device)
        if key not in self._tensors:
            self._tensors[key] = torch.as_tensor(
                self.cu_seqlens, dtype=torch.int32, device=device
            )
        return self._tensors[key]

    def conv_layout(self, width: int, device: torch.device) -> Layout:
        """Gather index, ``cu_seqlens`` and kept positions for a conv of ``width``.

        Each segment is preceded by the last ``width - 1`` inputs of its path, or by
        zeros (gather index ``num_tokens``, one past the row) where the path is
        shorter, so every kept output, the segments' own tokens in row order, reads
        what the whole path would give it, even from a conv that ignores
        ``cu_seqlens``.
        """
        key = ("conv", width, device)
        if key not in self._tensors:
            history = width - 1
            lengths = self.lengths
            counts = history + lengths
            cu_k = self.cu_seqlens_k
            slots = _ranges(cu_k[1:] - counts, counts)
            in_path = slots >= np.repeat(cu_k[:-1], counts)
            gather = np.where(
                in_path, self.kv_index[np.maximum(slots, 0)], self.num_tokens
            )
            cu = np.concatenate([[0], np.cumsum(counts)])
            keep = _ranges(cu[:-1] + history, lengths)
            self._tensors[key] = self._layout(gather, cu, keep, device)
        return self._tensors[key]

    def path_layout(self, device: torch.device) -> Layout:
        """Gather index, ``cu_seqlens`` and kept positions running every root-to-leaf
        path once. A segment keeps its outputs from its first leaf's path, which runs
        through it: an inner segment's own path would only rescan a prefix."""
        key = ("paths", device)
        if key not in self._tensors:
            leaf_of, leaves = self._leaves()
            cu_k = self.cu_seqlens_k
            path_lengths = np.diff(cu_k)
            gather = self.kv_index[_ranges(cu_k[leaves], path_lengths[leaves])]
            cu = np.concatenate([[0], np.cumsum(path_lengths[leaves])])
            start = np.zeros(self.num_segments, dtype=np.int64)
            start[leaves] = cu[:-1]
            lengths = self.lengths
            keep = _ranges(start[leaf_of] + path_lengths - lengths, lengths)
            self._tensors[key] = self._layout(gather, cu, keep, device)
        return self._tensors[key]

    def level_layout(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, list[Level]]:
        """Row positions sorted by depth (row order within a depth), each row
        position's index in that order, and the levels."""
        key = ("levels", device)
        if key not in self._tensors:
            lengths = self.lengths
            segment_of = np.repeat(np.arange(self.num_segments), lengths)
            order = np.argsort(self.depths[segment_of], kind="stable")
            inverse = np.empty_like(order)
            inverse[order] = np.arange(len(order))
            index_in_level = np.zeros(self.num_segments, dtype=np.int64)
            levels, start = [], 0
            for depth in range(self.num_levels):
                members = np.flatnonzero(self.depths == depth)
                index_in_level[members] = np.arange(len(members))
                cu = np.concatenate([[0], np.cumsum(lengths[members])])
                parents = None
                if depth:
                    parents = torch.as_tensor(
                        index_in_level[self.parents[members]], device=device
                    )
                levels.append(
                    Level(
                        start,
                        start + int(cu[-1]),
                        torch.as_tensor(cu, dtype=torch.int32, device=device),
                        parents,
                    )
                )
                start += int(cu[-1])
            self._tensors[key] = (
                torch.as_tensor(order, device=device),
                torch.as_tensor(inverse, device=device),
                levels,
            )
        return self._tensors[key]

    @staticmethod
    def _layout(gather, cu, keep, device) -> Layout:
        return (
            torch.as_tensor(gather, dtype=torch.long, device=device),
            torch.as_tensor(cu, dtype=torch.int32, device=device),
            torch.as_tensor(keep, dtype=torch.long, device=device),
        )


def tree_plan(kwargs: Mapping[str, Any]) -> TreePlan | None:
    """The plan of the tree-packed row described by ``kwargs``, or None without one.

    Built once per forward: it is kept on the ``tree_cu_seqlens_q`` tensor, which
    every layer of that forward, and its checkpointed recompute, receives.
    """
    cu_seqlens = kwargs.get("tree_cu_seqlens_q")
    if cu_seqlens is None:
        return None
    plan = getattr(cu_seqlens, _PLAN_ATTR, None)
    if plan is None:
        plan = TreePlan.from_kwargs(kwargs)
        setattr(cu_seqlens, _PLAN_ATTR, plan)
    return plan


def segment_ids(cu_seqlens: torch.Tensor, total: int) -> torch.Tensor:
    """``seq_idx`` (``[1, total]`` int32) for kernels that take boundaries that way."""
    ids = getattr(cu_seqlens, _SEGMENT_IDS_ATTR, None)
    if ids is None or ids.shape[1] != total:
        positions = torch.arange(
            total, device=cu_seqlens.device, dtype=cu_seqlens.dtype
        )
        ids = torch.searchsorted(cu_seqlens[1:], positions, right=True)
        ids = ids.to(torch.int32)[None]
        setattr(cu_seqlens, _SEGMENT_IDS_ATTR, ids)
    return ids
