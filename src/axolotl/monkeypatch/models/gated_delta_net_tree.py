"""Run GatedDeltaNet over the prefix trie of an async GRPO packed row.

TRL packs the rollouts of a rank into a trie: a token shared by several samples
is forwarded once, and tree attention lets each segment attend to the K/V of its
root-to-segment path. A GatedDeltaNet layer can follow the same paths: its in-
and out-projections are per token, so they run once per unique token, while the
causal conv and the gated delta rule run over every path gathered end to end
(FLA's varlen kernels, one sequence per path). Each segment keeps the outputs of
its own tokens, which are the tail of its path, so the result equals packing
every sample on its own; only the recurrence over shared prefixes is repeated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TreePaths:
    # packed position of every token on every root-to-segment path, end to end
    kv_index: torch.Tensor
    # path boundaries in ``kv_index``, for FLA's varlen kernels
    cu_seqlens: torch.Tensor
    # for each packed token, where in the gathered paths its own output is
    select: torch.Tensor


def tree_paths(kwargs: dict[str, Any]) -> TreePaths | None:
    """The trie layout TRL passes the model as ``tree_*`` kwargs, if any."""
    kv_index = kwargs.get("tree_kv_index")
    if kv_index is None:
        return None
    cu_q = kwargs["tree_cu_seqlens_q"].long()
    cu_k = kwargs["tree_cu_seqlens_k"]
    lengths = cu_q[1:] - cu_q[:-1]
    segment = torch.repeat_interleave(
        torch.arange(len(lengths), device=cu_q.device), lengths
    )
    offset = torch.arange(int(cu_q[-1]), device=cu_q.device) - cu_q[:-1][segment]
    select = cu_k[1:].long()[segment] - lengths[segment] + offset
    return TreePaths(kv_index, cu_k.to(torch.int32), select)
