# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Tree packing with TRL's experimental ``AsyncGRPOTrainer``.

TRL packs each rank's samples into a prefix trie (``TreeRow``) and passes the model
the ``tree_*`` kwargs the hooks read. Besides ``install``, a trainer needs:

- ``tree_kwargs_from_batch``: a collated batch's tree as forward kwargs, from either
  TRL format;
- ``hidden_layer_types``: TRL 1.16 refuses a model with any non-attention
  ``layer_types``; once ``install`` reports ``tree`` or ``chains``, hide them while
  TRL builds its trainer;
- for ``chains``: ``ChainRow`` and the chain batchers, which make every sample its
  own chain (they import TRL), and ``sample_boundaries``, a chain row's samples as
  the varlen kwargs stock layers read.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from typing import Any

import torch

_CHAINS = ("ChainRow", "ChainTokenBudgetBatcher", "ChainFixedCountBatcher")


def __getattr__(name: str) -> Any:
    if name in _CHAINS:
        from tree_recurrence.integrations import _trl_chains

        return getattr(_trl_chains, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _row(value: torch.Tensor) -> torch.Tensor:
    return value[0] if value.dim() > 1 else value


def tree_kwargs_from_batch(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """This rank's tree, from a batch collated by TRL, as the model's forward kwargs.

    TRL 1.16 pads ``tree_kv_index`` and the ``cu_seqlens`` with ``-1`` and gives the
    longest segment and path in ``tree_max_seqlens``; huggingface/trl#7686 pads with
    ``0`` and gives every length in ``tree_sizes`` (as its ``unpad_tree`` reads it).
    """
    row = {key: _row(value) for key, value in inputs.items() if key.startswith("tree_")}
    if "tree_sizes" in row:
        sizes = [n for n in row["tree_sizes"].tolist() if n > 0]
        max_seqlen_q, max_seqlen_k, n_kv, n_chunks, *level_sizes = sizes
        segments = sum(level_sizes)
        return {
            "tree_kv_index": row["tree_kv_index"][:n_kv],
            "tree_cu_seqlens_q": row["tree_cu_seqlens_q"][: segments + 1],
            "tree_cu_seqlens_k": row["tree_cu_seqlens_k"][: segments + 1],
            "tree_max_seqlen_q": max_seqlen_q,
            "tree_max_seqlen_k": max_seqlen_k,
            "tree_segment_parents": row["tree_segment_parents"][:segments],
            "tree_segment_depths": row["tree_segment_depths"][:segments],
            "tree_level_sizes": level_sizes,
            "tree_n_chunks": n_chunks,
        }
    kwargs: dict[str, Any] = {
        key: row[key][row[key] >= 0]
        for key in ("tree_kv_index", "tree_cu_seqlens_q", "tree_cu_seqlens_k")
    }
    if "tree_max_seqlens" in row:
        max_seqlen_q, max_seqlen_k = row["tree_max_seqlens"].tolist()[:2]
    else:
        max_seqlen_q = int(kwargs["tree_cu_seqlens_q"].diff().max())
        max_seqlen_k = int(kwargs["tree_cu_seqlens_k"].diff().max())
    kwargs["tree_max_seqlen_q"] = max_seqlen_q
    kwargs["tree_max_seqlen_k"] = max_seqlen_k
    return kwargs


def sample_boundaries(tree: Mapping[str, Any]) -> dict[str, Any]:
    """A chain row's samples as the varlen kwargs transformers' layers read.

    Every segment of a chain row is a whole sample, so forwards that only take
    boundaries from ``cu_seq_lens_q`` (transformers' Gated DeltaNet without these
    hooks, say) reset their state per sample too. Tree attention ignores them.
    """
    cu_seqlens = tree["tree_cu_seqlens_q"]
    max_length = tree.get("tree_max_seqlen_q")
    if max_length is None:
        max_length = int(cu_seqlens.diff().max())
    return {
        "cu_seq_lens_q": cu_seqlens,
        "cu_seq_lens_k": cu_seqlens,
        "max_length_q": max_length,
        "max_length_k": max_length,
    }


@contextlib.contextmanager
def hidden_layer_types(model: torch.nn.Module) -> Iterator[None]:
    """Hide ``layer_types`` from the trainer's ``__init__``.

    TRL 1.16 rejects any non-attention layer, since a recurrent layer cannot follow
    its trie by itself; ``install`` has made them follow it, or chose chains. The
    check is the only reader of ``layer_types`` while TRL builds the trainer.
    """
    text_config = model.config.get_text_config()
    layer_types = getattr(text_config, "layer_types", None)
    text_config.layer_types = None
    try:
        yield
    finally:
        text_config.layer_types = layer_types
