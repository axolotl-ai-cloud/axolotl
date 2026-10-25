"""Async GRPO packing for models with recurrent layers.

TRL packs each rank's samples into a prefix trie (``TreeRow``): samples that share
a prompt forward it once, and tree attention shows every segment the K/V of its
root-to-segment path. A recurrent or linear-attention layer (GatedDeltaNet, Mamba,
short convolutions) cannot follow such a path: it scans the row in order and
carries its state from one segment into the next.

Families registered in ``monkeypatch.models.recurrent_tree`` follow the trie
themselves (their patched mixers fork state at its branches), so those models keep
prefix sharing. For other recurrent models every sample becomes its own chain
(``ChainRow``). TRL's
``TreeRow.pack`` then lays the samples end to end with ``position_ids`` restarting
at 0 for each, tree attention reduces to plain varlen attention, and the recurrent
mixers see each sample's boundaries through axolotl's sample packing patches, which
derive ``cu_seqlens`` / ``seq_idx`` from those restarts.

Everything here works around TRL refusing non-attention layers (it does so until
huggingface/trl#7686 adds tree attention for linear attention through FLA). When a
TRL release packs those models itself, ``shares_prefixes`` is the one switch the
trainer reads: drop this module and the trainer's uses of it.
"""

from __future__ import annotations

import contextlib
import importlib.util
from typing import Any

from trl.experimental.async_grpo.async_grpo_trainer import (
    FixedCountBatcher,
    TokenBudgetBatcher,
)
from trl.experimental.async_grpo.packing import TreeRow

from axolotl.monkeypatch.models.recurrent_support import (
    PACKING_PATCHED,
    RECURRENT_MODEL_TYPES,
    recurrent_layer_types,
)
from axolotl.monkeypatch.models.recurrent_tree import follows_trees, tree_family
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class ChainRow(TreeRow):
    """A ``TreeRow`` whose samples never share tokens: each one is its own chain."""

    def fits(self, sample: dict[str, Any], budget: int) -> bool:
        return (
            len(self.tokens) + len(sample["input_ids"]) <= budget
            and self.trained + int(sample["completion_mask"].sum()) <= budget
        )

    def add(self, sample: dict[str, Any]) -> None:
        input_ids = sample["input_ids"].tolist()
        start = len(self.tokens)
        nodes = list(range(start, start + len(input_ids)))
        # A key no token pair can equal, so no later sample walks into this chain.
        self.roots[("sample", len(self.samples))] = start
        for depth, token in enumerate(input_ids):
            self.tokens.append(token)
            self.children.append({})
            self.load += depth + 1
            if depth:
                self.children[-2][(token, None)] = nodes[depth]
        self.samples.append(sample)
        self.sample_nodes.append(nodes)
        self.groups.add(sample["group_id"])
        self.trained += int(sample["completion_mask"].sum())


class ChainTokenBudgetBatcher(TokenBudgetBatcher):
    """TRL's ``TokenBudgetBatcher`` placing samples in ``ChainRow`` rows."""

    def __iter__(self):
        rows = [ChainRow() for _ in range(self.num_processes)]
        for sample in self.dataset:
            n = len(sample["input_ids"])
            if n > self.token_budget:
                LOG.warning(
                    f"Dropping a rollout sample of {n} tokens that exceeds "
                    f"token_budget={self.token_budget}. Raise token_budget to avoid "
                    "dropping samples."
                )
                self.metrics["batch/dropped_oversize_total"].append(1.0)
                continue
            while True:
                placed_row = self._place(rows, sample)
                if placed_row is not None:
                    placed_row.add(sample)
                    break
                yield rows
                rows = [ChainRow() for _ in range(self.num_processes)]


class ChainFixedCountBatcher(FixedCountBatcher):
    """TRL's ``FixedCountBatcher`` placing samples in ``ChainRow`` rows."""

    def __iter__(self):
        batch = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) < self.microbatch_size:
                continue
            groups: dict[Any, list] = {}
            for queued in batch:
                groups.setdefault(queued["group_id"], []).append(queued)
            group_list = list(groups.values())
            while len(group_list) < self.num_processes:
                largest = group_list.pop(
                    max(range(len(group_list)), key=lambda i: len(group_list[i]))
                )
                half = len(largest) // 2
                group_list += [largest[:half], largest[half:]]
            priced = []
            for group in group_list:
                priced.append(ChainRow())
                for queued in group:
                    priced[-1].add(queued)
            rows = [ChainRow() for _ in range(self.num_processes)]
            for group in sorted(priced, key=lambda row: row.load, reverse=True):
                row = min(rows, key=lambda row: row.load)
                for queued in group.samples:
                    row.add(queued)
            yield rows
            batch = []


@contextlib.contextmanager
def hidden_layer_types(model):
    """Hide ``layer_types`` from TRL's ``__init__``.

    TRL rejects any non-attention layer because its trie shares prompt prefixes,
    which a recurrent layer cannot follow by itself; ``shares_prefixes`` has already
    picked a packing those layers can take. The check is the only reader of
    ``layer_types`` during ``__init__``.
    """
    text_config = model.config.get_text_config()
    layer_types = text_config.layer_types
    text_config.layer_types = None
    try:
        yield
    finally:
        text_config.layer_types = layer_types


def sample_boundaries(tree_kwargs: dict[str, Any]) -> dict[str, Any]:
    """A chain row's samples as the varlen kwargs transformers' recurrent layers read.

    Every segment of a chain row is a whole sample. Stock forwards that only take
    boundaries from ``cu_seq_lens_q`` (transformers' GatedDeltaNet) then reset their
    state per sample too; tree attention ignores these kwargs.
    """
    cu_seqlens = tree_kwargs["tree_cu_seqlens_q"]
    max_length = tree_kwargs["tree_max_seqlen_q"]
    return {
        "cu_seq_lens_q": cu_seqlens,
        "cu_seq_lens_k": cu_seqlens,
        "max_length_q": max_length,
        "max_length_k": max_length,
    }


def _gated_delta_nets(model) -> list:
    return [m for m in model.modules() if type(m).__name__.endswith("GatedDeltaNet")]


def _has_fla() -> bool:
    return importlib.util.find_spec("fla") is not None


def shares_prefixes(model) -> bool:
    """Whether a packed row may share prompt prefixes across its samples.

    Attention follows the trie, and so do recurrent layers of a registered tree
    family whose patched forward is in place. Other recurrent layers need each
    sample as its own chain, which needs axolotl's packed-sequence boundary patch
    for their architecture.
    """
    config = model.config
    types = {config.model_type, config.get_text_config().model_type}
    recurrent = recurrent_layer_types(config)
    if not recurrent and not types & RECURRENT_MODEL_TYPES:
        return True
    if not types & PACKING_PATCHED:
        layers = f" ({sorted(recurrent)} layers)" if recurrent else ""
        raise ValueError(
            f"Async GRPO packs rollouts into rows, but {config.model_type}{layers} "
            "has recurrent layers without packed-sequence boundary support, so their "
            "state would leak across the packed samples. Train it with synchronous "
            "GRPO (without `trl.async_prefetch`)."
        )
    mixers = _gated_delta_nets(model)
    forwards = [type(m).forward for m in mixers]
    if any(not getattr(f, "_axolotl_gdn_kernel_interface", False) for f in forwards):
        raise ValueError(
            "Async GRPO packs rollouts into rows, but the model's GatedDeltaNet layers "
            "run the stock forward, which mixes state across packed samples. Load the "
            "model through axolotl, which applies its packing patch."
        )
    if mixers and not _has_fla():
        # The packed GatedDeltaNet forward needs FLA's varlen kernels; without them
        # transformers would scan in pure PyTorch (~20x slower) or mix samples.
        raise ValueError(
            "Async GRPO packs GatedDeltaNet layers with flash-linear-attention's varlen "
            "kernels; install `flash-linear-attention`."
        )
    family = tree_family(types)
    recurrent_mixers = [
        m for m in model.modules() if family and type(m).__name__ in family.mixers
    ]
    return bool(recurrent_mixers) and all(follows_trees(m) for m in recurrent_mixers)
