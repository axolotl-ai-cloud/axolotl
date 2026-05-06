"""ProTrain block-manager subpackage (§3.1.2).

Public surface:

- ``BlockMode`` — activation strategy enum (re-exported from ``strategy.py``).
- ``wrap_block`` / ``unwrap_block`` — per-block mode dispatcher.
- ``assign_modes`` — layout rules (swap-early, unopt-late, interleave).
- ``discover_blocks`` — find the transformer-block trees on a model.
- ``BlockTree`` — one tree (encoder, decoder, or single causal-LM tree).
- ``flatten_block_trees`` — concat trees into a forward-ordered block list.
- ``block_id_path_map`` — dotted-path -> global BlockId, for the trace.
"""

from __future__ import annotations

from axolotl.integrations.protrain.block.dispatcher import unwrap_block, wrap_block
from axolotl.integrations.protrain.block.layout_rules import (
    BlockTree,
    assign_modes,
    block_id_path_map,
    discover_blocks,
    flatten_block_trees,
)
from axolotl.integrations.protrain.block.strategy import (
    BlockMode,
    BlockStrategyMap,
    StrategyError,
)

__all__ = [
    "BlockMode",
    "BlockStrategyMap",
    "BlockTree",
    "StrategyError",
    "assign_modes",
    "block_id_path_map",
    "discover_blocks",
    "flatten_block_trees",
    "unwrap_block",
    "wrap_block",
]
