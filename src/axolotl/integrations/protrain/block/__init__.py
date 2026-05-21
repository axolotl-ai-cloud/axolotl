"""ProTrain block-manager subpackage."""

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
