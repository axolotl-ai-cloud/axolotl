"""ProTrain: automatic memory management for Axolotl (arXiv 2406.08334, MLSys 2026).

Exposed as an Axolotl plugin. User opt-in in YAML:

    plugins:
      - axolotl.integrations.protrain.ProTrainPlugin

See DESIGN.md for module layout and paper-section references.
"""

from axolotl.integrations.protrain.args import ProTrainArgs
from axolotl.integrations.protrain.plugin import ProTrainPlugin
from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    BlockStrategyMap,
    Bounds,
    ChunkId,
    ChunkLayout,
    CostConfig,
    HardwareProfile,
    OpId,
    OpRecord,
    ParamId,
    ProfilerConfig,
    ProfilerTrace,
    SearchResult,
    WrappedModel,
)

__all__ = [
    "BlockId",
    "BlockMode",
    "BlockStrategyMap",
    "Bounds",
    "ChunkId",
    "ChunkLayout",
    "CostConfig",
    "HardwareProfile",
    "OpId",
    "OpRecord",
    "ParamId",
    "ProTrainArgs",
    "ProTrainPlugin",
    "ProfilerConfig",
    "ProfilerTrace",
    "SearchResult",
    "WrappedModel",
]
