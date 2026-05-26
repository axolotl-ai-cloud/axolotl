"""ProTrain runtime/version sentinels.

Single source of truth for module-level sentinel constants. These are
runtime detection markers (e.g. checked by benchmark scripts via ``grep``
or imported by test fixtures) and version stamps. Adding a new sentinel?
Add it here, not in the module where it's read.

The original modules re-export each sentinel for backward compatibility
so external ``grep`` against the original file still finds a match and
import sites that reference the historical location continue to work.
"""

from __future__ import annotations

# ---- Architecture / runtime version sentinels ----
# Each one stamps a specific in-place subsystem version so external scripts
# can detect which fix is in the worktree without running the code.

# ARCH #8: round-robin partition for Mode C optim-state sharding.
# Originally lived in api/optim_wrapper.py. Bump when the partition scheme
# changes shape.
_PROTRAIN_PERSISTENT_ROUND_ROBIN_PARTITION_VERSION = 1

# ARCH #8 huge-tensor edge: within-shard fallback for huge persistent params.
# Originally lived in api/optim_wrapper.py. Bump when within-param shard
# scheme changes shape.
_PROTRAIN_PERSISTENT_HUGE_PARAM_WITHIN_SHARD_VERSION = 1

# ARCH #9: backend-aware status tensor for cross-world resume.
# Originally lived in api/checkpoint.py. See CHECKPOINT_DESIGN_PHASE2.md §13.
_CROSS_WORLD_NCCL_CPU_BRIDGE = "v1"

# ARCH #10: torch.compile compat decoration on hook factories.
# Originally lived in runtime/hooks.py.
_PROTRAIN_TORCH_COMPILE_COMPAT = 1

# ---- Cache / trace version ----

# Profiler trace schema version. Bumped on any schema change to
# ProfilerTrace; old payloads are then ignored. Originally lived in
# profiler/cache.py.
TRACE_VERSION = 24
