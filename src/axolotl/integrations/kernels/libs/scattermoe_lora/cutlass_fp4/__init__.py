"""sm120 CUTLASS NVFP4 grouped-GEMM MoE path (CuTe DSL).

Optional: requires an sm120 (consumer Blackwell) GPU and nvidia-cutlass-dsl>=4.6 (the sm120
block-scaled SF helpers, partition_fragment_SFA etc., landed in the 4.6 dev line). On any other
environment `cutlass_fp4_available()` returns False and the dsv4 MoE forward falls back to the
DeepGEMM (sm90/sm100) or chunked-dequant path. Lazy imports keep non-Blackwell / no-cutlass-dsl
environments (incl. CI) clean.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def cutlass_fp4_available() -> bool:
    """True iff the sm120 CUTLASS NVFP4 grouped path can run here."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        if torch.cuda.get_device_capability()[0] != 12:  # sm120 consumer Blackwell
            return False
        import cutlass.cute.nvgpu.warp.mma as _wm  # noqa: F401
        import cutlass.utils.blackwell_helpers as _bh

        # sm120 block-scaled SF helpers landed in 4.6; absent in 4.5.x
        return hasattr(_bh, "partition_fragment_SFA") and hasattr(_wm, "MmaMXF4NVF4Op")
    except Exception:
        return False
