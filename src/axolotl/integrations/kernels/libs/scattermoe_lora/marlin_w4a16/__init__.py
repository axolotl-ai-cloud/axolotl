"""sm120 Marlin NVFP4 **W4A16** forward backend for the grouped DeepSeek-V4 MoE.

Replaces the CUTLASS FP4xFP4 (lossy fp4 *activations*) sm120 base FORWARD with a Marlin weight-only
NVFP4 GEMM (bf16 activations, bit-correct). At the thin-M LoRA-FT shape (E=256, ~48 tok/expert) this
is ~1.79x faster AND removes the ~9.3% activation-quant error of the W4A4 path. The Marlin MoE GEMM
+ ``gptq_marlin_repack`` are vendored under ``_csrc/`` (extracted from vLLM, ported to a regular
``torch::Tensor`` ABI, bit-exact to vLLM) so there is NO vLLM runtime dependency. JIT-built once via
``torch.utils.cpp_extension.load`` and cached.

Forward-only: the backward stays the existing gradient-consistent ``_base_dx`` (fp8-read on sm120,
now enabled for the pad-64 layout via the BM=tile dX kernel; bf16-dequant elsewhere). A Marlin NVFP4
transpose-repack backward would inject ~18% gradient noise — re-quantizing W^T is an independent
4-bit quantization, which the maintainer's "bf16 grad required" bar rejects.

Lazy build/import keeps non-sm120 / no-nvcc environments (incl. CI) clean; ``marlin_w4a16_available``
returns False there and the dsv4 MoE forward falls back to CUTLASS (sm120) / DeepGEMM (sm90/100).
"""

from __future__ import annotations

import functools
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_HERE, "_csrc")
_MOE = os.path.join(_CSRC, "libtorch_stable", "moe", "marlin_moe_wna16")
_MARLIN_INC = os.path.join(_CSRC, "libtorch_stable", "quantization", "marlin")
_EXT = None


@functools.lru_cache(maxsize=1)
def marlin_w4a16_available() -> bool:
    """True iff the standalone Marlin W4A16 ext can build + run here: sm120 (consumer Blackwell),
    CUDA available, and a CUDA toolkit (nvcc) present to JIT-build the extension."""
    try:
        if not torch.cuda.is_available():
            return False
        if torch.cuda.get_device_capability()[0] != 12:  # sm120
            return False
        from torch.utils.cpp_extension import CUDA_HOME

        return CUDA_HOME is not None
    except Exception:
        return False


def load_ext():
    """JIT-build (cached) and return the vendored standalone Marlin ext module (sm120 only).

    Exposes ``moe_wna16_marlin_gemm`` (NVFP4 W4A16 MoE GEMM, bf16 act/out) and ``gptq_marlin_repack``
    (NVFP4 weight -> Marlin tile layout). Compiled for ``sm_120a``."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        _EXT = load(
            name="axolotl_marlin_w4a16",
            sources=[
                os.path.join(_MOE, "ops_standalone.cu"),
                os.path.join(_MOE, "sm80_kernel_bfloat16_fe2m1f_bfloat16.cu"),
                os.path.join(_MOE, "repack_standalone.cu"),
            ],
            extra_cuda_cflags=[
                "-O3",
                "-gencode=arch=compute_120a,code=sm_120a",
                "-DMARLIN_NAMESPACE_NAME=marlin_moe_wna16",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                # The __global__ template instantiations live in sm80_kernel_*.cu and are referenced
                # (as undefined externs) from ops_standalone.cu's kernel selector. Whole-program
                # compilation would emit them as static stubs; keep external linkage so the link
                # resolves. (Requires a CUDA >= 12.4 nvcc; matches the sm120 / CUDA-13 target.)
                "-static-global-template-stub=false",
                "-Xcompiler",
                "-fPIC",
            ],
            extra_cflags=["-O3", "-DMARLIN_NAMESPACE_NAME=marlin_moe_wna16"],
            extra_include_paths=[_CSRC, _MARLIN_INC, _MOE],
            verbose=False,
        )
    return _EXT
