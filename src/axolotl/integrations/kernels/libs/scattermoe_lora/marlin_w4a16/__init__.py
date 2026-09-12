"""Marlin NVFP4 **W4A16** forward backend for the grouped DeepSeek-V4 MoE (Ampere and newer).

Replaces the CUTLASS FP4xFP4 (lossy fp4 *activations*) base FORWARD with a Marlin weight-only NVFP4
GEMM (bf16 activations, bit-correct). At the thin-M LoRA-FT shape (E=256, ~48 tok/expert) this is
~1.79x faster than CUTLASS on sm120 AND removes the ~9.3% activation-quant error of the W4A4 path.
The Marlin MoE GEMM + ``gptq_marlin_repack`` are vendored under ``_csrc/`` (extracted from vLLM,
ported to a regular ``torch::Tensor`` ABI, bit-exact to vLLM) so there is NO vLLM runtime dependency.
JIT-built once (for the current device's arch) via ``torch.utils.cpp_extension.load`` and cached.

Portability: the kernel is the standard Ampere-class Marlin (``mma.m16n8k16`` bf16 + bit-twiddle FP4
decode) — NO Hopper/Blackwell-only intrinsics. The W4A16 (bf16-act x NVFP4-weight) config runs on
any sm80+ GPU; the ``__CUDA_ARCH__ < 890`` guard in the template only disables *fp8 activations*,
which this path never uses. This makes Marlin the fused base GEMM that also covers Ampere (A100,
sm80) and Ada (RTX 4090 / L40S, sm89), where DeepGEMM (sm90+) and CUTLASS-fp4 (sm120) don't run.

This backend provides the FORWARD only; the backward keeps the existing gradient-consistent
``_base_dx`` (fp8-read on sm120, now enabled for the pad-64 layout via the BM=tile dX kernel;
bf16-dequant elsewhere). A Marlin NVFP4 transpose-repack backward was evaluated and rejected: the
backward contracts the other axis, so reusing the weight needs an independent 4-bit re-quantization
of W^T that disagrees with the forward weight by ~18% — measured to land ~13% in the LoRA gradients
(it is a structured weight mismatch, not noise the low-rank projection averages out). fp8-read keeps
the backward gradient-consistent at 1 byte/wt, which is the floor for block-scaled 4-bit.

Lazy build/import keeps no-nvcc / pre-Ampere environments (incl. CI) clean; ``marlin_w4a16_available``
returns False there and the dsv4 MoE forward falls back to CUTLASS (sm120) / DeepGEMM (sm90/100) /
chunked dequant.
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
    """True iff the standalone Marlin W4A16 ext can build + run here: Ampere or newer (sm80+, the
    bf16 ``mma`` the W4A16 path needs), CUDA available, and an nvcc present to JIT-build it."""
    try:
        if not torch.cuda.is_available():
            return False
        if torch.cuda.get_device_capability()[0] < 8:  # bf16 mma needs sm80+
            return False
        from torch.utils.cpp_extension import CUDA_HOME

        return CUDA_HOME is not None
    except Exception:
        return False


def _gencode_for_current_device() -> str:
    """``-gencode`` for the running GPU's arch (build only what we run). Hopper+ uses the arch-
    specific ('a') target to match the vendored vLLM build; Ampere/Ada use the plain target."""
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    suffix = "a" if major >= 9 else ""
    return f"-gencode=arch=compute_{arch}{suffix},code=sm_{arch}{suffix}"


def load_ext():
    """JIT-build (cached) and return the vendored standalone Marlin ext module for this GPU's arch.

    Exposes ``moe_wna16_marlin_gemm`` (NVFP4 W4A16 MoE GEMM, bf16 act/out) and ``gptq_marlin_repack``
    (NVFP4 weight -> Marlin tile layout). Compiled for the current device's compute capability."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        major, minor = torch.cuda.get_device_capability()
        # Arch in the ext name so two arches in one torch-extensions cache don't collide.
        _EXT = load(
            name=f"axolotl_marlin_w4a16_sm{major}{minor}",
            sources=[
                os.path.join(_MOE, "ops_standalone.cu"),
                os.path.join(_MOE, "sm80_kernel_bfloat16_fe2m1f_bfloat16.cu"),
                os.path.join(_MOE, "repack_standalone.cu"),
            ],
            extra_cuda_cflags=[
                "-O3",
                _gencode_for_current_device(),
                "-DMARLIN_NAMESPACE_NAME=marlin_moe_wna16",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                # Keep external linkage on the __global__ template instantiations so
                # ops_standalone.cu's selector resolves them at link, not static stubs.
                # Requires CUDA >= 12.4 nvcc.
                "-static-global-template-stub=false",
                "-Xcompiler",
                "-fPIC",
            ],
            extra_cflags=["-O3", "-DMARLIN_NAMESPACE_NAME=marlin_moe_wna16"],
            extra_include_paths=[_CSRC, _MARLIN_INC, _MOE],
            verbose=False,
        )
    return _EXT
