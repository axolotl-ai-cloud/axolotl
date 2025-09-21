"""Vendored Triton contiguous grouped GEMM kernels from TorchTitan."""

from .cg_backward import ContiguousGroupedGEMM
from .cg_forward import (
    ContiguousGroupedGEMM as ContiguousGroupedGEMMForwardOnly,
    cg_grouped_gemm,
    cg_grouped_gemm_forward,
    cg_grouped_gemm_forward_dynamic,
)

__all__ = [
    "cg_grouped_gemm",
    "cg_grouped_gemm_forward",
    "cg_grouped_gemm_forward_dynamic",
    "ContiguousGroupedGEMM",
    "ContiguousGroupedGEMMForwardOnly",
]
