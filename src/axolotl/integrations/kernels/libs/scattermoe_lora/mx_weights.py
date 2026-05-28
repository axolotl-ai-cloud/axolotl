# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
MXFP4 expert weight container + helpers for the fused-dequant Triton kernels.

The container carries the packed uint8 ``[E_active, N, K/2]`` data and the
E8M0 ``[E_active, N, K/32]`` scales for the *active* experts of one MoE
step. ``parallel_linear_lora`` checks for this container instance and
routes to the MX-aware Triton kernels.

Layout: OCP block axis is the contraction axis ``K`` — the last storage
dim. The same buffer is consumed by both the forward kernel (K is the
matmul reduction axis) and the dX kernel (K is the output axis, with
scales broadcast within ``MX_BLOCK_SIZE``-element K blocks). No
pre-transpose / re-quantize is needed for the backward path.

The FP4 E2M1 codebook is the standard OCP-MX one (16 values:
``±{0, 0.5, 1, 1.5, 2, 3, 4, 6}``); we cache one fp32 copy per CUDA device.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

import torch

MX_BLOCK_SIZE = 32

# Standard OCP-MX fp4 e2m1 codebook (sign bit | 2-bit exp | 1-bit mantissa).
# Index by the raw 4-bit nibble. Cached fp32 tensor for kernel lookups.
_FP4_E2M1_LUT = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

_LUT_CACHE: dict[torch.device, torch.Tensor] = {}


def fp4_codebook(device: torch.device) -> torch.Tensor:
    """Return the cached 16-entry FP4 E2M1 → fp32 lookup on ``device``."""
    key = device
    lut = _LUT_CACHE.get(key)
    if lut is None or lut.device != device:
        lut = torch.tensor(_FP4_E2M1_LUT, dtype=torch.float32, device=device)
        _LUT_CACHE[key] = lut
    return lut


class MXLayout(enum.IntEnum):
    """Which axis the OCP-MX block scaling runs along, in *kernel* coords.

    Currently only ``FWD`` (block axis = K) is supported and used by both
    the forward and dX kernels. The enum is kept as a future extension
    point for swizzled or N-axis-blocked variants.
    """

    FWD = 0


@dataclass
class MXWeights:
    """Packed + scale tensors for one MoE projection's active experts.

    Attributes
    ----------
    packed:
        ``uint8`` tensor, shape ``[E_active, N, K/2]``.
    scales:
        ``uint8`` (E8M0) tensor, shape ``[E_active, N, K/32]``.
    K, N:
        Logical contraction/output dimensions of the dequantized W.
    layout:
        Which axis the block scaling runs along (see ``MXLayout``).
    block_size:
        OCP MX block size; only ``32`` is supported by the kernels.
    """

    packed: torch.Tensor
    scales: torch.Tensor
    K: int
    N: int
    layout: MXLayout = MXLayout.FWD
    block_size: int = MX_BLOCK_SIZE
    num_experts: Optional[int] = None  # E_active; convenience field
    orig_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        assert self.block_size == MX_BLOCK_SIZE, (
            f"only block_size={MX_BLOCK_SIZE} is supported, got {self.block_size}"
        )
        # scales are E8M0 (float8_e8m0fnu) in torchao; viewed as uint8 here so
        # the Triton kernel can load them with simple integer arithmetic.
        if self.scales.dtype != torch.uint8:
            self.scales = self.scales.view(torch.uint8)
        assert self.packed.dtype == torch.uint8, (
            f"packed must be uint8, got {self.packed.dtype}"
        )
        if self.num_experts is None:
            self.num_experts = self.packed.size(0)

    @property
    def device(self) -> torch.device:
        return self.packed.device


def _torchao_mxtensor_cls():
    """Return the torchao MXTensor class, or ``None`` if torchao is missing."""
    try:
        from torchao.prototype.mx_formats.mx_tensor import MXTensor
    except ImportError:
        return None
    return MXTensor


def _mx_qdata(mx) -> torch.Tensor:
    """Read the packed-nibble buffer off an MXTensor, tolerating torchao
    renaming the attribute between versions."""
    qdata = getattr(mx, "qdata", None)
    if qdata is None:
        qdata = getattr(mx, "_data", None)
    if qdata is None:
        raise AttributeError(
            "torchao MXTensor exposes neither .qdata nor ._data; "
            "this torchao version is unsupported."
        )
    return qdata


def _mx_scale(mx) -> torch.Tensor:
    """Read the E8M0 scale buffer off an MXTensor, tolerating torchao
    renaming the attribute between versions."""
    scale = getattr(mx, "scale", None)
    if scale is None:
        scale = getattr(mx, "_scale_e8m0", None)
    if scale is None:
        raise AttributeError(
            "torchao MXTensor exposes neither .scale nor ._scale_e8m0; "
            "this torchao version is unsupported."
        )
    return scale


def _construct_mxtensor_subset(
    parent, qdata_slice: torch.Tensor, scale_slice: torch.Tensor
):
    """Construct a new MXTensor that shares ``parent``'s metadata but uses
    the provided ``qdata_slice`` / ``scale_slice`` buffers.

    Pinned to torchao 0.17.0's positional constructor (qdata, scale,
    elem_dtype, block_size, orig_dtype, kernel_preference,
    act_quant_kwargs, is_swizzled_scales). Optional attributes are read via
    ``getattr`` so we degrade gracefully if a future torchao version drops
    or renames one — the single point of pain for torchao internals access
    across this codebase.
    """
    MXTensor = _torchao_mxtensor_cls()
    if MXTensor is None:
        raise ImportError("MXFP4 path requires torchao (install `torchao>=0.7`).")
    kernel_preference = getattr(parent, "kernel_preference", None)
    act_quant_kwargs = getattr(parent, "act_quant_kwargs", None)
    is_swizzled_scales = getattr(parent, "is_swizzled_scales", False)
    return MXTensor(
        qdata_slice,
        scale_slice,
        parent.elem_dtype,
        parent.block_size,
        parent.orig_dtype,
        kernel_preference,
        act_quant_kwargs,
        is_swizzled_scales,
    )


def selective_mx_weights_fwd(mx_param, active_experts: torch.Tensor) -> MXWeights:
    """Slice an MXFP4 expert parameter to the active set, keeping the K-axis
    block layout (FWD). The returned ``MXWeights.packed`` has shape
    ``[num_active, N, K/2]`` and is directly consumable by the forward MX
    kernel via ``parallel_linear_lora``."""
    MXTensor = _torchao_mxtensor_cls()
    if MXTensor is None:
        raise ImportError("MXFP4 fused path requires torchao>=0.7 (install `torchao`).")
    assert isinstance(mx_param, MXTensor), (
        f"selective_mx_weights_fwd expects an MXTensor, got {type(mx_param)}"
    )
    assert mx_param.elem_dtype == torch.float4_e2m1fn_x2, (
        "only MXFP4 (float4_e2m1fn_x2) is supported"
    )
    sub_qdata = _mx_qdata(mx_param)[active_experts].contiguous()
    sub_scale = _mx_scale(mx_param)[active_experts].contiguous()
    # Logical dims (kernel's K, N): the contraction axis is K, the OCP block
    # axis is the LAST storage axis (= K). N is the leading non-expert axis.
    N = sub_qdata.size(1)
    K = sub_qdata.size(2) * 2
    return MXWeights(
        packed=sub_qdata,
        scales=sub_scale,
        K=K,
        N=N,
        layout=MXLayout.FWD,
        block_size=mx_param.block_size,
        num_experts=sub_qdata.size(0),
        orig_dtype=mx_param.orig_dtype,
    )
