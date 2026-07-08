# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
NVFP4 base-weight support for the sonicmoe MoE-LoRA backend.

Grouped up/down GEMMs and the gated activation that the LoRA forward composes
over frozen expert weights, plus NVFP4 detection and dequantization helpers. The
frozen base ``w1`` / ``w2`` may be a torchao ``NVFP4Tensor`` (block-scaled FP4).

Backends:
  - ``"torch"``: dense base, per-expert ``F.linear`` loop. Differentiable and
    CPU-correct (float64 for gradcheck); the reference path.
  - ``"dequant"``: dequantizes the NVFP4 base to dense per matmul, then runs the
    same math. The GPU fallback when the base is packed NVFP4.
  - ``"fp4_cute"``: in-kernel block-scaled W4A4 grouped GEMM on Blackwell
    (SM100/SM110) via ``fp4_cute_ops``. Weights stay packed; activations are
    NVFP4-quantized after grouping; backward dX runs per-expert chunked
    dequant matmuls (never through the packed fp4 operand).

Imports of ``sonicmoe``, ``torchao``, ``quack``, ``triton``, and CUDA-only ops
are lazy (inside functions) so this module imports and tests cleanly on CPU with
no GPU or upstream kernel packages.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _torchao_nvfp4tensor_cls():
    """Return torchao's ``NVFP4Tensor`` class, or ``None`` if torchao is absent."""
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    except ImportError:
        return None
    return NVFP4Tensor


def is_nvfp4_param(w) -> bool:
    """True iff ``w`` is a torchao ``NVFP4Tensor``, False for a dense tensor.

    Rolls its own torchao lookup rather than reusing
    ``scattermoe_lora.selective_dequant.is_nvfp4_param``: importing that module
    triggers the ``scattermoe_lora`` package ``__init__``, which imports triton
    at module load and crashes on a CPU/no-triton box.
    """
    NVFP4Tensor = _torchao_nvfp4tensor_cls()
    return NVFP4Tensor is not None and isinstance(w, NVFP4Tensor)


def dequantize_expert_weight(w: torch.Tensor) -> torch.Tensor:
    """Dequantize an NVFP4 expert weight to dense (bf16 on GPU), else identity.

    A plain dense tensor is returned unchanged so the reference/CPU path is a
    no-op.
    """
    if not is_nvfp4_param(w):
        return w
    if w.qdata.is_cuda:
        from .triton_nvfp4 import dequant_nvfp4_triton, triton_available

        if triton_available():
            # torchao's dequantize() gathers a value table per element (slow
            # enough to dominate the fp4_cute backward); this is one kernel.
            return dequant_nvfp4_triton(
                w.qdata, w.scale, w.per_tensor_scale, w.orig_dtype
            )
    return w.dequantize()


def dequantize_expert_slice(w: torch.Tensor, e: int) -> torch.Tensor:
    """Dense weight of expert ``e`` from an ``[E, dim1, dim2]`` stack.

    Keeps backward memory at one dense expert instead of E. torchao's
    NVFP4Tensor only implements rank-2 slicing, so the expert slice is rebuilt
    from components; dense tensors just index.
    """
    if not is_nvfp4_param(w):
        return w[e]
    assert not getattr(w, "is_swizzled_scales", False), (
        "swizzled NVFP4 scales unsupported (our loaders emit row-major)"
    )
    pts = w.per_tensor_scale
    sliced = type(w)(
        w.qdata[e : e + 1],
        w.scale[e : e + 1],
        w.block_size,
        w.orig_dtype,
        per_tensor_scale=pts[e : e + 1] if pts is not None else None,
    )
    return sliced.dequantize().squeeze(0)


def resolve_gated_activation(config) -> str:
    """Canonical gated-activation name from an HF text config.

    Honors ``hidden_activation`` (Gemma's key, e.g. ``gelu_pytorch_tanh``) before
    ``hidden_act`` (most other models). Reading only ``hidden_act`` would miss
    Gemma and silently run SwiGLU where the model wants tanh-GeGLU.
    """
    act = getattr(config, "hidden_activation", None) or getattr(
        config, "hidden_act", None
    )
    return (act or "silu").lower()


def gated_activation(
    h: torch.Tensor, act: str, *, concat: bool, limit: float | None = None
) -> torch.Tensor:
    """Apply a GLU gated activation to a ``[..., 2*I]`` pre-activation.

    Returns ``[..., I]``. ``concat=True`` (HF default) splits gate/up as the
    first/second half; ``concat=False`` (interleaved) takes even/odd lanes.

    Supported ``act`` (``up * f(gate)``):
      - swiglu / silu -> ``f = silu``
      - geglu / gelu -> ``f = gelu`` (erf)
      - gelu_tanh / gelu_pytorch_tanh -> ``f = gelu(approximate="tanh")`` (Gemma)
      - reglu / relu -> ``f = relu``

    ``limit`` (e.g. DeepSeek-V4 ``limit=10``) applies the model's clamped-SwiGLU
    bound before the nonlinearity (``gate.clamp(max=limit)``, ``up.clamp(-limit,
    limit)``), matching the eager experts' ``_apply_gate``; otherwise outliers on
    a clamped model blow up. The gate nonlinearity is computed in fp32 then cast
    back, matching upstream sonic-moe's ``_swiglu`` / ``_geglu``.
    """
    if concat:
        i = h.shape[-1] // 2
        gate = h[..., :i]
        up = h[..., i:]
    else:
        gate = h[..., 0::2]
        up = h[..., 1::2]

    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)

    # Upstream sonic-moe evaluates the gate nonlinearity in fp32 to protect bf16/fp16
    # precision. Only upcast lower-precision dtypes; leave fp32/fp64 as-is so we never
    # truncate (which would also break a float64 gradcheck).
    compute_dtype = (
        torch.float32 if gate.dtype in (torch.bfloat16, torch.float16) else gate.dtype
    )
    g = gate.to(compute_dtype)

    a = act.lower()
    if a in ("swiglu", "silu"):
        activated = F.silu(g)
    elif a in ("gelu_tanh", "gelu_pytorch_tanh"):
        activated = F.gelu(g, approximate="tanh")
    elif a in ("geglu", "gelu"):
        activated = F.gelu(g)
    elif a in ("reglu", "relu"):
        activated = F.relu(g)
    else:
        raise ValueError(f"unsupported gated activation: {act!r}")

    return up * activated.to(gate.dtype)


def _grouped_gemm(
    x_grouped: torch.Tensor,
    weight: torch.Tensor,
    expert_offsets: torch.Tensor,
    *,
    backend: str,
) -> torch.Tensor:
    """Per-expert ``x_e @ w[e].T`` over expert-sorted tokens.

    ``weight`` is ``[E, out, in]``; token rows in
    ``x_grouped[offsets[e - 1]:offsets[e]]`` (offsets[-1] == 0) go to expert e.
    """
    if backend == "fp4_cute":
        from .fp4_cute_ops import grouped_nvfp4_linear

        return grouped_nvfp4_linear(x_grouped, weight, expert_offsets)
    if backend == "dequant":
        weight = dequantize_expert_weight(weight)
    elif backend != "torch":
        raise ValueError(f"unknown backend: {backend!r}")

    from .nvfp4_lora import _use_grouped_mm

    if _use_grouped_mm(x_grouped) and weight.dtype == x_grouped.dtype:
        return torch._grouped_mm(
            x_grouped,
            weight.transpose(-2, -1),
            offs=expert_offsets.to(torch.int32),
        )

    E = weight.shape[0]
    outs = []
    start = 0
    offsets = expert_offsets.tolist()
    for e in range(E):
        end = offsets[e]
        outs.append(F.linear(x_grouped[start:end], weight[e]))
        start = end
    return torch.cat(outs, dim=0)


def grouped_up_gemm(
    x_grouped: torch.Tensor,
    w1: torch.Tensor,
    expert_offsets: torch.Tensor,
    *,
    backend: str,
    concat: bool,  # noqa: ARG001 (layout handled downstream in gated_activation)
) -> torch.Tensor:
    """Grouped gate/up projection.

    ``x_grouped``: ``[T_total, H]`` tokens gathered/sorted by expert.
    ``w1``: ``[E, 2*I, H]`` frozen base (dense for ``"torch"``, possibly NVFP4
    for ``"dequant"``). ``expert_offsets``: ``[E+1]`` int64 cumulative counts.
    Returns preact ``h``: ``[T_total, 2*I]`` where expert-e rows are
    ``x_e @ w1[e].T``.

    ``concat`` describes the gate/up interleaving of the ``2*I`` output columns;
    it is consumed later by :func:`gated_activation`, not here, since both
    layouts produce the same ``x_e @ w1[e].T`` product.
    """
    return _grouped_gemm(x_grouped, w1, expert_offsets[1:], backend=backend)


def grouped_down_gemm(
    a_grouped: torch.Tensor,
    w2: torch.Tensor,
    expert_offsets: torch.Tensor,
    *,
    backend: str,
) -> torch.Tensor:
    """Grouped down projection.

    ``a_grouped``: ``[T_total, I]``; ``w2``: ``[E, H, I]``; returns
    ``y``: ``[T_total, H]`` where expert-e rows are ``a_e @ w2[e].T``.
    """
    return _grouped_gemm(a_grouped, w2, expert_offsets[1:], backend=backend)
