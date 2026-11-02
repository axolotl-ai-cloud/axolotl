"""Fused LoRA for low-bit (NVFP4 / MXFP4 / MXFP8) attention projections.

When an MLA attention projection's frozen base weight is a torchao ``NVFP4Tensor`` / ``MXTensor``
(a ``nonexpert_quantization`` policy quantized it, or the checkpoint ships it low-bit), PEFT still
wraps the ``nn.Linear`` with LoRA but its default path can't run the base GEMM. This provides a
fused LoRA forward: dequantize the frozen 4-/8-bit weight transiently to the compute dtype (the
weight stays low-bit in memory), run ``X@W + s·X_drop@A@B``, with a custom autograd ``Function`` so
``dX`` and the LoRA grads flow (base frozen -> no weight grad). Marlin-W4A16-accelerated base GEMM
is a later optimization; this is the correct, memory-efficient path.
"""

from __future__ import annotations

import torch

from axolotl.kernels.lora import _apply_dropout, get_lora_parameters
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _dequantize(weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # torchao NVFP4Tensor / MXTensor both expose .dequantize()
    return weight.dequantize().to(dtype)


class _LowbitLora(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_drop, weight, A, B, s, bias, lora_bias):
        dtype = x.dtype
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        x_lora = x2 if x_drop is None else x_drop.reshape(-1, shape[-1])
        W = _dequantize(weight, dtype)  # [out, in]
        out = x2 @ W.t()
        if A is not None:
            xA = x_lora @ A.t().to(dtype)
            out = out.addmm_(xA, B.t().to(dtype), alpha=float(s))
            if lora_bias is not None:
                out = out + float(s) * lora_bias.to(dtype)
        else:
            xA = None
        if bias is not None:
            out = out + bias.to(dtype)
        ctx.save_for_backward(x_lora, weight, A, B, xA)
        ctx.s = float(s)
        ctx.shape = shape
        ctx.has_dropout = x_drop is not None
        ctx.has_lora_bias = lora_bias is not None
        return out.view(*shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad):
        x_lora, weight, A, B, xA = ctx.saved_tensors
        g = grad.reshape(-1, grad.shape[-1])
        W = _dequantize(
            weight, g.dtype
        )  # frozen base -> recompute (keeps weight low-bit)
        dX = g @ W
        dX_drop = dA = dB = d_lora_bias = None
        if A is not None:
            s = ctx.s
            dxA = g @ B.to(g.dtype)
            if ctx.has_dropout:
                dX_drop = (dxA @ A.to(g.dtype)).mul_(s).view(*ctx.shape)
            else:
                dX = dX.addmm_(dxA, A.to(g.dtype), alpha=s)
            dA = (dxA.t() * s) @ x_lora
            dB = (g.t() * s) @ xA
            if ctx.has_lora_bias:
                d_lora_bias = g.sum(dim=0) * s
        return dX.view(*ctx.shape), dX_drop, None, dA, dB, None, None, d_lora_bias


def _quantize_weight(w_bf16: torch.Tensor, kind: str):
    """Quantize a bf16 weight to a frozen torchao low-bit tensor for the fused-LoRA base GEMM."""
    if kind == "nvfp4":
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        pts = (w_bf16.abs().max() / (6.0 * 448.0)).reshape(()).float().clamp(min=1e-12)
        return NVFP4Tensor.to_nvfp4(
            w_bf16, per_tensor_scale=pts, is_swizzled_scales=False
        )
    from torchao.prototype.mx_formats.mx_tensor import MXTensor

    elem = torch.float4_e2m1fn_x2 if kind == "mxfp4" else torch.float8_e4m3fn
    return MXTensor.to_mx(w_bf16, elem_dtype=elem, block_size=32)


def quantize_attn_bases(model, kind: str, suffixes) -> int:
    """Post-PEFT-wrap: quantize the frozen MLA attention LoRA bases to nvfp4/mxfp4/mxfp8.

    Done AFTER ``get_peft_model`` (so PEFT wrapped a normal Linear rather than hijacking a
    torchao-subclass weight into its TorchaoLoraLinear path). The attention LoRA kernel then
    dispatches these bases to the fused low-bit path. ``kind`` in {nvfp4, mxfp4, mxfp8}.
    """
    import torch.nn as nn
    from peft.tuners.lora.layer import LoraLayer

    from axolotl.integrations.kernels.libs.mistral_large4.lora_attn import is_mla_proj
    from axolotl.integrations.kernels.libs.scattermoe_lora.selective_dequant import (
        is_mxfp4_param,
        is_nvfp4_param,
    )
    from axolotl.kernels.quantize import _is_float8_tensor, dequantize_fp8

    def _is_mxtensor(t):
        return type(t).__name__ == "MXTensor"

    n = skipped = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, LoraLayer) or not is_mla_proj(name, suffixes):
            continue
        base = mod.base_layer
        w = base.weight
        if is_nvfp4_param(w) or is_mxfp4_param(w) or _is_mxtensor(w):
            continue  # already a low-bit torchao tensor
        if not w.is_floating_point():
            # integer storage (e.g. bitsandbytes) needs its own scales to dequantize
            skipped += 1
            continue
        # get a bf16 view of the frozen base (dequant fp8 if needed)
        if (
            w.dtype == torch.float8_e4m3fn
            and getattr(base, "weight_scale_inv", None) is not None
        ):
            w_bf16 = dequantize_fp8(w, base.weight_scale_inv, torch.bfloat16)
        elif _is_float8_tensor(w):
            w_bf16 = w.dequantize().to(torch.bfloat16)
        else:
            w_bf16 = w.detach().to(torch.bfloat16)
        base.weight = nn.Parameter(_quantize_weight(w_bf16, kind), requires_grad=False)
        if hasattr(base, "weight_scale_inv"):
            base.weight_scale_inv = None
        n += 1
    if n:
        LOG.info("mistral_large4: quantized %d MLA attention bases to %s", n, kind)
    if skipped:
        LOG.warning(
            "mistral_large4: left %d integer-quantized MLA attention bases unquantized "
            "(%s needs a float or fp8 base)",
            skipped,
            kind,
        )
    return n


def lowbit_lora_forward(lora_layer, x, kind: str):
    """Fused LoRA forward for an NVFP4/MX base attention projection, or None to fall back.

    ``kind`` in {"nvfp4", "mxfp4", "mxfp8"} (already classified by the caller).
    """
    W, b, _qs, A, B, s, lb, drop, mag = get_lora_parameters(lora_layer)
    if mag is not None:  # DoRA: let the generic path handle it
        return None
    if not hasattr(W, "dequantize"):
        return None
    x_drop = _apply_dropout(drop, x, lora_layer.training)  # None when no dropout
    return _LowbitLora.apply(x, x_drop, W, A, B, s if A is not None else 0.0, b, lb)
