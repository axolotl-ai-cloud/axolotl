"""Native blockwise-FP8 fused LoRA for DeepSeek-V4 non-expert projections.

PEFT's default path runs the frozen FP8 base as a dequant→bf16 GEMM (no fp8 throughput) and
the LoRA as a separate pass. For the *large-output* attention projections (``q_b_proj`` 1024→
32768, ``o_b_proj`` 8192→4096) a native blockwise-FP8 GEMM is 2.4–3.2× faster than the
dequant path (benched); the small ones (``q_a``/``kv``) are a wash, so we patch only the big
two.

``torch._scaled_mm`` (and DeepGEMM) have no autograd formula, so the native fp8 forward has to
live inside a custom ``autograd.Function``:
  * forward  — ``fp8_linear`` (DeepGEMM 128×128 on B200, Triton fallback elsewhere) for the
    frozen base, fused with the LoRA update via ``addmm_``;
  * backward — ``dX`` through a bf16 dequant of the weight (Option A: the block scale sits on
    the contraction axis, so an fp8 ``dX`` would need a transposed fp8 copy = 2× weight mem),
    plus the LoRA ``dA``/``dB``. The base weight is frozen (no weight grad).

Weight stays 128×128 blockwise (lossless vs the checkpoint); only the activation is quantized
(1×128), matching the DeepSeek/NVIDIA recipe. Gated by config ``dsv4_fp8_lora_kernel``.
"""

from __future__ import annotations

import types

import torch

from axolotl.kernels.lora import get_lora_parameters
from axolotl.kernels.quantize import dequantize_fp8
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Only the large-output projections where native fp8 beats the dequant GEMM.
_TARGET_SUFFIXES = ("self_attn.q_b_proj", "self_attn.o_b_proj")


def _fp8_linear():
    from transformers.integrations.finegrained_fp8 import fp8_linear

    return fp8_linear


class _NativeFp8Lora(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, qdata, scale_inv, block_size, A, B, scaling, bias):
        fp8_linear = _fp8_linear()
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        # frozen base: native blockwise fp8 GEMM (activation quantized 1x128 inside)
        out = fp8_linear(
            x2,
            qdata,
            scale_inv,
            block_size=list(block_size),
            bias=bias,
            output_dtype=x.dtype,
        )
        # fused LoRA update via addmm_ (avoids a [M,out] temp)
        xA = x2 @ A.t()
        out = out.addmm_(xA, B.t(), alpha=float(scaling))
        ctx.save_for_backward(x2, qdata, scale_inv, A, B, xA)
        ctx.scaling = float(scaling)
        ctx.block_size = tuple(block_size)
        ctx.shape = shape
        return out.view(*shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad):
        x2, qdata, scale_inv, A, B, xA = ctx.saved_tensors
        s = ctx.scaling
        g = grad.reshape(-1, grad.shape[-1])
        # dX through a bf16 dequant of W (Option A: block scale on the contraction axis, so an
        # fp8 dX would need a transposed fp8 weight copy = 2x weight mem). W transient.
        w_deq = dequantize_fp8(qdata, scale_inv, g.dtype)
        dxA = g @ B
        dX = (g @ w_deq).addmm_(dxA, A, alpha=s)
        dA = (dxA.t() * s) @ x2  # [r, in]
        dB = (g.t() * s) @ xA  # [out, r]
        return dX.view(*ctx.shape), None, None, None, dA, dB, None, None


def _fp8_from_weight(w, base_layer):
    """Return (qdata, scale_inv, block_size) for a Float8Tensor- or raw-FP8 weight, else None."""
    qdata = getattr(w, "qdata", None)
    if qdata is not None:  # torchao Float8Tensor
        return qdata, w.scale, list(getattr(w, "block_size", None) or [128, 128])
    if (
        isinstance(w, torch.Tensor) and w.dtype == torch.float8_e4m3fn
    ):  # raw transformers FP8
        si = getattr(base_layer, "weight_scale_inv", None)
        return (
            (w, si, list(getattr(base_layer, "block_size", None) or [128, 128]))
            if si is not None
            else None
        )
    return None


def _base_is_fp8(base_layer):
    return _fp8_from_weight(getattr(base_layer, "weight", None), base_layer) is not None


def _make_forward(lora_layer):
    def forward(self, x, *args, **kwargs):
        # get_lora_parameters unshards the LoRA A/B DTensors (FSDP2), grad-tracked. The base
        # Float8Tensor is all-gathered by FSDP before forward, so this reads the live weight.
        W, bias, _qs, A, B, scaling, _lora_bias, dropout, _mag = get_lora_parameters(
            self
        )
        fp8 = _fp8_from_weight(W, self.base_layer)
        if (
            fp8 is None or A is None
        ):  # disabled/merged adapter, or base not fp8: fall back
            return self._dsv4_orig_forward(x, *args, **kwargs)
        qdata, scale_inv, block_size = fp8
        xin = dropout(x) if dropout is not None else x
        return _NativeFp8Lora.apply(
            xin, qdata, scale_inv, block_size, A, B, scaling, bias
        )

    return forward


def patch_dsv4_attn_fp8_lora(model, enabled: bool = False) -> int:
    """Swap the forward of LoRA-wrapped large attention projections (q_b/o_b) on an FP8 base
    for the native-blockwise-fp8 fused-LoRA Function. No-op unless ``enabled``
    (config ``dsv4_fp8_lora_kernel``)."""
    if not enabled:
        return 0
    from peft.tuners.lora.layer import LoraLayer

    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, LoraLayer):
            continue
        if not any(name.endswith(suf) for suf in _TARGET_SUFFIXES):
            continue
        if not _base_is_fp8(mod.base_layer) or hasattr(mod, "_dsv4_orig_forward"):
            continue
        mod._dsv4_orig_forward = mod.forward  # fallback for disabled/merged adapter
        mod.forward = types.MethodType(_make_forward(mod), mod)
        n += 1
    if n:
        LOG.info(
            "Patched %d DeepSeek-V4 attention projections with native-fp8 fused LoRA", n
        )
    return n
