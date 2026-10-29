"""Fused, quantization-dispatching LoRA for the Mistral-Large-4 MLA attention projections.

The generic ``lora_kernels`` path assumes standard ``q/k/v/o_proj`` attention and cannot see MLA
(``q_proj`` + ``kv_a_proj_with_mqa`` + ``kv_b_proj`` + ``o_proj``, no ``k/v_proj``). This patcher
walks the PEFT-wrapped MLA projections and swaps their forward for a fused LoRA kernel that
dispatches the frozen base GEMM on the base weight's quantization:

  * **bf16 / nf4(bnb) / fp8-as-dequant** — the generic fused ``LoRA_O`` autograd kernel (a
    per-Linear ``X@W + s·X_drop@A@B`` with quant handled by ``dequantize``). Reused as-is.
  * **fp8-blockwise (native)** — ``_NativeFp8Lora`` (``fp8_linear`` / DeepGEMM 128×128 on
    sm90/sm100, Triton fallback on sm120): real fp8 throughput on the large projections, no
    dequant. This is the block-FP8 Mistral-Large-4 checkpoint's format.
  * **nvfp4 / mxfp4 / mxfp8** — dispatched to ``lora_lowbit`` (dequantize the frozen 4-/8-bit
    weight transiently → fused LoRA; weight stays low-bit in memory). Marlin-W4A16-accelerated
    base GEMM is a later optimization.

Mistral-Large-4's MLA has no gradientless indexer (unlike DSV4), so adapting all four projections
is safe.
"""

from __future__ import annotations

import types

from axolotl.kernels.lora import LoRA_O, _apply_dropout, get_lora_parameters
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# The MLA projections. q_proj is a single matrix (q_lora_rank=null); KV-source layers own
# kv_a_proj_with_mqa/kv_a_layernorm, all layers have kv_b_proj + o_proj.
MLA_PROJ_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.kv_a_proj_with_mqa",
    "self_attn.kv_b_proj",
    "self_attn.o_proj",
)


def is_mla_proj(name: str, suffixes=MLA_PROJ_SUFFIXES) -> bool:
    """Whether ``name`` is a text-decoder MLA projection (never the Pixtral vision tower)."""
    return "language_model.layers." in name and any(
        name.endswith(suf) for suf in suffixes
    )


# Large-output projections where native fp8 (vs dequant) is worth it. q_proj (H -> heads*qk_head)
# and kv_b_proj (kv_lora -> heads*(nope+v)) are the big ones; o_proj (heads*v -> H) is large on
# the input side. kv_a is small (H -> kv_lora+rope), a wash, so dequant is fine there.
_NATIVE_FP8_SUFFIXES = ("self_attn.q_proj", "self_attn.kv_b_proj", "self_attn.o_proj")


def _base_weight(base_layer):
    return getattr(base_layer, "weight", None)


def _is_fp8_base(base_layer) -> bool:
    from axolotl.integrations.kernels.libs.dsv4.lora_fp8 import _fp8_from_weight

    return _fp8_from_weight(_base_weight(base_layer), base_layer) is not None


def _quant_kind(base_layer) -> str:
    """Classify the frozen base weight's quantization for dispatch."""
    w = _base_weight(base_layer)
    if w is None:
        return "bf16"
    try:
        from axolotl.integrations.kernels.libs.scattermoe_lora.selective_dequant import (
            is_mxfp4_param,
            is_nvfp4_param,
        )

        if is_nvfp4_param(w):
            return "nvfp4"
        if is_mxfp4_param(w):
            return "mxfp4"
    except Exception:  # pragma: no cover - detection helpers optional
        pass
    if getattr(w, "__class__", type(None)).__name__ == "MXTensor":
        # mxfp8 weights carry an element dtype; treat e4m3 as mxfp8, e2m1 as mxfp4
        elem = getattr(w, "_elem_dtype", None) or getattr(w, "elem_dtype", None)
        return "mxfp8" if "e4m3" in str(elem) else "mxfp4"
    if _is_fp8_base(base_layer):
        return "fp8"
    if getattr(w, "quant_state", None) is not None:
        return "nf4"  # bnb 4-bit
    return "bf16"


def _generic_forward(self, x, *args, **kwargs):
    """bf16 / nf4 / fp8-as-dequant fused LoRA via the shared LoRA_O kernel."""
    W, b, qs, A, B, s, lb, drop, mag = get_lora_parameters(self)
    if A is None:  # disabled / merged adapter -> original PEFT forward
        return self._ml4_orig_forward(x, *args, **kwargs)
    # LoRA_O takes [batch, seq, in] only; expand_kv feeds kv_b_proj a [B, 1, S, kv_lora_rank] latent
    shape = x.shape
    x = x.reshape(-1, *shape[-2:]) if x.dim() > 2 else x.unsqueeze(0)
    X_drop = _apply_dropout(drop, x, self.training)
    out = LoRA_O.apply(x, X_drop, W, b, qs, A, B, s, lb, mag)
    return out.view(*shape[:-1], out.shape[-1])


def _make_forward(name: str, native_fp8: bool):
    is_native_fp8_target = native_fp8 and any(
        name.endswith(suf) for suf in _NATIVE_FP8_SUFFIXES
    )

    def forward(self, x, *args, **kwargs):
        kind = _quant_kind(self.base_layer)

        if kind == "fp8" and is_native_fp8_target:
            from axolotl.integrations.kernels.libs.dsv4.lora_fp8 import (
                _fp8_from_weight,
                _NativeFp8Lora,
            )

            W, b, _qs, A, B, s, _lb, drop, _mag = get_lora_parameters(self)
            fp8 = _fp8_from_weight(W, self.base_layer)
            # The blockwise-fp8 GEMM needs the contraction dim aligned to the block. Real
            # Mistral-Large-4 dims are all 128-multiples; guard so odd dims fall back cleanly.
            block_k = fp8[2][-1] if fp8 is not None else 128
            aligned = fp8 is not None and (W.shape[-1] % block_k == 0)
            if fp8 is not None and A is not None and aligned:
                qdata, scale_inv, block_size = fp8
                xin = drop(x) if drop is not None else x
                # PEFT keeps LoRA A/B in fp32; the fused fp8 GEMM contracts against the bf16
                # activation, so match the compute dtype (grads still flow to the fp32 masters).
                return _NativeFp8Lora.apply(
                    xin,
                    qdata,
                    scale_inv,
                    block_size,
                    A.to(xin.dtype),
                    B.to(xin.dtype),
                    s,
                    b.to(xin.dtype) if b is not None else None,
                )
            # unaligned dims / disabled adapter -> fused dequant path (still fp8-aware via LoRA_O)
            return _generic_forward(self, x, *args, **kwargs)

        if kind in ("nvfp4", "mxfp4", "mxfp8"):
            try:
                from .lora_lowbit import lowbit_lora_forward

                out = lowbit_lora_forward(self, x, kind)
                if out is not None:
                    return out
            except Exception as e:  # pragma: no cover - fall back, never break training
                LOG.warning_once(
                    "mistral_large4: low-bit (%s) fused attn LoRA unavailable (%s); "
                    "falling back to dequant path",
                    kind,
                    e,
                )
            # fall through to generic (dequant) if the low-bit kernel isn't available

        return _generic_forward(self, x, *args, **kwargs)

    return forward


def patch_mistral_large4_attn_lora(
    model, enabled: bool = True, native_fp8: bool = True
) -> int:
    """Swap MLA attention LoRA layers for the fused quant-dispatching LoRA forward.

    ``native_fp8`` gates the native-blockwise-fp8 kernel on the large projections; the other
    quantizations (bf16/nf4/nvfp4/mx) always route to their fused path. No-op unless ``enabled``.
    """
    if not enabled:
        return 0
    from peft.tuners.lora.layer import LoraLayer

    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, LoraLayer) or not is_mla_proj(name):
            continue
        # already patched, or carries the NVFP4 merge-aware forward
        if hasattr(mod, "_ml4_orig_forward") or hasattr(mod, "_ma_orig_forward"):
            continue
        mod._ml4_orig_forward = mod.forward
        mod.forward = types.MethodType(_make_forward(name, native_fp8), mod)
        n += 1
    if n:
        LOG.info(
            "mistral_large4: patched %d MLA attention projections with fused LoRA "
            "(native_fp8=%s)",
            n,
            native_fp8,
        )
    return n
