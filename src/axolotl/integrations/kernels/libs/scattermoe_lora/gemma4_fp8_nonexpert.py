"""In-place fp8 per-channel quantization of gemma4 non-expert linears.

Targets every ``nn.Linear`` that is NOT inside a ``Gemma4TextExperts`` block
(experts stay NVFP4Tensor, untouched).  Norms, embeddings, router projections,
vision tower weights, and lm_head are also skipped by class or name.

After this runs each targeted linear has:
  - ``weight``           : fp8_e4m3fn  [out, in]   (1 byte/param)
  - ``weight_scale_inv`` : bfloat16    [out, 1]    (per-row scale = max_abs/fp8_max)
  - ``forward``          : overridden to dequant on-the-fly and call F.linear

The name ``weight_scale_inv`` matches the axolotl LoRA kernel's lookup so the
backward dequantize path works without further changes.

The dequant-in-forward path is the same as the DSV4 bf16-mode fallback —
frozen fp8 base, bf16 activations, bf16 LoRA adapters layered on top by PEFT.
The fused kernel rewrite (``torch._scaled_mm``) is a follow-up.

Usage (in plugin.py ``pre_lora_load``):
    from .gemma4_fp8_nonexpert import quantize_gemma4_nonexpert_linears
    quantize_gemma4_nonexpert_linears(model)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Names/substrings that should NEVER be fp8-quantized (precision-sensitive or tiny).
_SKIP_NAME_SUBSTRINGS = (
    "norm",
    "embed",
    "lm_head",
    "router",
    "vision_tower",
    "audio",
)

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _should_skip_by_name(name: str) -> bool:
    return any(sub in name for sub in _SKIP_NAME_SUBSTRINGS)


def _patch_linear_forward(mod: nn.Linear) -> None:
    """Monkey-patch a single nn.Linear whose weight was already replaced with
    an fp8 tensor + weight_scale_inv bfloat16 parameter [out, 1].

    The attribute name ``weight_scale_inv`` is chosen to match what the axolotl
    LoRA kernel (``lora.py:get_lora_parameters``) probes for on an fp8 weight::

        quant_state = getattr(base_layer, "weight_scale_inv", None)

    so the LoRA backward's ``dequantize(q_weight, q_quant)`` call receives the
    per-channel scale and correctly dequantizes to bf16.
    """

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        w_fp8: torch.Tensor = self.weight  # float8_e4m3fn [out, in]
        scale: torch.Tensor = self.weight_scale_inv  # bfloat16 [out, 1]
        w_bf16 = w_fp8.to(torch.bfloat16) * scale
        x_compute = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
        out = F.linear(x_compute, w_bf16, self.bias)
        return out.to(x.dtype) if x.dtype != torch.bfloat16 else out

    import types

    mod.forward = types.MethodType(_forward, mod)


def quantize_gemma4_nonexpert_linears(model: nn.Module) -> int:
    """Quantize all non-expert nn.Linear weights in a loaded gemma4 model to
    fp8_e4m3fn per-channel in-place.  Returns the number of modules quantized.

    Safe to call multiple times: already-quantized modules (weight.dtype ==
    float8_e4m3fn) are skipped.
    """
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

        _expert_cls = Gemma4TextExperts
    except ImportError:
        _expert_cls = None

    expert_paths: set[str] = set()
    if _expert_cls is not None:
        for path, mod in model.named_modules():
            if isinstance(mod, _expert_cls):
                expert_paths.add(path)

    def _is_under_expert(path: str) -> bool:
        return any(path == ep or path.startswith(ep + ".") for ep in expert_paths)

    quantized = 0
    by_type: dict[str, int] = {}

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if _is_under_expert(name):
            continue
        if _should_skip_by_name(name):
            continue
        w = mod.weight
        if not isinstance(w, nn.Parameter):
            continue
        if w.dtype == torch.float8_e4m3fn:
            continue  # already quantized
        if w.ndim != 2:
            continue  # embeddings are sometimes Linear subclasses

        w_data = w.data.to(torch.bfloat16) if w.dtype != torch.bfloat16 else w.data
        scale_1d = w_data.abs().amax(dim=1).clamp(min=1e-12)
        # "weight_scale_inv" name + [out, 1] shape match the axolotl LoRA kernel's fp8 quant_state lookup.
        scale = (scale_1d / _FP8_MAX).to(torch.bfloat16).unsqueeze(1)
        w_fp8 = (
            (w_data / scale.to(w_data.dtype))
            .clamp(-_FP8_MAX, _FP8_MAX)
            .to(torch.float8_e4m3fn)
        )

        mod.weight = nn.Parameter(w_fp8, requires_grad=False)
        mod.register_parameter(
            "weight_scale_inv", nn.Parameter(scale, requires_grad=False)
        )
        _patch_linear_forward(mod)

        key = type(mod).__name__
        by_type[key] = by_type.get(key, 0) + 1
        quantized += 1

    if quantized:
        LOG.info(
            "Gemma4 frankenstein: fp8-quantized %d non-expert linears in-place "
            "(per-channel e4m3, dequant-in-forward): %s",
            quantized,
            by_type,
        )
    return quantized


def verify_gemma4_frankenstein(model: nn.Module) -> dict:
    """Sanity-check the frankenstein state.  Returns a dict of counts."""
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        _has_nvfp4 = True
    except ImportError:
        _has_nvfp4 = False
        NVFP4Tensor = None

    n_nvfp4 = n_fp8 = n_bf16_lin = n_skip = 0
    fp8_bytes = bf16_lin_bytes = expert_bytes = 0

    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

        _expert_cls = Gemma4TextExperts
    except ImportError:
        _expert_cls = None

    expert_paths: set[str] = set()
    if _expert_cls is not None:
        for path, mod in model.named_modules():
            if isinstance(mod, _expert_cls):
                expert_paths.add(path)

    def _is_under_expert(path):
        return any(path == ep or path.startswith(ep + ".") for ep in expert_paths)

    for name, param in model.named_parameters():
        if _has_nvfp4 and isinstance(param, NVFP4Tensor):
            n_nvfp4 += 1
            expert_bytes += param.data.numel() * 1  # uint8
            continue
        if param.dtype == torch.float8_e4m3fn:
            n_fp8 += 1
            fp8_bytes += param.numel()
            continue
        if param.dtype in (torch.bfloat16, torch.float16, torch.float32):
            if name.endswith(".weight") and not _is_under_expert(
                name.rsplit(".", 1)[0]
            ):
                if not _should_skip_by_name(name):
                    n_bf16_lin += 1
                    bf16_lin_bytes += param.numel() * 2
                else:
                    n_skip += 1
            else:
                n_skip += 1

    return {
        "nvfp4_expert_params": n_nvfp4,
        "fp8_nonexpert_params": n_fp8,
        "bf16_linear_unexpected": n_bf16_lin,
        "skipped_params": n_skip,
        "expert_bytes_GB": expert_bytes / 1e9,
        "fp8_bytes_GB": fp8_bytes / 1e9,
        "bf16_linear_unexpected_bytes_GB": bf16_lin_bytes / 1e9,
    }
