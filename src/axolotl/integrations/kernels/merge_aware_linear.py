"""Merge-aware LoRA forward for dequantized non-expert NVFP4 linears.

Checkpoints like nvidia/Qwen3-30B-A3B-NVFP4 quantize attention projections, not
just experts. The loader dequantizes those linears to dense bf16 (keeping the
original grid's per-tensor scale on the module), PEFT wraps them as ordinary
``lora.Linear`` layers, and at merge time the LoRA delta re-quantizes onto the
base NVFP4 grid, which erases most of it. When ``nvfp4_merge_aware`` is on,
this module replaces the wrapped forward with

    ``out = x @ Q(W + scaling * (B @ A))^T + bias``

where ``Q`` is the merge-identity quantizer over the frozen base per-tensor
scale, so training optimizes exactly the function the merged checkpoint will
compute. Gradient semantics match the expert path: the GEMM operand is the
snapped weight (``dx`` flows through it), while ``dA``/``dB`` flow through
``W_eff`` via the straight-through estimator.
"""

import types

import torch
import torch.nn.functional as F

from axolotl.utils.logging import get_logger

from .libs.sonicmoe.nvfp4_lora import merge_aware_enabled
from .libs.sonicmoe.nvfp4_quant import fake_quant_nvfp4_dispatch

LOG = get_logger(__name__)


def _merge_aware_lora_linear_forward(self, x, *args, **kwargs):
    if not merge_aware_enabled() or self.disable_adapters or self.merged:
        return self._ma_orig_forward(x, *args, **kwargs)
    adapters = [a for a in self.active_adapters if a in self.lora_A.keys()]
    if len(adapters) != 1:
        return self._ma_orig_forward(x, *args, **kwargs)
    adapter = adapters[0]

    base = self.get_base_layer()
    w = base.weight
    lora_a = self.lora_A[adapter].weight
    lora_b = self.lora_B[adapter].weight
    scaling = self.scaling[adapter]
    # delta in the base dtype, matching the merge writer's PEFT get_delta_weight
    w_eff = w + (lora_b @ lora_a).to(w.dtype) * scaling
    pts = base._nvfp4_pts.to(device=w_eff.device)
    w_fq = w_eff + (fake_quant_nvfp4_dispatch(w_eff.detach(), pts) - w_eff.detach())
    bias = None if base.bias is None else base.bias.to(x.dtype)
    result = F.linear(x, w_fq.to(x.dtype), bias)
    dropout = self.lora_dropout[adapter]
    if self.training and isinstance(dropout, torch.nn.Dropout) and dropout.p > 0:
        # dropout(x) != x only for the low-rank branch: out = x @ W_eff^T +
        # scaling * (dropout(x) - x) @ (BA)^T. The residual is zero-mean and
        # vanishes at eval, so the merged checkpoint still equals the eval
        # forward exactly; only the snapped term goes through Q.
        x_res = (dropout(x) - x).to(lora_a.dtype)
        result = result + scaling * F.linear(F.linear(x_res, lora_a), lora_b).to(
            result.dtype
        )
    return result


def install_merge_aware_lora_linears(model: torch.nn.Module) -> int:
    """Swap the forward of every PEFT ``lora.Linear`` whose base weight was
    dequantized from NVFP4 at load (marked by ``_nvfp4_pts``). Idempotent.
    Returns the number of wrapped modules."""
    from peft.tuners.lora.layer import Linear as LoraLinear

    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, LoraLinear):
            continue
        base = module.get_base_layer()
        if getattr(base, "_nvfp4_pts", None) is None:
            continue
        if hasattr(module, "_ma_orig_forward"):
            count += 1
            continue
        if any(module.use_dora.get(a) for a in module.lora_A.keys()):
            LOG.warning("merge-aware: skipping %s (DoRA is unsupported)", name)
            continue
        module._ma_orig_forward = module.forward  # type: ignore[assignment]
        module.forward = types.MethodType(  # type: ignore[method-assign]
            _merge_aware_lora_linear_forward, module
        )
        count += 1
    return count
