"""Merge-aware LoRA forward for merged BF16 serving (``lora_merge_aware: bf16``).

Merged serving computes ``bf16(W + scaling * B @ A)``, which at RL learning rates
rounds away most of the LoRA update. This forward replaces each PEFT
``lora.Linear`` forward with

    ``out = x @ bf16(W + scaling * B @ A)^T + bias``

so training optimizes exactly the weights a merged sync or merged checkpoint
serves. The cast is a straight-through estimator: ``dA``/``dB`` are the ordinary
low-rank LoRA gradients, computed without materializing a dense ``dW``. The merged
weight is recomputed in backward instead of being saved for it.
"""

import types

import torch
import torch.nn.functional as F

from axolotl.utils.logging import get_logger
from axolotl.utils.lora_merge_precision import merged_lora_weight

LOG = get_logger(__name__)

_DTYPES = {"bf16": torch.bfloat16}
_FLOAT_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


class _MergedLoraLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, lora_a, lora_b, scaling, dtype):
        merged = merged_lora_weight(weight, lora_a, lora_b, scaling, dtype)
        ctx.save_for_backward(x, weight, lora_a, lora_b)
        ctx.scaling = scaling
        ctx.dtype = dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.linear(
                x, merged.to(x.dtype), None if bias is None else bias.to(x.dtype)
            )

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, lora_a, lora_b = ctx.saved_tensors
        grad_x = grad_bias = grad_a = grad_b = None
        with torch.autocast(device_type=grad_out.device.type, enabled=False):
            if ctx.needs_input_grad[0]:
                merged = merged_lora_weight(
                    weight, lora_a, lora_b, ctx.scaling, ctx.dtype
                )
                grad_x = (grad_out @ merged.to(grad_out.dtype)).to(x.dtype)
            grad_2d = grad_out.reshape(-1, grad_out.shape[-1])
            x_2d = x.reshape(-1, x.shape[-1]).to(grad_out.dtype)
            if ctx.needs_input_grad[2]:
                grad_bias = grad_2d.sum(0)
            if ctx.needs_input_grad[3]:
                grad_b_x = grad_2d @ lora_b.to(grad_2d.dtype)
                grad_a = ((grad_b_x.t() @ x_2d) * ctx.scaling).to(lora_a.dtype)
            if ctx.needs_input_grad[4]:
                x_a = x_2d @ lora_a.to(x_2d.dtype).t()
                grad_b = ((grad_2d.t() @ x_a) * ctx.scaling).to(lora_b.dtype)
        return grad_x, None, grad_bias, grad_a, grad_b, None, None


def _merge_aware_forward(self, x, *args, **kwargs):
    if (
        self.disable_adapters
        or self.merged
        or args
        or any(value is not None for value in kwargs.values())
    ):
        return self._axolotl_lora_merge_aware_orig_forward(x, *args, **kwargs)
    adapters = [a for a in self.active_adapters if a in self.lora_A]
    if len(adapters) != 1:
        if adapters and not getattr(self, "_axolotl_lora_merge_aware_warned", False):
            LOG.warning(
                "lora_merge_aware: multiple active adapters; this LoRA linear uses the "
                "ordinary (not merge-aware) forward."
            )
            self._axolotl_lora_merge_aware_warned = True
        return self._axolotl_lora_merge_aware_orig_forward(x, *args, **kwargs)
    adapter = adapters[0]
    base = self.get_base_layer()
    lora_a = self.lora_A[adapter]
    lora_b = self.lora_B[adapter]
    scaling = self.scaling[adapter]
    device_type = x.device.type
    compute_dtype = (
        torch.get_autocast_dtype(device_type)
        if torch.is_autocast_enabled(device_type)
        else x.dtype
    )
    result = _MergedLoraLinear.apply(
        x.to(compute_dtype),
        base.weight,
        base.bias,
        lora_a.weight,
        lora_b.weight,
        scaling,
        self._axolotl_lora_merge_aware_dtype,
    )
    dropout = self.lora_dropout[adapter]
    if self.training and isinstance(dropout, torch.nn.Dropout) and dropout.p > 0:
        # Only the low-rank branch sees dropout: out = x @ W_merged^T +
        # scaling * B @ A @ (dropout(x) - x). The residual is zero-mean and vanishes
        # in eval, where the forward equals the merged weights exactly.
        x_res = self._cast_input_dtype(dropout(x) - x, lora_a.weight.dtype)
        result = result + (lora_b(lora_a(x_res)) * scaling).to(result.dtype)
    return result


def _unsupported_reason(module) -> str | None:
    if "forward" in module.__dict__:
        return "another integration already replaced its forward"
    if module.fan_in_fan_out:
        return "fan_in_fan_out"
    if module.lora_variant or any(module.use_dora.values()):
        return "DoRA or a LoRA variant"
    if any(module.lora_bias.values()):
        return "lora_bias"
    base = module.get_base_layer()
    if not isinstance(base, torch.nn.Linear):
        return f"base layer is {type(base).__name__}"
    if getattr(base, "_nvfp4_pts", None) is not None:
        return "NVFP4 base, merged onto the NVFP4 grid (see nvfp4_merge_aware)"
    weight = base.weight
    if type(weight) not in (torch.Tensor, torch.nn.Parameter):
        return f"base weight is a {type(weight).__name__}"
    if weight.dtype not in _FLOAT_DTYPES:
        return f"base weight dtype is {weight.dtype}"
    if weight.requires_grad:
        return "base weight is trainable"
    return None


def install_lora_merge_aware_linears(
    model: torch.nn.Module, mode: str = "bf16"
) -> tuple[int, list[str]]:
    """Swap the forward of every supported PEFT ``lora.Linear`` in ``model``. Idempotent.

    Returns the number of merge-aware modules and ``"name (reason)"`` for each LoRA
    linear that keeps the ordinary forward.
    """
    from peft.tuners.lora.layer import Linear as LoraLinear

    dtype = _DTYPES[mode]
    installed = 0
    skipped = []
    for name, module in model.named_modules():
        if not isinstance(module, LoraLinear):
            continue
        if getattr(module, "_axolotl_lora_merge_aware_dtype", None) is not None:
            installed += 1
            continue
        reason = _unsupported_reason(module)
        if reason is not None:
            skipped.append(f"{name} ({reason})")
            continue
        module._axolotl_lora_merge_aware_orig_forward = module.forward
        module._axolotl_lora_merge_aware_dtype = dtype
        module.forward = types.MethodType(_merge_aware_forward, module)
        installed += 1
    return installed, skipped


def configure_lora_merge_aware(cfg, model) -> None:
    """Install the ``lora_merge_aware`` forward on a PEFT model, or raise if nothing
    could be made merge-aware."""
    mode = cfg.get("lora_merge_aware")
    if not mode:
        return
    installed, skipped = install_lora_merge_aware_linears(model, mode)
    if not installed:
        raise ValueError(
            f"lora_merge_aware: {mode} found no supported LoRA linear to train "
            f"merge-aware ({'; '.join(skipped[:5]) or 'no PEFT LoRA linears'})."
        )
    LOG.info("lora_merge_aware: %s forward on %d LoRA linears", mode, installed)
    if skipped:
        LOG.warning(
            "lora_merge_aware: %d LoRA linears keep the ordinary forward, so their "
            "merged %s weights can still lose the trained update: %s%s",
            len(skipped),
            mode,
            "; ".join(skipped[:5]),
            " ..." if len(skipped) > 5 else "",
        )
