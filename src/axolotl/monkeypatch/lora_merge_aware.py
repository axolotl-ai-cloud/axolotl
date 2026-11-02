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

import functools
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


def _full_tensor(tensor: torch.Tensor, device) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    if tensor.is_cpu and device is not None:
        tensor = tensor.to(device)
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def iter_merge_aware_vllm_params(vllm_generation):
    """Yield the model's weights under the names vLLM expects, merging each LoRA linear out
    of place from the full base weight and factors with ``merged_lora_weight``.

    Replaces TRL's FSDP2 sync, which merges the adapter into the sharded weights in place
    and unmerges afterwards: the sharded merge does not reproduce the merge-aware forward's
    rounding, and the round trip drifts the base weights. Every rank must iterate, since
    gathering a sharded tensor is a collective.
    """
    from peft.tuners.lora.layer import Linear as LoraLinear

    model = vllm_generation.model
    device = getattr(vllm_generation.accelerator, "device", None)
    lora_linears = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoraLinear)
    }
    with torch.no_grad():
        for name, param in model.named_parameters():
            vllm_name = name.removeprefix("base_model.model.").replace(
                ".base_layer", ""
            )
            if model.prefix in vllm_name or "original_module" in vllm_name:
                continue
            vllm_name = vllm_generation._fix_param_name_to_vllm(
                vllm_name, extra_prefixes=["modules_to_save.default."]
            )
            tensor = _full_tensor(param, device)
            owner, _, leaf = name.rpartition(".base_layer.")
            module = lora_linears.get(owner) if leaf == "weight" else None
            adapters = (
                [a for a in module.active_adapters if a in module.lora_A]
                if module is not None and not module.disable_adapters
                else []
            )
            for adapter in adapters:
                tensor = merged_lora_weight(
                    tensor,
                    _full_tensor(module.lora_A[adapter].weight, device),
                    _full_tensor(module.lora_B[adapter].weight, device),
                    module.scaling[adapter],
                    getattr(module, "_axolotl_lora_merge_aware_dtype", None)
                    or param.dtype,
                )
            yield vllm_name, tensor


def merge_aware_vllm_sync(vllm_generation) -> None:
    """Under FSDP2, make a TRL ``VLLMGeneration`` serve merge-aware LoRA weights computed
    out of place, exactly as the merge-aware forward computes them."""
    from peft.tuners.lora.layer import LoraLayer

    dist = getattr(vllm_generation, "_dist", None)
    model = getattr(vllm_generation, "model", None)
    if (
        model is None
        or not getattr(dist, "is_fsdp", False)
        or getattr(dist, "fsdp_version", None) != 2
    ):
        return
    lora_layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
    if not any(
        getattr(m, "_axolotl_lora_merge_aware_dtype", None) is not None
        for m in lora_layers
    ):
        return
    from peft.tuners.lora.layer import Linear as LoraLinear

    unsupported = sorted(
        {
            type(m).__name__
            for m in lora_layers
            if not isinstance(m, LoraLinear)
            or m.lora_variant
            or any(m.lora_bias.values())
        }
    )
    if unsupported:
        raise ValueError(
            "lora_merge_aware under FSDP2 merges LoRA linears out of place for the vLLM "
            f"sync; these LoRA layers cannot be merged that way: {', '.join(unsupported)}"
        )
    vllm_generation._iter_named_params = functools.partial(
        iter_merge_aware_vllm_params, vllm_generation
    )
