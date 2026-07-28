"""Eligibility gates and environment setup for the B200 factored LoRA MLP kernel.

Deliberately light on imports (torch only) so config validation can use these
helpers without pulling in triton/bitsandbytes.
"""

import os
import warnings

import torch
from torch import nn

# The certified cuBLAS workspace cap for the B200 factored kernel: the largest
# cap that keeps both certification axes -- speed (>1.0x vs the reference
# factored implementation at every grad-accum K) and memory (<=1.02x).
# ":16:8" is the bitwise-deterministic alternative with slightly lower memory
# but a narrower speed margin.
CUBLAS_WORKSPACE_CONFIG_RECOMMENDED = ":4096:2"


def maybe_set_cublas_workspace_config() -> None:
    """Best-effort application of the certified ``CUBLAS_WORKSPACE_CONFIG``.

    cuBLAS reads this env var once, at handle creation (the first cuBLAS op in
    the process), so the canonical place to set it is the process launch
    environment. This helper applies it when unset, warns when it may be too
    late, and never clobbers an explicit different value.
    """
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current == CUBLAS_WORKSPACE_CONFIG_RECOMMENDED:
        return
    if current is not None:
        warnings.warn(
            f"CUBLAS_WORKSPACE_CONFIG is already set to {current!r} (not the "
            f"recommended {CUBLAS_WORKSPACE_CONFIG_RECOMMENDED!r}) -- leaving "
            "your explicit choice untouched, but the certified memory figures "
            "for lora_mlp_kernel_b200 assume the recommended value.",
            stacklevel=2,
        )
        return
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG_RECOMMENDED
    if torch.cuda.is_initialized():
        warnings.warn(
            "CUBLAS_WORKSPACE_CONFIG was applied after CUDA initialization -- "
            "cuBLAS may already have created handles, in which case the "
            "setting has no effect. Set CUBLAS_WORKSPACE_CONFIG in the "
            "process launch environment (before starting python) for the "
            "certified lora_mlp_kernel_b200 memory figures to hold.",
            stacklevel=2,
        )


def is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)


def lora_mlp_b200_config_eligible(lora_config, activation) -> tuple[bool, str]:
    """Run-wide gate for the B200 factored MLP kernel, checked once at patch time.

    Returns (eligible, reason-if-not). The arch gate is mandatory: every
    certified number for this kernel is B200/sm_100-specific and tunings do
    not transfer between compute-capability classes.
    """
    if not is_sm100():
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        return False, (
            f"device compute capability {cap} is not (10, 0) (B200/sm_100); "
            "the kernel is certified on B200 only"
        )
    if activation != "silu":
        return False, f"activation {activation!r} is not silu/SwiGLU"
    if lora_config.lora_dropout > 0:
        return False, (
            f"lora_dropout={lora_config.lora_dropout} is not implemented by "
            "the B200 factored kernel"
        )
    if getattr(lora_config, "use_dora", False):
        return False, "DoRA is not supported by the B200 factored kernel"
    return True, ""


def lora_mlp_b200_module_supported(
    gate_proj: nn.Module, up_proj: nn.Module, down_proj: nn.Module
) -> tuple[bool, str]:
    """Per-MLP gate: bf16 dense base weights, LoRA adapters, no bias, no quant."""
    for name, proj in (
        ("gate_proj", gate_proj),
        ("up_proj", up_proj),
        ("down_proj", down_proj),
    ):
        if not hasattr(proj, "lora_A"):
            return False, f"{name} has no LoRA adapter"
        base_layer = proj.base_layer if hasattr(proj, "base_layer") else proj
        weight = base_layer.weight
        if getattr(weight, "quant_state", None) is not None:
            return False, f"{name} has a quantized base weight"
        if weight.dtype != torch.bfloat16:
            return False, f"{name} base weight dtype {weight.dtype} is not bfloat16"
        if base_layer.bias is not None:
            return False, f"{name} has a base bias"
        active_adapter = (
            proj.active_adapters[0]
            if hasattr(proj, "active_adapters")
            else proj.active_adapter
        )
        linear_a = proj.lora_A[active_adapter]
        linear_b = proj.lora_B[active_adapter]
        if linear_b.bias is not None:
            return False, f"{name} has a LoRA bias"
        if (
            linear_a.weight.dtype != torch.bfloat16
            or linear_b.weight.dtype != torch.bfloat16
        ):
            return False, f"{name} LoRA adapter dtype is not bfloat16"
    return True, ""
