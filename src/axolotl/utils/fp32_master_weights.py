"""Helpers for keeping fp32 master weights for trainable params under FSDP2."""

from __future__ import annotations

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

LOW_PRECISION_DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16, torch.float16)


def should_use_fp32_master_weights(cfg) -> bool:
    """Resolve whether trainable params get fp32 master copies under FSDP2.

    Accelerate applies this upcast itself for FSDP1 and DeepSpeed. Without it,
    Adam updates at typical SFT learning rates round away below bf16's mantissa.
    """
    explicit = getattr(cfg, "fp32_master_weights", None)
    if explicit is not None:
        return bool(explicit)

    if str(getattr(cfg, "fsdp_version", None)) != "2":
        return False
    if getattr(cfg, "adapter", None):
        # PEFT already keeps LoRA params in fp32; the base is frozen.
        return False
    if getattr(cfg, "fp8", None):
        # torchao float8 keeps its own high-precision weights.
        return False
    return True


def tag_model_fp32_master_weights(model: "torch.nn.Module", cfg) -> bool:
    """Attach the resolved fp32 master weight decision for FSDP2 prepare."""
    enabled = should_use_fp32_master_weights(cfg)
    if not enabled:
        if hasattr(model, "_axolotl_fp32_master_weights"):
            delattr(model, "_axolotl_fp32_master_weights")
        return False

    model._axolotl_fp32_master_weights = True
    return True


def upcast_master_weights(model: "torch.nn.Module") -> int:
    """Upcast low-precision trainable params to fp32 in place.

    Runs before ``fully_shard`` so the shards are fp32 and ``MixedPrecisionPolicy``
    casts back down for compute. Free under ``cpu_ram_efficient_loading``: the
    params are on meta, and the full state dict is cast per-param on load.
    """
    upcasted = 0
    for _name, param in model.named_parameters():
        if param.requires_grad and param.dtype in LOW_PRECISION_DTYPES:
            param.data = param.data.to(torch.float32)
            upcasted += 1

    if upcasted:
        LOG.info(
            "Upcast %d trainable param(s) to fp32 master weights; compute stays in "
            "the mixed-precision dtype, optimizer state and checkpoints become fp32. "
            "Set `fp32_master_weights: false` for pure bf16 and lower memory.",
            upcasted,
        )
    return upcasted
