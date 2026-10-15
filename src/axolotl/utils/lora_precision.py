"""Precision policy for trainable LoRA parameters."""

import torch


def upcast_lora_parameters(model):
    """Keep parameter identity while promoting only trainable LoRA tensors."""
    for name, param in model.named_parameters():
        if param.requires_grad and any(
            part.startswith("lora_") for part in name.split(".")
        ):
            param.data = param.data.to(torch.float32)
            if param.grad is not None:
                param.grad = param.grad.to(torch.float32)


def lora_fsdp2_precision_policy(policy):
    """Preserve loaded parameter dtypes and reduce gradients in FP32."""
    from dataclasses import replace

    return replace(policy, param_dtype=None, reduce_dtype=torch.float32)


def configure_deepspeed_lora_precision(config):
    """Return an independent DeepSpeed config with FP32 accumulation/reduction."""
    import copy

    config = copy.deepcopy(config)
    zero = config.get("zero_optimization", {})
    offload = zero.get("offload_optimizer") or {}
    if zero.get("stage", 0) in (1, 2) and offload.get("device", "none") != "none":
        raise ValueError(
            "lora_fp32_gradients does not support DeepSpeed ZeRO-1/2 optimizer "
            "offload: its accumulation path retains low-precision gradients. "
            "Disable optimizer offload or use ZeRO-3."
        )
    config.setdefault("fp16", {})["fp16_master_weights_and_grads"] = False
    config.setdefault("data_types", {})["grad_accum_dtype"] = "fp32"
    config["communication_data_type"] = "fp32"
    bf16 = config.setdefault("bf16", {})
    bf16["bf16_master_weights_and_grads"] = False
    bf16["bf16_optimizer_states"] = False
    bf16["immediate_grad_update"] = True
    return config
