"""Precision policy for trainable LoRA parameters."""

import functools

import torch
from torch.utils._pytree import tree_leaves, tree_map_only


def upcast_lora_parameters(model):
    """Keep parameter identity while promoting only trainable LoRA tensors."""
    for name, param in model.named_parameters():
        if param.requires_grad and any(
            part.startswith("lora_") for part in name.split(".")
        ):
            param.data = param.data.to(torch.float32)
            if param.grad is not None:
                param.grad = param.grad.to(torch.float32)


def _compute_in_input_dtype(forward):
    """Run an fp32 module on fp32 inputs and cast outputs back to the caller's dtype."""

    @functools.wraps(forward)
    def wrapped(*args, **kwargs):
        dtype = next(
            (
                t.dtype
                for t in tree_leaves((args, kwargs))
                if torch.is_tensor(t) and t.is_floating_point()
            ),
            None,
        )
        if dtype is None or dtype == torch.float32:
            return forward(*args, **kwargs)
        args, kwargs = tree_map_only(
            torch.Tensor,
            lambda t: t.float() if t.is_floating_point() else t,
            (args, kwargs),
        )
        return tree_map_only(
            torch.Tensor,
            lambda t: t.to(dtype) if t.is_floating_point() else t,
            forward(*args, **kwargs),
        )

    return wrapped


def upcast_modules_to_save(model, embedding_modules):
    """Give trainable `modules_to_save` copies fp32 master weights; half precision drops
    most Adam updates (norms near 1.0 never move). Embeddings/lm_head are skipped: fp32
    vocab-sized copies are costly and CCE/Liger read lm_head.weight directly."""
    from peft.utils import ModulesToSaveWrapper

    for name, wrapper in model.named_modules():
        if (
            not isinstance(wrapper, ModulesToSaveWrapper)
            or name.rsplit(".", 1)[-1] in embedding_modules
            or isinstance(wrapper.original_module, torch.nn.Embedding)
        ):
            continue
        for module in wrapper.modules_to_save.values():
            params = [
                p
                for p in module.parameters()
                if p.requires_grad and p.dtype in (torch.float16, torch.bfloat16)
            ]
            for param in params:
                param.data = param.data.to(torch.float32)
            if params:
                module.forward = _compute_in_input_dtype(module.forward)


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
