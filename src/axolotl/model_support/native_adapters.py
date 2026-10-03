"""Native diffusion-model PEFT guards."""

from __future__ import annotations

from typing import Any

_NATIVE_DIFFUSION_ATTN_IMPLS = {"eager", "sdpa", "flex_attention"}


def validate_native_diffusion_lora(cfg: Any, *, model_name: str) -> None:
    """Reject native adapter modes whose correctness has not been established."""

    if not getattr(cfg, "diffusion", None):
        return
    diffusion_cfg = cfg.diffusion
    from_causal_lm = (
        diffusion_cfg.get("from_causal_lm", False)
        if isinstance(diffusion_cfg, dict)
        else getattr(diffusion_cfg, "from_causal_lm", False)
    )
    attn_implementation = getattr(cfg, "attn_implementation", None)
    supported_attn_implementations = _NATIVE_DIFFUSION_ATTN_IMPLS | {"varlen"}
    if (
        not from_causal_lm
        and attn_implementation is not None
        and attn_implementation not in supported_attn_implementations
    ):
        raise ValueError(
            f"{model_name} native diffusion supports attn_implementation values "
            f"{sorted(supported_attn_implementations)}; got {attn_implementation!r}."
        )
    adapter = getattr(cfg, "adapter", None)
    if adapter not in {"lora", "qlora"}:
        raise ValueError(
            f"{model_name} native diffusion training currently requires adapter: lora; "
            f"got {adapter!r}."
        )
    if (
        adapter == "qlora"
        or getattr(cfg, "load_in_4bit", False)
        or getattr(cfg, "load_in_8bit", False)
    ):
        raise ValueError(
            f"{model_name} native diffusion LoRA does not support quantized adapters."
        )
    if getattr(cfg, "fsdp_config", None):
        raise ValueError(
            f"{model_name} native diffusion LoRA does not support FSDP yet."
        )
    if getattr(cfg, "peft_use_dora", False):
        raise ValueError(f"{model_name} native diffusion LoRA does not support DoRA.")
    if getattr(cfg, "peft_layer_replication", None):
        raise ValueError(
            f"{model_name} native diffusion LoRA does not support layer replication."
        )
    if getattr(cfg, "lora_target_parameters", None):
        raise ValueError(
            f"{model_name} native diffusion LoRA supports linear module targets only."
        )
    if getattr(cfg, "lora_target_linear", False):
        raise ValueError(
            f"{model_name} native diffusion LoRA requires explicit text projection targets; "
            "lora_target_linear could select vision or router projections."
        )

    targets = getattr(cfg, "lora_target_modules", None) or []
    if not targets:
        raise ValueError(
            f"{model_name} native diffusion LoRA requires explicit text attention or "
            "dense-MLP target paths."
        )
