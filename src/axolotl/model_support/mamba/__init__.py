"""Explicit native/FLA loading for pure Mamba language models."""

from axolotl.model_support.base import ModelSupport
from axolotl.model_support.profile import (
    ModelHookPhase,
    ModelHooks,
    ModelProfile,
    ModelStrategyOverrides,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import VANILLA_CAUSAL_LM


def _loader():
    from .loading import MambaModelLoader

    return MambaModelLoader


def _validate_adapters(context):
    config = getattr(context.model, "config", None)
    overrides = getattr(context.cfg, "overrides_of_model_config", None) or {}
    backend = getattr(config, "mamba_backend", overrides.get("mamba_backend"))
    if backend != "fla" or not getattr(context.cfg, "adapter", None):
        return
    if context.cfg.adapter != "lora":
        raise ValueError("FLA Mamba currently supports LoRA or full fine-tuning")
    if getattr(context.cfg, "lora_target_parameters", None):
        raise ValueError(
            "FLA Mamba LoRA requires module targets, not parameter targets"
        )
    if context.model is not None:
        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils.other import ModulesToSaveWrapper

        unsupported = [
            name
            for name, module in context.model.named_modules()
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper))
            and name.rsplit(".", 1)[-1] not in ("in_proj", "out_proj")
        ]
        if unsupported:
            raise ValueError(
                "FLA Mamba LoRA targets must be in_proj or out_proj; unsupported: "
                + ", ".join(unsupported)
            )


@register_model_support
class MambaSupport(ModelSupport):
    """Mamba checkpoints with an optional FLA implementation."""

    model_types = ("mamba", "mamba2")
    profile = ModelProfile(
        family=VANILLA_CAUSAL_LM,
        strategies=ModelStrategyOverrides(auto_model_cls=_loader),
        hooks=ModelHooks(
            by_phase={
                ModelHookPhase.CONFIGURE_RUN: (_validate_adapters,),
                ModelHookPhase.AFTER_ADAPTER_LOAD: (_validate_adapters,),
            }
        ),
    )
