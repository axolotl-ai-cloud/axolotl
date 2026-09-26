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
    if backend == "fla" and getattr(context.cfg, "adapter", None):
        raise ValueError(
            "FLA Mamba currently supports full fine-tuning; its fused projections "
            "bypass adapter modules. Use mamba_backend: transformers for adapters."
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
