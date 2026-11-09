"""Ricursive (Ripple) model support.

The published checkpoints ship inference-only remote code, so the pre-load
hooks redirect transformers' dynamic-module loading to the training-ready copy
in this directory.
"""

from axolotl.model_support.base import ModelSupport, Unsupported
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelMatchers,
    ModelProfile,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import VANILLA_CAUSAL_LM

RICURSIVE_PACKAGE = "axolotl.model_support.ricursive"
RICURSIVE_MODULES = ("configuration_ricursive", "modeling_ricursive")
_RICURSIVE_MARKERS = ("ricursive", "ripple")


def _matches_ricursive_cfg(cfg) -> bool:
    return any(
        marker in (getattr(cfg, field, None) or "").lower()
        for field in ("base_model_config", "tokenizer_config")
        for marker in _RICURSIVE_MARKERS
    )


def _redirect_remote_code(_context: ModelHookContext) -> None:
    from axolotl.model_support.remote_code import redirect_dynamic_modules

    redirect_dynamic_modules(RICURSIVE_PACKAGE, RICURSIVE_MODULES)


def _before_model_build(context: ModelHookContext) -> None:
    _redirect_remote_code(context)
    if getattr(context.cfg, "cut_cross_entropy", False):
        from cut_cross_entropy.transformers.patch import PATCH_FNS

        PATCH_FNS["ricursive"] = (
            "axolotl.model_support.ricursive.cut_cross_entropy",
            "patch_ricursive",
        )


@register_model_support
class RicursiveSupport(ModelSupport):
    """Descriptor for Ricursive Ripple (`ricursive`) dense checkpoints."""

    model_types = ("ricursive",)
    profile = ModelProfile(
        family=VANILLA_CAUSAL_LM,
        capabilities={
            "liger": Unsupported(
                "Liger has no ricursive entry, and its SwiGLU would drop the FFN clamp."
            ),
            "lora_kernels": Unsupported(
                "The fused MLP drops the FFN clamp, and the fused QKV feeds v_proj the "
                "normed input instead of the value-residual blend."
            ),
        },
        matchers=ModelMatchers(cfg=_matches_ricursive_cfg),
        hooks=ModelHooks(
            by_phase={
                ModelHookPhase.BEFORE_CONFIG_LOAD: (_redirect_remote_code,),
                ModelHookPhase.BEFORE_TOKENIZER_LOAD: (_redirect_remote_code,),
                ModelHookPhase.BEFORE_MODEL_BUILD: (_before_model_build,),
            }
        ),
    )
