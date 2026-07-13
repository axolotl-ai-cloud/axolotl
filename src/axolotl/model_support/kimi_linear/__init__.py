"""Kimi-Linear model support.

The full modeling, configuration, and tokenization code ships in this
directory; the descriptor's pre-load hooks redirect transformers' remote-code
loading to these in-tree copies.
"""

from axolotl.model_support.base import ModelSupport
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelMatchers,
    ModelProfile,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import VANILLA_CAUSAL_LM


def _matches_kimi_cfg(cfg) -> bool:
    return any(
        "kimi-linear" in (getattr(cfg, field, None) or "").lower()
        for field in ("base_model_config", "tokenizer_config")
    )


def _before_config_load(context: ModelHookContext) -> None:
    if "kimi-linear" in (getattr(context.cfg, "base_model_config", None) or "").lower():
        from .patch_kimi_linear import patch_kimi_config

        patch_kimi_config()


def _before_tokenizer_load(context: ModelHookContext) -> None:
    if "kimi-linear" in (getattr(context.cfg, "tokenizer_config", None) or "").lower():
        from .patch_kimi_linear import patch_kimi_tokenizer

        patch_kimi_tokenizer()


def _before_model_build(_context: ModelHookContext) -> None:
    from .patch_kimi_linear import patch_kimi_model

    patch_kimi_model()


@register_model_support
class KimiLinearSupport(ModelSupport):
    """Descriptor for Kimi-Linear (hybrid linear-attention remote-code model)."""

    model_types = ("kimi_linear",)
    profile = ModelProfile(
        family=VANILLA_CAUSAL_LM,
        matchers=ModelMatchers(cfg=_matches_kimi_cfg),
        hooks=ModelHooks(
            by_phase={
                ModelHookPhase.BEFORE_CONFIG_LOAD: (_before_config_load,),
                ModelHookPhase.BEFORE_TOKENIZER_LOAD: (_before_tokenizer_load,),
                ModelHookPhase.BEFORE_MODEL_BUILD: (_before_model_build,),
            }
        ),
    )
