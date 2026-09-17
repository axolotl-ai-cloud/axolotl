"""Bailing hybrid (inclusionAI Ling 3.0) model support.

The published checkpoints ship inference-only remote code, so the pre-load
hooks redirect transformers' dynamic-module loading to the training-ready copy
in this directory.
"""

from axolotl.model_support.base import ModelSupport
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelMatchers,
    ModelProfile,
    ModelRegistrationOverrides,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import VANILLA_CAUSAL_LM

BAILING_PACKAGE = "axolotl.model_support.bailing_hybrid"
BAILING_MODULES = ("configuration_bailing_moe_v3", "modeling_bailing_moe_v3")
_BAILING_MARKERS = ("ling-3.0", "ling-3-", "bailing")


def _matches_bailing_cfg(cfg) -> bool:
    return any(
        marker in (getattr(cfg, field, None) or "").lower()
        for field in ("base_model_config", "tokenizer_config")
        for marker in _BAILING_MARKERS
    )


def _redirect_remote_code(_context: ModelHookContext) -> None:
    from axolotl.model_support.remote_code import redirect_dynamic_modules

    redirect_dynamic_modules(BAILING_PACKAGE, BAILING_MODULES)


def _reject_context_parallel(context: ModelHookContext) -> None:
    """Ring attention shards the sequence across ranks and nothing hands a KDA
    layer's recurrent state to the next one, so each rank would restart it from zero
    -- silently, with a loss curve that still looks reasonable."""
    if (context.cfg.context_parallel_size or 1) > 1:
        raise ValueError(
            "context_parallel_size > 1 is not supported for Ling 3.0 "
            "(bailing_hybrid): its linear-attention layers carry a recurrent state "
            "that is not exchanged across context-parallel ranks. "
            "Set context_parallel_size: 1."
        )


def _weight_conversions():
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    from transformers.core_model_loading import WeightRenaming

    # the fused-expert layout and router come from deepseek_v3; only the router
    # bias is spelled differently in a Bailing checkpoint
    return {
        "bailing_hybrid": get_checkpoint_conversion_mapping("deepseek_v3")
        + [WeightRenaming("mlp.gate.expert_bias", "mlp.gate.e_score_correction_bias")]
    }


@register_model_support
class BailingHybridSupport(ModelSupport):
    """Descriptor for Ling 3.0 (`bailing_hybrid`) hybrid linear-attention MoE."""

    model_types = ("bailing_hybrid",)
    profile = ModelProfile(
        family=VANILLA_CAUSAL_LM,
        matchers=ModelMatchers(cfg=_matches_bailing_cfg),
        registrations=ModelRegistrationOverrides(
            weight_conversions=_weight_conversions
        ),
        hooks=ModelHooks(
            by_phase={
                ModelHookPhase.BEFORE_CONFIG_LOAD: (_redirect_remote_code,),
                ModelHookPhase.CONFIGURE_RUN: (_reject_context_parallel,),
                ModelHookPhase.BEFORE_TOKENIZER_LOAD: (_redirect_remote_code,),
                ModelHookPhase.BEFORE_MODEL_BUILD: (_redirect_remote_code,),
            }
        ),
    )
