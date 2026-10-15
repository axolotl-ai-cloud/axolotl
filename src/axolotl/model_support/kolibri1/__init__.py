"""Kolibri-1 (Aleph Alpha) model support.

The checkpoints ship no transformers modeling code, so the pre-load hooks
register the in-tree port with the transformers auto classes.
"""

import json
import os

from axolotl.model_support.base import ModelSupport, Unsupported
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelMatchers,
    ModelProfile,
    ModelRegistrationOverrides,
    ModelStrategyOverrides,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import VANILLA_CAUSAL_LM


def _matches_kolibri_cfg(cfg) -> bool:
    name = cfg.base_model_config or cfg.base_model or ""
    if "kolibri" in name.lower():
        return True
    config_path = os.path.join(name, "config.json")
    if not os.path.isfile(config_path):
        return False
    with open(config_path, encoding="utf-8") as f:
        return json.load(f).get("model_type") == "kolibri1"


def register_kolibri1(_context: ModelHookContext | None = None) -> None:
    from transformers import AutoConfig, AutoModelForCausalLM

    from .modeling_kolibri1 import Kolibri1Config, Kolibri1ForCausalLM

    AutoConfig.register("kolibri1", Kolibri1Config, exist_ok=True)
    AutoModelForCausalLM.register(Kolibri1Config, Kolibri1ForCausalLM, exist_ok=True)


def _lora_attention_cls(_cfg) -> type:
    from .modeling_kolibri1 import Kolibri1Attention

    return Kolibri1Attention


def _weight_conversions():
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    from transformers.core_model_loading import WeightRenaming

    return {
        "kolibri1": get_checkpoint_conversion_mapping("qwen3_moe")
        + [WeightRenaming("moe.router.expert_bias", "mlp.gate.e_score_correction_bias")]
    }


@register_model_support
class Kolibri1Support(ModelSupport):
    """Descriptor for Kolibri-1 (`kolibri1`) sliding-window MoE checkpoints."""

    model_types = ("kolibri1",)
    profile = ModelProfile(
        family=VANILLA_CAUSAL_LM,
        capabilities={
            "cut_cross_entropy": Unsupported(
                "The generic CCE patch imports transformers.models.<model_type>, "
                "which an in-tree port does not have."
            ),
        },
        strategies=ModelStrategyOverrides(lora_attention_cls=_lora_attention_cls),
        matchers=ModelMatchers(cfg=_matches_kolibri_cfg),
        registrations=ModelRegistrationOverrides(
            weight_conversions=_weight_conversions
        ),
        hooks=ModelHooks(
            by_phase={
                ModelHookPhase.BEFORE_CONFIG_LOAD: (register_kolibri1,),
                ModelHookPhase.BEFORE_TOKENIZER_LOAD: (register_kolibri1,),
                ModelHookPhase.BEFORE_MODEL_BUILD: (register_kolibri1,),
            }
        ),
    )
