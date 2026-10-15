"""Qwen4-Exp model support (Qwen3.8-Flash-Next: hybrid GDN/QSA + MoE + n-gram PLE)."""

from axolotl.model_support.base import ModelSupport, Unsupported
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelProfile,
    ModelStrategyOverrides,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import IMAGE_TEXT_TO_TEXT


def _get_processing_strategy_cls() -> type:
    from axolotl.processing_strategies import Qwen3_5ProcessingStrategy

    return Qwen3_5ProcessingStrategy


def _before_model_build(context: ModelHookContext) -> None:
    from axolotl.monkeypatch.models.qwen4_exp.modeling import (
        patch_qwen4_exp_modeling_packing,
        patch_qwen4_exp_qsa_indexer,
    )

    # the upstream QSA indexer loops in Python over every (batch, query) pair
    patch_qwen4_exp_qsa_indexer()

    if context.cfg.sample_packing:
        patch_qwen4_exp_modeling_packing()


@register_model_support
class Qwen4ExpSupport(ModelSupport):
    """Descriptor for Qwen4-Exp (Qwen3.8-Flash-Next)."""

    model_types = ("qwen4_exp", "qwen4_exp_text")
    profile = ModelProfile(
        family=IMAGE_TEXT_TO_TEXT,
        capabilities={
            "lora_kernels": Unsupported(
                "The GDN projections are named as in qwen3_5, so the linear-attention "
                "kernel rewrite matches on name but not on Qwen4Exp's forward."
            ),
            "sdpa_varlen": Unsupported(
                "The QSA layers overlay the indexer's token mask onto the causal "
                "mask, so the 4D mask cannot be dropped: varlen's mask builder "
                "returns None and the indexer has no mask to read."
            ),
            "liger": Unsupported(
                "FusedRMSNormGated hardcodes a silu gate, but Qwen4Exp's gate is "
                "configurable and Qwen3.8-Flash-Next sets output_gate_type=sigmoid; "
                "and LigerRMSNorm drops Qwen4ExpTextRMSNorm's group_size, which the "
                "PLE norms rely on."
            ),
        },
        strategies=ModelStrategyOverrides(
            processing_strategy_cls=_get_processing_strategy_cls,
        ),
        hooks=ModelHooks(
            by_phase={
                ModelHookPhase.BEFORE_MODEL_BUILD: (_before_model_build,),
            }
        ),
    )
