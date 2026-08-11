"""Muse Glimmer model support (Gemma2-shaped text decoder + Kimi-K2.5-shaped vision tower)."""

from axolotl.model_support.base import ModelSupport, Supported, Unsupported
from axolotl.model_support.profile import (
    ModelMatchers,
    ModelProfile,
    ModelStrategyOverrides,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import IMAGE_TEXT_TO_TEXT


def _get_processing_strategy_cls() -> type:
    from .processing import MuseGlimmerProcessingStrategy

    return MuseGlimmerProcessingStrategy


def _matches_processor(processor) -> bool:
    try:
        from transformers.models.muse_glimmer import MuseGlimmerProcessor
    except ImportError:
        return False
    return isinstance(processor, MuseGlimmerProcessor)


@register_model_support
class MuseGlimmerSupport(ModelSupport):
    """Descriptor for Muse Glimmer."""

    model_types = ("muse_glimmer",)
    profile = ModelProfile(
        family=IMAGE_TEXT_TO_TEXT,
        capabilities={
            "cut_cross_entropy": Unsupported(
                "There is no MuseGlimmerForCausalLM for the llama-like fallback to patch, "
                "and the logits are scaled by output_multiplier then tanh-softcapped "
                "outside lm_head, which CCE's fused head does not reproduce."
            ),
            "liger": Supported(
                "GLU and vision-tower LayerNorm kernels only; RMSNorm, RoPE and FLCE are skipped."
            ),
            "lora_kernels": Unsupported(
                "The fused QKV/O source rewrite does not match MuseGlimmerTextAttention's "
                "forward, which sigmoid-gates the attention output through an extra "
                "self_attn.gate_proj."
            ),
        },
        strategies=ModelStrategyOverrides(
            processing_strategy_cls=_get_processing_strategy_cls,
        ),
        matchers=ModelMatchers(processor=_matches_processor),
    )
