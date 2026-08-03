"""CohereCompass model support (Cohere North family: text decoder + SigLIP-shaped vision tower)."""

from axolotl.model_support.base import ModelSupport, Supported
from axolotl.model_support.profile import (
    ModelMatchers,
    ModelProfile,
    ModelStrategyOverrides,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import IMAGE_TEXT_TO_TEXT


def _get_processing_strategy_cls() -> type:
    from .processing import CohereCompassProcessingStrategy

    return CohereCompassProcessingStrategy


def _matches_processor(processor) -> bool:
    try:
        from transformers.models.cohere_compass import CohereCompassProcessor
    except ImportError:
        return False
    return isinstance(processor, CohereCompassProcessor)


@register_model_support
class CohereCompassSupport(ModelSupport):
    """Descriptor for CohereCompass."""

    model_types = ("cohere_compass",)
    profile = ModelProfile(
        family=IMAGE_TEXT_TO_TEXT,
        capabilities={
            "cut_cross_entropy": Supported(
                "Matches the unpatched loss to bf16 noise on North-Micro-Vision, including logit_scale."
            ),
            "lora_kernels": Supported(
                "The QKV/O source rewrite preserves the rope_on_all_layers and sliding-window branches."
            ),
            "liger": Supported(
                "RMSNorm, GLU and vision-tower LayerNorm kernels only; RoPE and FLCE are skipped."
            ),
        },
        strategies=ModelStrategyOverrides(
            processing_strategy_cls=_get_processing_strategy_cls,
        ),
        matchers=ModelMatchers(processor=_matches_processor),
    )
