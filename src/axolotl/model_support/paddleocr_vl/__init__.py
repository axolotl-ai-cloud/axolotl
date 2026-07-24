"""PaddleOCR-VL model support (ERNIE-4.5 decoder + NaViT vision encoder)."""

from axolotl.model_support.base import ModelSupport, Unsupported
from axolotl.model_support.profile import (
    ModelMatchers,
    ModelProfile,
    ModelStrategyOverrides,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import IMAGE_TEXT_TO_TEXT


def _get_processing_strategy_cls() -> type:
    from .processing import PaddleOCRVLProcessingStrategy

    return PaddleOCRVLProcessingStrategy


def _matches_processor(processor) -> bool:
    try:
        from transformers.models.paddleocr_vl import PaddleOCRVLProcessor
    except ImportError:
        return False
    return isinstance(processor, PaddleOCRVLProcessor)


@register_model_support
class PaddleOCRVLSupport(ModelSupport):
    """Descriptor for PaddleOCR-VL."""

    model_types = ("paddleocr_vl",)
    profile = ModelProfile(
        family=IMAGE_TEXT_TO_TEXT,
        capabilities={
            "cut_cross_entropy": Unsupported(
                "CCE does not patch PaddleOCRVLForConditionalGeneration."
            ),
            "liger": Unsupported(),
            "lora_kernels": Unsupported(
                "The fused QKV/O source rewrite does not match PaddleOCRAttention's forward."
            ),
        },
        strategies=ModelStrategyOverrides(
            processing_strategy_cls=_get_processing_strategy_cls,
        ),
        matchers=ModelMatchers(processor=_matches_processor),
    )
