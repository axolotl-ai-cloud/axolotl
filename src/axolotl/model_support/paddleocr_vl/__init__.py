"""PaddleOCR-VL model support (ERNIE-4.5 decoder + NaViT vision encoder)."""

from typing import TYPE_CHECKING

from axolotl.model_support.base import ModelSupport, Unsupported
from axolotl.model_support.registry import register_model_support

if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from axolotl.processing_strategies import ProcessingStrategy


@register_model_support
class PaddleOCRVLSupport(ModelSupport):
    """Descriptor for PaddleOCR-VL."""

    model_types = ("paddleocr_vl",)

    is_multimodal = True
    capabilities = {
        "cut_cross_entropy": Unsupported(
            "CCE does not patch PaddleOCRVLForConditionalGeneration."
        ),
        "liger": Unsupported(),
        "lora_kernels": Unsupported(
            "The fused QKV/O source rewrite does not match PaddleOCRAttention's forward."
        ),
    }

    def get_auto_model_cls(self) -> type:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText

    def get_processing_strategy_cls(self) -> type["ProcessingStrategy"]:
        from .processing import PaddleOCRVLProcessingStrategy

        return PaddleOCRVLProcessingStrategy

    def matches_processor(self, processor: "ProcessorMixin") -> bool:
        try:
            from transformers.models.paddleocr_vl import PaddleOCRVLProcessor
        except ImportError:
            return False
        return isinstance(processor, PaddleOCRVLProcessor)
