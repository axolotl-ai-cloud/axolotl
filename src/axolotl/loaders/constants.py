"""Shared constants for axolotl.loaders module"""

from transformers import AutoModelForImageTextToText
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
)

MULTIMODAL_AUTO_MODEL_MAPPING = dict(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES)

MULTIMODAL_AUTO_MODEL_MAPPING["lfm2-vl"] = AutoModelForImageTextToText

try:
    from transformers import VoxtralForConditionalGeneration

    # transformers >4.53.2
    MULTIMODAL_AUTO_MODEL_MAPPING["voxtral"] = VoxtralForConditionalGeneration
except ImportError:
    pass
