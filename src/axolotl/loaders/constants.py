"""Shared constants for axolotl.loaders module"""

from transformers import (
    Gemma3ForConditionalGeneration,
    Llama4ForConditionalGeneration,
    LlavaForConditionalGeneration,
    Mistral3ForConditionalGeneration,
    MllamaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)

MULTIMODAL_AUTO_MODEL_MAPPING = {
    "mllama": MllamaForConditionalGeneration,
    "llama4": Llama4ForConditionalGeneration,
    "llava": LlavaForConditionalGeneration,
    "qwen2_vl": Qwen2VLForConditionalGeneration,
    "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
    "mistral3": Mistral3ForConditionalGeneration,
    "gemma3": Gemma3ForConditionalGeneration,
}
