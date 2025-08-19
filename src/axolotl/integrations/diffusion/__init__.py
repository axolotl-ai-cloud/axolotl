"""Diffusion LM training plugin init."""

from transformers import AutoConfig, AutoModel

from .args import DiffusionArgs
from .configuration import DiffusionConfig, LlamaForDiffusionConfig, MistralForDiffusionConfig
from .models import LlamaForDiffusionLM, MistralForDiffusionLM
from .plugin import DiffusionPlugin

# Register custom configurations
AutoConfig.register("llama_diffusion", LlamaForDiffusionConfig)
AutoConfig.register("mistral_diffusion", MistralForDiffusionConfig)

# Register custom models
AutoModel.register(LlamaForDiffusionConfig, LlamaForDiffusionLM)
AutoModel.register(MistralForDiffusionConfig, MistralForDiffusionLM)

__all__ = [
    "DiffusionArgs", 
    "DiffusionPlugin",
    "DiffusionConfig",
    "LlamaForDiffusionConfig",
    "MistralForDiffusionConfig", 
    "LlamaForDiffusionLM",
    "MistralForDiffusionLM",
]
