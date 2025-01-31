"""
module for modeling relaxed recursive transformers model
"""
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_rrt_llama import RelaxedRecursiveLlamaConfig
from .modeling_rrt_llama import (
    RelaxedRecursiveLlamaForCausalLM,
    RelaxedRecursiveLlamaModel,
)


def register_rrt_model():
    """
    Register Relaxed Recursive Transformers model with transformers
    """

    # Register configs
    AutoConfig.register("llama-rrt", RelaxedRecursiveLlamaConfig)

    # Register models
    AutoModel.register(RelaxedRecursiveLlamaConfig, RelaxedRecursiveLlamaModel)
    AutoModelForCausalLM.register(
        RelaxedRecursiveLlamaConfig, RelaxedRecursiveLlamaForCausalLM
    )
