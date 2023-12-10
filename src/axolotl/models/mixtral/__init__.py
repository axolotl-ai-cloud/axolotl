"""
Custom modeling code for mixtral
"""

from .configuration_moe_mistral import MixtralConfig  # noqa
from .modeling_moe_mistral import (  # noqa
    MixtralForCausalLM,
    replace_mixtral_mlp_with_swiglu,
)
