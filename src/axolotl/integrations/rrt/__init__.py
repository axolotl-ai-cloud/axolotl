"""
Axolotl Plugin for Relaxed Recursive Transformers
"""

import logging

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from axolotl.integrations.base import BasePlugin
from axolotl.integrations.rrt.modeling import register_rrt_model
from axolotl.integrations.rrt.modeling.modeling_rrt_llama import (
    RelaxedRecursiveLlamaConfig,
    RelaxedRecursiveLlamaForCausalLM,
    RelaxedRecursiveLlamaModel,
)

LOG = logging.getLogger(__name__)


class RelaxedRecursiveTransformerPlugin(BasePlugin):
    """
    Plugin for Relaxed Recursive Transformers integration with Axolotl
    """

    def get_input_args(self):
        return "axolotl.integrations.rrt.args.RelaxedRecursiveTransformerArgs"

    def register(self):
        LOG.info(
            "Registering Relaxed Recursive Transformers modeling with transformers"
        )
        register_rrt_model()


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
