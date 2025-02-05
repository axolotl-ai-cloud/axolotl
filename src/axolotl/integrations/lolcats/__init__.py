"""
Module for the Plugin for LoLCATs linear attention integration with Axolotl.

Low-rank Linear Conversion via Attention Transfer
"""

import logging

from axolotl.integrations.base import BasePlugin
from axolotl.integrations.lolcats.trainer.distill_attention_xent_mse import (
    DistillAttentionXentMSETrainer,
)

from .args import LinearAttentionArgs  # pylint: disable=unused-import. # noqa: F401

LOG = logging.getLogger("axolotl.integrations.lolcats")


class LinearizePlugin(BasePlugin):
    """
    Plugin for lolcats integration with Axolotl.
    """

    def __init__(self):
        super().__init__()

        # Register the Linear Llama model with transformers
        from axolotl.integrations.lolcats.linear_llama.modeling_linear_llama import (
            register_linear_llama,
        )

        register_linear_llama()

    def get_input_args(self):
        return "axolotl.integrations.lolcats.LinearAttentionArgs"

    def get_trainer_cls(self, cfg):
        # defualt to XentMSE
        # TODO: add check to allow MSE_linear
        if cfg.linearize:
            return DistillAttentionXentMSETrainer

        return None
