"""
Module for the Plugin for LoLCATs linear attention integration with Axolotl.

Low-rank Linear Conversion via Attention Transfer
"""

import logging

from axolotl.integrations.base import BasePlugin
from axolotl.integrations.lolcats.trainer.distill_attention_xent_mse import (
    DistillAttentionXentMSETrainer,
)

LOG = logging.getLogger("axolotl.integrations.lolcats")


class LinearizePlugin(BasePlugin):
    """
    Plugin for lolcats integration with Axolotl.
    """

    def get_input_args(self):
        return "axolotl.integrations.lolcats.LinearAttentionArgs"

    def get_trainer_cls(self, cfg):
        # defualt to XentMSE
        # TODO: add check to allow MSE_linear
        if cfg.linearize:
            return DistillAttentionXentMSETrainer

        return None
