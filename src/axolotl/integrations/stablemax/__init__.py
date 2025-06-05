# StableMax integration entry point

import torch
from axolotl.integrations.base import BasePlugin
from .stablemax import stablemax_cross_entropy
from .args import StableMaxArgs  # noqa: F401

class StableMaxPlugin(BasePlugin):
    """
    Plugin for StableMax integration with Axolotl.
    """

    def get_input_args(self):
        return "axolotl.integrations.stablemax.StableMaxArgs"

    def pre_model_load(self, cfg):
        """
        Patch the loss function to use StableMax cross-entropy if enabled.
        """
        if getattr(cfg, "stablemax", False):
            # Patch torch.nn.functional.cross_entropy to use stablemax_cross_entropy
            import torch.nn.functional as F
            F.cross_entropy = stablemax_cross_entropy
