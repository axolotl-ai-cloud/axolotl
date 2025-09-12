# StableMax integration entry point

import torch

from axolotl.integrations.base import BasePlugin

from .args import StableMaxArgs  # noqa: F401
from .stablemax import stablemax_cross_entropy


class StableMaxPlugin(BasePlugin):
    """
    Plugin for StableMax integration with Axolotl.
    """

    def get_input_args(self):
        return "axolotl.integrations.stablemax.StableMaxArgs"

    def pre_model_load(self, cfg):
        """
        Patch the loss function to use StableMax cross-entropy if enabled.

        WARNING: This globally replaces torch.nn.functional.cross_entropy with
        stablemax_cross_entropy, affecting ALL subsequent calls to cross_entropy
        throughout the entire application. This includes calls from other libraries,
        models, and any code that uses torch.nn.functional.cross_entropy.

        Do not enable StableMax simultaneously with other cross-entropy patches
        such as Liger or CutCrossEntropy to avoid runtime conflicts.
        """
        if getattr(cfg, "stablemax", False):
            # Check for conflicts with other cross-entropy patches
            self._check_cross_entropy_conflicts(cfg)

            # Patch torch.nn.functional.cross_entropy to use stablemax_cross_entropy
            import torch.nn.functional as F

            F.cross_entropy = stablemax_cross_entropy

    def _check_cross_entropy_conflicts(self, cfg):
        """
        Check for conflicts with other integrations that patch cross_entropy.
        """
        conflicts = []

        # Check for Liger cross entropy
        if getattr(cfg, "liger_cross_entropy", False):
            conflicts.append("Liger cross_entropy")
        if getattr(cfg, "liger_fused_linear_cross_entropy", False):
            conflicts.append("Liger fused_linear_cross_entropy")

        # Check for CutCrossEntropy
        if getattr(cfg, "cut_cross_entropy", False):
            conflicts.append("CutCrossEntropy")

        if conflicts:
            raise ValueError(
                f"StableMax cannot be enabled simultaneously with other cross-entropy "
                f"patches: {', '.join(conflicts)}. Please disable one of these "
                f"integrations to avoid runtime conflicts."
            )
