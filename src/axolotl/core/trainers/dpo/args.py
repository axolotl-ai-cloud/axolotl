"""
Axolotl specific DPO args
"""

from dataclasses import dataclass, field
from typing import Optional

from trl import DPOConfig

from axolotl.core.training_args import AxolotlTrainingMixins


@dataclass
class AxolotlDPOConfig(AxolotlTrainingMixins, DPOConfig):
    """
    DPO config for DPO training
    """

    dpo_norm_loss: bool | None = False
    rpo_alpha: Optional[float] = field(default=None)
