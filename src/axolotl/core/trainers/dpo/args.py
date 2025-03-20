"""
Axolotl specific DPO args
"""

from dataclasses import dataclass

from trl import DPOConfig

from axolotl.core.training_args import AxolotlTrainingMixins


@dataclass
class AxolotlDPOConfig(AxolotlTrainingMixins, DPOConfig):
    """
    DPO config for DPO training
    """
