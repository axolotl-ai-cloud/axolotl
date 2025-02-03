"""
Axolotl Specific Training Args
"""
from trl import GRPOConfig

from axolotl.core.training_args import AxolotlTrainingMixins


class AxolotlGRPOConfig(AxolotlTrainingMixins, GRPOConfig):
    """
    Axolotl GRPO Config for GRPO training
    """
