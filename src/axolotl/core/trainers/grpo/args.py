"""
Axolotl Specific Training Args
"""

from dataclasses import dataclass

from trl import GRPOConfig

from axolotl.core.training_args import AxolotlTrainingMixins


@dataclass
class AxolotlGRPOConfig(AxolotlTrainingMixins, GRPOConfig):
    """Axolotl GRPO Config for GRPO training"""

    context_parallel_size: int | None = None
