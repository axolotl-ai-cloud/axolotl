"""
Axolotl Specific Training Args
"""

from dataclasses import dataclass

from trl import GRPOConfig

from axolotl.core.trainers.async_grpo.config import (
    AxolotlAsyncGRPOConfig,  # noqa: F401
)
from axolotl.core.trainers.mixins.lora_merge_check import DEFAULT_MERGE_CHECK_INTERVAL
from axolotl.core.training_args import AxolotlTrainingMixins


@dataclass
class AxolotlGRPOConfig(AxolotlTrainingMixins, GRPOConfig):
    """Axolotl GRPO Config for GRPO training"""

    context_parallel_size: int | None = None
    vllm_merge_check_interval: int = DEFAULT_MERGE_CHECK_INTERVAL
