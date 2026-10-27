"""
Axolotl Specific Training Args
"""

from dataclasses import dataclass

from trl import GRPOConfig

from axolotl.core.trainers.grpo.fast_async_trainer import FastAsyncGRPOConfig
from axolotl.core.trainers.grpo.merge_check import DEFAULT_MERGE_CHECK_INTERVAL
from axolotl.core.training_args import AxolotlTrainingMixins


@dataclass
class AxolotlGRPOConfig(AxolotlTrainingMixins, GRPOConfig):
    """Axolotl GRPO Config for GRPO training"""

    context_parallel_size: int | None = None
    vllm_merge_check_interval: int = DEFAULT_MERGE_CHECK_INTERVAL


@dataclass
class AxolotlAsyncGRPOConfig(AxolotlTrainingMixins, FastAsyncGRPOConfig):
    """Axolotl Async GRPO Config — adds async prefetch, streaming scoring, and IS correction."""

    context_parallel_size: int | None = None
    vllm_merge_check_interval: int = DEFAULT_MERGE_CHECK_INTERVAL
