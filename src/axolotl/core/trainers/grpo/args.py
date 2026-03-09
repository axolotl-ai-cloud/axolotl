"""
Axolotl Specific Training Args
"""

from dataclasses import dataclass

from trl import GRPOConfig

from axolotl.core.training_args import AxolotlTrainingMixins
from axolotl.monkeypatch.trainer.async_grpo import AsyncGRPOConfig


@dataclass
class AxolotlGRPOConfig(AxolotlTrainingMixins, GRPOConfig):
    """Axolotl GRPO Config for GRPO training"""

    context_parallel_size: int | None = None


@dataclass
class AxolotlAsyncGRPOConfig(AxolotlTrainingMixins, AsyncGRPOConfig):
    """Axolotl Async GRPO Config — adds async prefetch, streaming scoring, and IS correction."""

    context_parallel_size: int | None = None
