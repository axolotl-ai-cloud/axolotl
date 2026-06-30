"""
Axolotl Specific Training Args
"""

from dataclasses import dataclass, field

from trl import GRPOConfig

from axolotl.core.trainers.grpo.fast_async_trainer import FastAsyncGRPOConfig
from axolotl.core.training_args import AxolotlTrainingMixins


@dataclass
class AxolotlGRPOConfig(AxolotlTrainingMixins, GRPOConfig):
    """Axolotl GRPO Config for GRPO training"""

    context_parallel_size: int | None = None
    advantage_estimator: str = field(
        default="grpo",
        metadata={
            "help": "Advantage estimator: 'grpo' (group mean/std), 'rloo' "
            "(leave-one-out baseline), or 'reinforce_plus_plus' (group-mean "
            "baseline, batch-std normalization)."
        },
    )


@dataclass
class AxolotlAsyncGRPOConfig(AxolotlTrainingMixins, FastAsyncGRPOConfig):
    """Axolotl Async GRPO Config — adds async prefetch, streaming scoring, and IS correction."""

    context_parallel_size: int | None = None
    advantage_estimator: str = field(
        default="grpo",
        metadata={
            "help": "Advantage estimator: 'grpo' (group mean/std), 'rloo' "
            "(leave-one-out baseline), or 'reinforce_plus_plus' (group-mean "
            "baseline, batch-std normalization)."
        },
    )
