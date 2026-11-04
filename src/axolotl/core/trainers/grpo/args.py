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


@dataclass
class AxolotlMOPDConfig(AxolotlGRPOConfig):
    """GRPO config for MOPD, which also allows ``num_generations=1``."""

    def __post_init__(self):
        try:
            super().__post_init__()
        except ValueError as err:
            # MOPD advantages are per-token rather than group-relative, so the
            # paper's single rollout per prompt is valid. TRL raises this check
            # after its batch-size setup, leaving only vLLM IS-clip sanity checks.
            if self.num_generations != 1 or "at least 2 generations" not in str(err):
                raise
