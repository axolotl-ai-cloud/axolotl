"""
Axolotl Specific Training Args
"""

from dataclasses import dataclass, field
from typing import Any

from trl import GRPOConfig

from axolotl.core.trainers.grpo.fast_async_trainer import FastAsyncGRPOConfig
from axolotl.core.training_args import AxolotlTrainingMixins


@dataclass
class AxolotlGRPOConfig(AxolotlTrainingMixins, GRPOConfig):
    """Axolotl GRPO Config for GRPO training"""

    context_parallel_size: int | None = None
    # Async GRPO fields (from TRL rebase)
    use_data_producer: bool | None = None
    async_prefetch: bool | None = None
    streaming_partial_batch: bool | None = None
    replay_buffer_size: int | None = None
    replay_recompute_logps: bool | None = None
    reward_num_workers: int | None = None
    skip_zero_advantage_batches: bool | None = None
    streaming_min_groups: int | None = None
    reroll_max_groups: int | None = None
    reroll_start_fraction: float | None = None
    prefetch_depth: int | None = None
    off_policy_mask_threshold: float | None = None
    use_bias_correction_kl: bool | None = None
    vllm_importance_sampling_cap: float | None = None
    vllm_importance_sampling_correction: str | None = None
    vllm_importance_sampling_mode: str | None = None
    vllm_sync_interval: int | None = None
    vllm_lora_sync: bool | None = None
    use_liger_kernel: bool | None = None


@dataclass
class AxolotlAsyncGRPOConfig(AxolotlTrainingMixins, FastAsyncGRPOConfig):
    """Axolotl Async GRPO Config — adds async prefetch, streaming scoring, and IS correction."""

    context_parallel_size: int | None = None
