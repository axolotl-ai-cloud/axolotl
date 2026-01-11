"""
Axolotl GDPO Training Args

GDPO (Group Reward-Decoupled Normalization Policy Optimization) extends GRPO
with decoupled per-reward normalization for multi-reward RL training.
"""

from dataclasses import dataclass, field

from trl import GRPOConfig

from axolotl.core.training_args import AxolotlTrainingMixins


@dataclass
class AxolotlGDPOConfig(AxolotlTrainingMixins, GRPOConfig):
    """
    Axolotl GDPO Config for GDPO training.

    GDPO extends GRPO by normalizing each reward function independently before
    combining them, which preserves reward signal resolution in multi-reward scenarios.

    Attributes:
        context_parallel_size: Degree of sequence parallelism.
        gdpo_decoupled_norm: Enable decoupled per-reward normalization (GDPO's core feature).
        gdpo_batch_norm: Apply batch-wise normalization after combining advantages.
        gdpo_epsilon: Epsilon for numerical stability in GDPO normalization.
        gdpo_per_reward_scale: Scale each reward by its std before combining.
    """

    context_parallel_size: int | None = None

    # GDPO-specific parameters
    gdpo_decoupled_norm: bool = field(
        default=True,
        metadata={
            "help": "Enable decoupled per-reward normalization (GDPO's core feature). "
            "When True, each reward function is normalized independently before combining."
        },
    )
    gdpo_batch_norm: bool = field(
        default=False,
        metadata={
            "help": "Apply batch-wise normalization after combining advantages. "
            "Useful for stabilizing training with large batches."
        },
    )
    gdpo_epsilon: float = field(
        default=1e-4,
        metadata={"help": "Epsilon for numerical stability in GDPO normalization."},
    )
    gdpo_per_reward_scale: bool = field(
        default=True,
        metadata={
            "help": "Scale each reward by its std before combining. "
            "Only applied when scale_rewards is also True."
        },
    )
