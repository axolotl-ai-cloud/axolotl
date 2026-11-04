"""Training arguments of axolotl's async GRPO trainer."""

from __future__ import annotations

from dataclasses import dataclass, field

from axolotl.core.trainers.async_grpo.trl_compat import (
    SCORE_CENTERING,
    AsyncGRPOConfig,
)
from axolotl.core.trainers.mixins.lora_merge_check import (
    DEFAULT_MERGE_CHECK_INTERVAL,
)
from axolotl.core.training_args_base import AxolotlTrainingMixins


@dataclass
class AxolotlAsyncGRPOConfig(AxolotlTrainingMixins, AsyncGRPOConfig):
    """TRL's ``AsyncGRPOConfig`` plus axolotl's training fields and extensions."""

    reward_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Weight of each reward function. Defaults to 1.0 for every function."
        },
    )
    multi_objective_aggregation: str = field(
        default="sum_then_normalize",
        metadata={
            "help": "'sum_then_normalize' normalizes the weighted reward sum within each "
            "group; 'normalize_then_sum' (GDPO) normalizes each reward first."
        },
    )
    scale_rewards: str = field(
        default="group",
        metadata={
            "help": "'group' divides advantages by the group's reward std, 'none' only "
            "centers them."
        },
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={
            "help": "Drop completions that hit max_completion_length before packing."
        },
    )
    skip_zero_advantage_batches: bool = field(
        default=True,
        metadata={
            "help": "Drop zero-advantage samples (groups whose rewards are all equal) "
            "before packing; they would cost a forward and backward for no gradient."
        },
    )
    reward_num_workers: int = field(
        default=1,
        metadata={
            "help": "Processes that run synchronous reward functions, scoring as many "
            "groups at once. 1 keeps TRL's thread-based scoring."
        },
    )
    scoring_token_budget: int | None = field(
        default=None,
        metadata={
            "help": "Max tokens in one forward of a trainer-side scorer (EBFT's "
            "feature network); a row's scored sequences are split into as many "
            "forwards as needed. Defaults to token_budget; <= 0 scores each row in "
            "one forward."
        },
    )
    loss_type: str = field(
        default="dapo",
        metadata={
            "help": "'dapo' is TRL's objective, the PPO surrogate against the vLLM "
            "sampler's log-probabilities clipped at [1 - epsilon, 1 + epsilon_high]. "
            "'cispo' weights each token's policy gradient by the policy/sampler "
            "ratio, bounded at vllm_importance_sampling_cap, and clips nothing. "
            "'score_centering' (TRL main only) is TRL's score centering: the "
            "importance-weighted score minus its expectation over the sampler's "
            "num_sampler_logprobs most likely tokens, dropping tokens whose ratio "
            "falls outside [vllm_importance_sampling_clip_min, "
            "vllm_importance_sampling_clip_max]."
        },
    )
    vllm_importance_sampling_mode: str = field(
        default="token_truncate",
        metadata={
            "help": "With loss_type 'cispo': 'token_truncate' caps the weight at the "
            "cap, 'token_mask' drops tokens whose ratio exceeds the cap from the loss."
        },
    )
    vllm_importance_sampling_cap: float = field(
        default=3.0,
        metadata={"help": "With loss_type 'cispo': the bound on the weight."},
    )
    rollout_loop: str | None = field(
        default=None,
        metadata={
            "help": "Fully qualified name of an AxolotlRolloutLoop subclass to run in "
            "the rollout process instead of TRL's generate-and-score loop."
        },
    )
    rollout_loop_kwargs: dict | None = field(
        default=None,
        metadata={"help": "Extra keyword arguments for rollout_loop."},
    )
    vllm_merge_check_interval: int = field(
        default=DEFAULT_MERGE_CHECK_INTERVAL,
        metadata={
            "help": "With merged weight sync, measure how much of the LoRA update "
            "survives rounding the merged weights at the first sync and then every N "
            "optimizer steps (lora_merge/retained). 0 disables."
        },
    )
    vllm_lora_alias: str | None = field(
        default=None,
        metadata={
            "help": "With adapter-only weight sync, also serve each synced adapter "
            "under this fixed name, for clients that address vLLM by a fixed model "
            "name. Syncs then drain in-flight requests before switching."
        },
    )

    def __post_init__(self):
        if SCORE_CENTERING:
            # TRL main trains with score centering by default; loss_type decides here,
            # so that the rollout worker requests top log-probs only when it is used.
            self.score_centering = self.loss_type == "score_centering"
        super().__post_init__()
