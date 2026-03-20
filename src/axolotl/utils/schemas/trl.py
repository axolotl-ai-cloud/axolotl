"""Pydantic models for TRL trainer configuration"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class TRLConfig(BaseModel):
    """
    Input args for TRL.
    """

    beta: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Beta parameter for the RL training. Same as `rl_beta`. Use"
        },
    )
    max_completion_length: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Maximum length of the completion for RL training."
        },
    )

    # GRPO specific args
    # Ref: https://github.com/huggingface/trl/blob/26d86757a7c7e24e397ea44f57ecce6031dfac01/trl/trainer/grpo_config.py#L23
    use_vllm: bool = Field(
        default=False,
        json_schema_extra={"description": "Whether to use VLLM for RL training."},
    )
    vllm_mode: Literal["server", "colocate"] | None = Field(
        default=None,
        json_schema_extra={
            "description": "VLLM mode to use, one of 'server' or 'colocate'"
        },
    )
    vllm_server_host: str | None = Field(
        default="0.0.0.0",  # nosec B104
        json_schema_extra={"description": "Host of the vLLM server to connect to."},
    )
    vllm_server_port: int | None = Field(
        default=8000,
        json_schema_extra={"description": "Port of the vLLM server to connect to."},
    )
    vllm_server_timeout: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Total timeout (in seconds) to wait for the vLLM server to respond."
        },
    )
    vllm_guided_decoding_regex: str | None = Field(
        default=None,
        json_schema_extra={"description": "Regex for vLLM guided decoding."},
    )

    reward_funcs: list[str] | None = Field(
        default=None,
        json_schema_extra={
            "description": "List of reward functions to load. Paths must be importable from current dir."
        },
    )
    reward_weights: list[float] | None = Field(
        default=None,
        json_schema_extra={
            "description": "List of reward weights for the reward functions."
        },
    )
    num_generations: int | None = Field(
        default=None,
        json_schema_extra={"description": "Number of generations to sample."},
    )
    log_completions: bool | None = Field(
        default=False,
        json_schema_extra={"description": "Whether to log completions."},
    )
    num_completions_to_print: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Number of completions to print when log_completions is True."
        },
    )
    importance_sampling_level: Literal["sequence", "token"] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Controls whether importance sampling ratios are computed at the `'token'` or `'sequence'` level. "
            "For GSPO, use `sequence`, default is None which corresponds to the original GRPO paper."
        },
    )

    sync_ref_model: bool | None = Field(
        default=False,
        json_schema_extra={"description": "Whether to sync the reference model."},
    )
    ref_model_mixup_alpha: float | None = Field(
        default=0.9,
        json_schema_extra={"description": "Mixup alpha for the reference model."},
    )
    ref_model_sync_steps: int | None = Field(
        default=64,
        json_schema_extra={"description": "Sync steps for the reference model."},
    )
    scale_rewards: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Whether to scale rewards by their standard deviation."
        },
    )

    temperature: float | None = Field(
        default=None,
        json_schema_extra={"description": "Sampling temperature for the GRPO policy."},
    )
    top_p: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Top-p sampling probability for the generation policy."
        },
    )
    top_k: int | None = Field(
        default=None,
        json_schema_extra={"description": "Top-k sampling for the generation policy."},
    )
    min_p: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Minimum probability for the generation policy."
        },
    )
    repetition_penalty: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Penalty for tokens that appear in prompt and generated text."
        },
    )
    generation_kwargs: dict[str, Any] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Additional generation parameters passed to vLLM SamplingParams. "
            "Useful for stop_token_ids, seed, frequency_penalty, etc."
        },
    )
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Additional kwargs for the chat template. "
            "E.g., {enable_thinking: false} for Qwen3.5 models."
        },
    )
    num_iterations: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Number of iterations per batch (μ) for GRPO."
        },
    )
    epsilon: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Epsilon value for clipping in the GRPO algorithm."
        },
    )
    epsilon_high: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Upper-bound epsilon value for clipping in the GRPO algorithm."
        },
    )
    use_liger_loss: bool | None = Field(
        default=None,
        json_schema_extra={"description": "Whether to use Liger loss for GRPO."},
    )
    loss_type: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Loss formulation to use. Supported values: grpo, bnpo, dr_grpo."
        },
    )
    mask_truncated_completions: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Whether to exclude truncated completions from loss calculation."
        },
    )
    vllm_enable_sleep_mode: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Enable sleep mode for vLLM to offload VRAM when idle"
        },
    )
    rollout_func: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Path to custom rollout function. Must be importable from current dir."
        },
    )
    multi_objective_aggregation: (
        Literal["sum_then_normalize", "normalize_then_sum"] | None
    ) = Field(
        default=None,
        json_schema_extra={
            "description": "Multi-objective reward aggregation strategy. "
            "'sum_then_normalize' (GRPO default): weights and sums rewards first, then normalizes. "
            "'normalize_then_sum' (GDPO): normalizes each reward independently, then sums."
        },
    )

    # Async GRPO fields
    use_data_producer: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Use the GRPODataProducer protocol for online data generation."
        },
    )
    async_prefetch: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Generate rollouts in a background thread while training on the previous rollout."
        },
    )
    prefetch_depth: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Number of rollouts to prefetch ahead of training."
        },
    )
    vllm_sync_interval: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Sync model weights to vLLM every N optimizer steps (async mode only)."
        },
    )
    streaming_partial_batch: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Score prompt groups incrementally instead of the full batch at once."
        },
    )
    streaming_min_groups: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Minimum prompt groups to score per streaming chunk."
        },
    )
    vllm_importance_sampling_correction: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Apply IS correction for distribution mismatch between vLLM and training model."
        },
    )
    vllm_importance_sampling_mode: (
        Literal["token_truncate", "token_mask", "sequence_truncate", "sequence_mask"]
        | None
    ) = Field(
        default=None,
        json_schema_extra={
            "description": "IS mode: token_truncate, token_mask, sequence_truncate, or sequence_mask."
        },
    )
    vllm_importance_sampling_cap: float | None = Field(
        default=None,
        json_schema_extra={"description": "Cap C for IS ratio clipping/masking."},
    )
    off_policy_mask_threshold: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "KL threshold for off-policy sequence masking (OPSM). None = disabled."
        },
    )
    use_bias_correction_kl: bool | None = Field(
        default=None,
        json_schema_extra={"description": "Apply IS correction to KL divergence term."},
    )

    reward_num_workers: int = Field(
        default=1,
        json_schema_extra={
            "description": "Number of persistent subprocess workers for parallel reward computation. Each worker has its "
            "own main thread so signal.alarm() (used by math_verify) works correctly. Work is sharded across "
            "workers by prompt groups. Only used with use_data_producer=True and non-nn.Module reward functions."
        },
    )
    replay_buffer_size: int = Field(
        default=0,
        json_schema_extra={
            "description": "[Experimental, disabled by default] Size of the replay buffer for storing high-signal rollout "
            "groups. When > 0, groups with reward variance are cached and used to replace zero-signal groups "
            "(where all rewards are identical). Set to 0 to disable. Only used with use_data_producer=True."
        },
    )
    replay_recompute_logps: bool = Field(
        default=True,
        json_schema_extra={
            "description": "When True (default), recompute old_per_token_logps for replayed groups using the current "
            "training model. This fixes the importance sampling mismatch that occurs when replaying stale data. "
            "Only relevant when replay_buffer_size > 0."
        },
    )
    reroll_start_fraction: float = Field(
        default=1.0,
        json_schema_extra={
            "description": "Fraction of total training steps after which deferred re-rolling begins. Zero-signal prompts "
            "(where all rewards in a group are identical) are buffered and re-injected into later batches when the "
            "model is more likely to solve them. Set to 1.0 to disable. Only used with use_data_producer=True."
        },
    )
    reroll_max_groups: int = Field(
        default=1,
        json_schema_extra={
            "description": "Maximum number of prompt groups to replace with re-roll candidates per batch. Higher values "
            "increase data utilization but reduce prompt diversity. Only used with use_data_producer=True."
        },
    )
    skip_zero_advantage_batches: bool = Field(
        default=True,
        json_schema_extra={
            "description": "When True, skip gradient computation for micro-batches where all advantages are zero (no learning "
            "signal). This avoids the forward/backward pass entirely when no learning signal is present. The step is "
            "logged with skipped_zero_adv_batches=1 for monitoring."
        },
    )
    vllm_lora_sync: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Sync LoRA adapter to vLLM via filesystem instead of merging + NCCL broadcast. "
            "Auto-selects vllm_serve_lora serve module. Syncs only LoRA adapter weights vs full merged model."
        },
    )
