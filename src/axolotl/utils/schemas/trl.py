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
    generation_batch_size: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Batch size for generation. Controls how many unique prompts are generated per step. Should be num_generations * data_parallel_size for full DP utilization."
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
    scale_rewards: bool | Literal["group", "batch", "none"] = Field(
        default=True,
        json_schema_extra={
            "description": "How to scale rewards by their standard deviation. One of "
            "'group' (per prompt-group, GRPO default), 'batch' (whole batch), or 'none'. "
            "Booleans are accepted for back-compat (True->'group', False->'none')."
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
        json_schema_extra={
            "description": "Deprecated by TRL: GRPO scores tokens with TRL's fused LM "
            "head, so this only applies Liger's layer kernels."
        },
    )
    loss_type: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Loss formulation to use. Synchronous GRPO: grpo, bnpo, "
            "dr_grpo, dapo, cispo, and the others TRL's GRPOTrainer supports. Async "
            "GRPO: dapo (default, TRL's surrogate clipped against the vLLM sampler), "
            "cispo (policy gradient weighted by the policy/sampler ratio, capped at "
            "vllm_importance_sampling_cap), or, experimentally and with TRL installed "
            "from main, score_centering (TRL's score centering with masked importance "
            "sampling)."
        },
    )
    mask_truncated_completions: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Whether to exclude truncated completions from loss calculation."
        },
    )

    # Entropy regularization (TRL >= 1.8)
    entropy_coef: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Static entropy regularization coefficient for GRPO, adding an "
            "entropy bonus to encourage exploration."
        },
    )
    use_adaptive_entropy: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Enable adaptive entropy regularization (Skywork-OR1 style), "
            "adjusting entropy_coef toward entropy_target."
        },
    )
    entropy_target: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Target entropy when use_adaptive_entropy is enabled."
        },
    )
    entropy_coef_delta: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Step size for adjusting entropy_coef toward entropy_target."
        },
    )
    entropy_coef_min: float | None = Field(
        default=None,
        json_schema_extra={"description": "Lower bound for the adaptive entropy_coef."},
    )
    entropy_coef_max: float | None = Field(
        default=None,
        json_schema_extra={"description": "Upper bound for the adaptive entropy_coef."},
    )

    # Bidirectional vLLM importance-sampling clip (TRL >= 1.6)
    vllm_importance_sampling_clip_min: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Lower clip bound for the vLLM/training importance-sampling "
            "ratio. Async GRPO with loss_type score_centering: tokens whose ratio is "
            "below it are dropped (TRL's default 0.5)."
        },
    )
    vllm_importance_sampling_clip_max: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Upper clip bound for the vLLM/training importance-sampling "
            "ratio. Async GRPO with loss_type score_centering: tokens whose ratio is "
            "above it are dropped (TRL's default 5.0)."
        },
    )
    num_sampler_logprobs: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Async GRPO with loss_type score_centering: how many of the "
            "vLLM sampler's most likely tokens, with their log-probs, are requested "
            "for every generated token to center the score over (TRL's default 8). "
            "The server's --max-logprobs bounds it."
        },
    )

    log_multimodal: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to log multimodal content (images, videos) alongside "
            "completions. Disable to reduce log size."
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
            "description": "Deprecated: the data producer trainer was removed. Maps to "
            "async_prefetch: true with max_staleness: 0, which discards in-flight "
            "rollouts at every weight sync; async_prefetch with the default "
            "max_staleness is faster."
        },
    )
    async_prefetch: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Overlap generation and training. With rl: grpo/gdpo this runs "
            "on TRL's AsyncGRPOTrainer: a vLLM server generates continuously while the "
            "trainer consumes scored samples up to max_staleness policy versions old. "
            "Structured EBFT runs on it too."
        },
    )
    vllm_sync_interval: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Sync model weights to vLLM every N optimizer steps (async mode only)."
        },
    )
    max_staleness: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Async GRPO: drop rollouts generated more than this many weight "
            "syncs ago. TRL defaults to 4."
        },
    )
    max_inflight_tasks: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Async GRPO: concurrent generation requests sent to vLLM. "
            "Defaults to max_staleness x samples per optimizer step."
        },
    )
    token_budget: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Async GRPO: max unique tokens packed into one rank's "
            "micro-batch. Defaults to the vLLM server's max_model_len; <= 0 packs "
            "micro_batch_size samples per rank instead."
        },
    )
    scoring_token_budget: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Async GRPO with rewards scored on the trainer (EBFT): max "
            "tokens in one scoring forward. A row's scored sequences (completions, "
            "references, whole conversations) are split into as many forwards as "
            "needed. Defaults to token_budget; <= 0 scores each row in one forward."
        },
    )
    vllm_importance_sampling_correction: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Apply IS correction for distribution mismatch between vLLM and "
            "training model. Async GRPO: only with loss_type cispo, which always "
            "weights each token by the policy/sampler ratio."
        },
    )
    vllm_importance_sampling_mode: (
        Literal["token_truncate", "token_mask", "sequence_truncate", "sequence_mask"]
        | None
    ) = Field(
        default=None,
        json_schema_extra={
            "description": "IS mode: token_truncate, token_mask, sequence_truncate, or "
            "sequence_mask. Async GRPO with loss_type cispo supports the token modes: "
            "token_truncate caps the weight, token_mask drops tokens beyond the cap."
        },
    )
    vllm_importance_sampling_cap: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Cap C for IS ratio clipping/masking. Async GRPO with "
            "loss_type cispo: the weight bound (default 3.0)."
        },
    )
    off_policy_mask_threshold: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "KL threshold for off-policy sequence masking (OPSM). None = "
            "disabled. Synchronous GRPO only."
        },
    )
    use_bias_correction_kl: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Apply IS correction to KL divergence term. Synchronous GRPO "
            "only."
        },
    )

    reward_num_workers: int = Field(
        default=1,
        json_schema_extra={
            "description": "Number of persistent subprocess workers for parallel reward computation. Each worker has its "
            "own main thread so signal.alarm() (used by math_verify) works correctly. Work is sharded across "
            "workers by prompt groups. Async GRPO, for synchronous reward functions."
        },
    )
    skip_zero_advantage_batches: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Async GRPO: drop samples whose advantage is zero (no learning "
            "signal) before packing, logged as sample/dropped_zero_adv_total. With "
            "rewards scored on the trainer they are taken out of the loss instead."
        },
    )
    vllm_lora_sync: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Sync LoRA adapter to vLLM via filesystem instead of merging + NCCL broadcast. "
            "Enables LoRA on the native vLLM server and registers each synced adapter under a versioned name via /v1/load_lora_adapter. "
            "Syncs only LoRA adapter weights vs full merged model. Async GRPO picks adapter-only sync "
            "whenever the server has LoRA enabled, which `axolotl vllm-serve` does when this is set."
        },
    )
    vllm_merge_check_interval: int | None = Field(
        default=None,
        ge=0,
        json_schema_extra={
            "description": "With LoRA and merged vLLM weight sync (vllm_lora_sync off), measure how much of "
            "the adapter update survives rounding the merged weights to vLLM's dtype at the first sync and then "
            "every N optimizer steps. Logged as lora_merge/retained, with a warning when most of the update is "
            "lost. Defaults to 50; 0 disables."
        },
    )
