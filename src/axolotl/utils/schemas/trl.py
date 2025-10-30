"""Pydantic models for TRL trainer configuration"""

from typing import Literal

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
