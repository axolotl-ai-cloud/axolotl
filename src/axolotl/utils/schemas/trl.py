"""Pydantic models for TRL trainer configuration"""

from pydantic import BaseModel, Field


class TRLConfig(BaseModel):
    """
    Input args for TRL.
    """

    beta: float | None = Field(
        default=None,
        json_schema_extra={"description": "Beta for RL training"},
    )
    max_completion_length: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Maximum length of the completion for RL training"
        },
    )

    # GRPO specific args
    # Ref: https://github.com/huggingface/trl/blob/26d86757a7c7e24e397ea44f57ecce6031dfac01/trl/trainer/grpo_config.py#L23
    use_vllm: bool = Field(
        default=False,
        json_schema_extra={"description": "Whether to use VLLM for RL training"},
    )
    vllm_server_host: str | None = Field(
        default="0.0.0.0",  # nosec B104
        json_schema_extra={"description": "Host of the vLLM server to connect to"},
    )
    vllm_server_port: int | None = Field(
        default=8000,
        json_schema_extra={"description": "Port of the vLLM server to connect to"},
    )
    vllm_server_timeout: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up "
            "after the timeout, a `ConnectionError` is raised."
        },
    )
    vllm_guided_decoding_regex: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled."
        },
    )

    reward_funcs: list[str] | None = Field(
        default=None,
        json_schema_extra={"description": "List of reward functions to load"},
    )
    reward_weights: list[float] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Weights for each reward function. Must match the number of reward functions."
        },
    )
    num_generations: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Number of generations to sample. The global batch size (num_processes * per_device_batch_size) must be divisible by this value."
        },
    )
    log_completions: bool | None = Field(
        default=False,
        json_schema_extra={"description": "Whether to log completions"},
    )
    sync_ref_model: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": (
                "Whether to sync the reference model every `ref_model_sync_steps` "
                "steps, using the `ref_model_mixup_alpha` parameter."
            )
        },
    )
    ref_model_mixup_alpha: float | None = Field(
        default=0.9,
        json_schema_extra={
            "description": "Mixup alpha for the reference model. Requires `sync_ref_model=True`."
        },
    )
    ref_model_sync_steps: int | None = Field(
        default=64,
        json_schema_extra={
            "description": "Sync steps for the reference model. Requires `sync_ref_model=True`."
        },
    )
    scale_rewards: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Whether to scale the rewards for GRPO by dividing them by their standard deviation."
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
            "description": "Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far."
        },
    )
    num_iterations: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Number of iterations per batch (denoted as μ in the algorithm) for GRPO."
        },
    )
    epsilon: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Epsilon value for clipping in the GRPO algorithm."
        },
    )
