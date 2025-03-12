"""
GRPO specific configuration args
"""

from typing import Optional

from pydantic import BaseModel, Field


class TRLConfig(BaseModel):
    """
    Input args for TRL.
    """

    beta: Optional[float] = Field(
        default=None,
        json_schema_extra={"description": "Beta for RL training"},
    )
    max_completion_length: Optional[int] = Field(
        default=None,
        json_schema_extra={
            "description": "Maximum length of the completion for RL training"
        },
    )

    # GRPO specific args
    # Ref: https://github.com/huggingface/trl/blob/e3244d2d096ff1e2e248c931d06d39e165e20623/trl/trainer/grpo_config.py#L22
    use_vllm: Optional[bool] = Field(
        default=False,
        json_schema_extra={"description": "Whether to use VLLM for RL training"},
    )
    vllm_device: Optional[str] = Field(
        default="auto",
        json_schema_extra={"description": "Device to use for VLLM"},
    )
    vllm_gpu_memory_utilization: Optional[float] = Field(
        default=0.9,
        json_schema_extra={"description": "GPU memory utilization for VLLM"},
    )
    vllm_dtype: Optional[str] = Field(
        default="auto",
        json_schema_extra={"description": "Data type for VLLM"},
    )
    vllm_max_model_len: Optional[int] = Field(
        default=None,
        json_schema_extra={
            "description": "Maximum length of the model context for VLLM"
        },
    )

    reward_funcs: Optional[list[str]] = Field(
        default=None,
        json_schema_extra={"description": "List of reward functions to load"},
    )
    reward_weights: Optional[list[float]] = Field(
        default=None,
        json_schema_extra={
            "description": "Weights for each reward function. Must match the number of reward functions."
        },
    )
    num_generations: Optional[int] = Field(
        default=None,
        json_schema_extra={
            "description": "Number of generations to sample. The global batch size (num_processes * per_device_batch_size) must be divisible by this value."
        },
    )
    log_completions: Optional[bool] = Field(
        default=False,
        json_schema_extra={"description": "Whether to log completions"},
    )
    sync_ref_model: Optional[bool] = Field(
        default=False,
        json_schema_extra={
            "description": (
                "Whether to sync the reference model every `ref_model_sync_steps` "
                "steps, using the `ref_model_mixup_alpha` parameter."
            )
        },
    )
    ref_model_mixup_alpha: Optional[float] = Field(
        default=0.9,
        json_schema_extra={
            "description": "Mixup alpha for the reference model. Requires `sync_ref_model=True`."
        },
    )
    ref_model_sync_steps: Optional[int] = Field(
        default=64,
        json_schema_extra={
            "description": "Sync steps for the reference model. Requires `sync_ref_model=True`."
        },
    )
