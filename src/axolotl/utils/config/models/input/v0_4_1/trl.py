"""
GRPO specific configuration args
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class TRLConfig(BaseModel):
    """
    Input args for TRL.
    """

    beta: Optional[float] = None
    max_completion_length: Optional[int] = Field(
        default=None,
        json_schema_extra={
            "description": "Maximum length of the completion for RL training"
        },
    )

    # GRPO specific args
    use_vllm: Optional[bool] = False
    vllm_device: Optional[str] = "auto"
    vllm_gpu_memory_utilization: Optional[float] = 0.9
    vllm_max_model_len: Optional[int] = None
    vllm_dtype: Optional[str] = "auto"

    reward_funcs: Optional[List[str]] = None
    num_generations: Optional[int] = None
    log_completions: Optional[bool] = False

    sync_ref_model: Optional[bool] = False
    ref_model_mixup_alpha: Optional[float] = 0.9
    ref_model_sync_steps: Optional[int] = 64
