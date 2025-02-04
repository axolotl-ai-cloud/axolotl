"""
GRPO specific configuration args
"""
from typing import List, Optional

from pydantic import BaseModel


class GRPOConfig(BaseModel):
    """
    Input args for GRPO.
    """

    grpo_use_vllm: Optional[bool] = False
    grpo_vllm_device: Optional[str] = "auto"
    grpo_vllm_gpu_memory_utilization: Optional[float] = 0.9
    grpo_reward_funcs: Optional[List[str]] = None
    grpo_num_generations: Optional[int] = None
