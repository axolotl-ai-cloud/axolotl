"""
Configuration arguments for optimized MoE kernels
"""

from typing import Optional

from pydantic import BaseModel, Field


class MoeOptimizedArgs(BaseModel):
    """Arguments for optimized MoE kernel configuration"""

    moe_kernels: Optional[bool] = Field(
        default=None,
        description="Enable optimized MoE kernels for faster MoE training",
    )

    moe_group_size: Optional[int] = Field(
        default=128, description="Group size for contiguous grouped GEMM operations"
    )

    moe_persistent_kernel: Optional[bool] = Field(
        default=True, description="Use persistent kernel with L2 cache optimization"
    )

    moe_kernel_models: Optional[list[str]] = Field(
        default=None,
        description="List of models to apply MoE kernel optimization to (e.g., ['mixtral', 'qwen3_moe'])",
    )
