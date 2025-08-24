"""
Configuration arguments for optimized MoE kernels
"""

from pydantic import BaseModel, Field


class MoeOptimizedArgs(BaseModel):
    """Arguments for optimized MoE kernel configuration"""

    moe_kernels: bool | None = Field(
        default=None,
        description="Enable optimized MoE kernels for faster MoE training",
    )

    moe_group_size: int | None = Field(
        default=128, description="Group size for contiguous grouped GEMM operations"
    )

    moe_persistent_kernel: bool | None = Field(
        default=True, description="Use persistent kernel with L2 cache optimization"
    )

    moe_kernel_models: list[str] | None = Field(
        default=None,
        description="List of models to apply MoE kernel optimization to (e.g., ['mixtral', 'qwen3_moe'])",
    )
