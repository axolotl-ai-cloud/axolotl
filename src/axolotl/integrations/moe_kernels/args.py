"""Configuration arguments for MOE kernels integration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MoeKernelsArgs:
    """Arguments for MOE kernels integration."""
    
    moe_kernels_enabled: bool = False
    """Enable MOE kernels optimization"""
    
    moe_kernels_models: Optional[list[str]] = None
    """List of model types to apply MOE kernels to (e.g., ['deepseek_v3'])"""
    
    moe_kernels_group_size_m: int = 128
    """Group size for MOE kernels (alignment parameter)"""
    
    moe_kernels_persistent_kernel: bool = True
    """Whether to use persistent kernels for better performance"""
    
    moe_kernels_use_triton: bool = True
    """Whether to use Triton kernels for MOE operations"""
    
    moe_kernels_use_symmetric_memory: bool = True
    """Whether to use symmetric memory for MOE communication"""