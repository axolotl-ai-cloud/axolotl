# MoE Optimized Kernels

This module contains optimized Triton-based kernels for Mixture of Experts (MoE) models.

## Implementation

The core kernels are based on the contiguous grouped GEMM approach from PyTorch's TorchTitan project:
- **Source**: https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/kernels/triton_contiguous_group_gemm
- **Approach**: Contiguous grouped matrix multiplication for expert computations
- **Optimizations**: L2 cache optimization, persistent kernels, auto-tuning

## Key Features

- **Triton CUDA kernels** for high-performance expert computation
- **Token sorting** by expert assignment for memory coalescence
- **Grouped GEMM** operations to reduce kernel launch overhead
- **Auto-tuning** configurations for optimal performance across hardware
- **Persistent kernels** with L2 cache optimization

## Files

- `triton_kernels.py`: Core Triton kernel implementations
- `__init__.py`: Public API exports

## Usage

```python
from axolotl.kernels.moe import cg_grouped_gemm_forward

# Perform grouped GEMM for MoE forward pass
output = cg_grouped_gemm_forward(
    inputs,           # [M, K] input tensor
    expert_weights,   # [num_experts, N, K] expert weights
    expert_indices,   # [M] expert assignment per token
    group_size_m=128, # Group size for optimization
    persistent_kernel=True  # Use L2 cache optimization
)
```

## Performance

These kernels replace the slow for-loop expert access pattern in standard MoE implementations with fused operations, providing significant speedups especially for models with many experts.

## Credits

Based on the excellent work by the PyTorch TorchTitan team. Adapted for use in axolotl with additional model-specific optimizations.
