# Optimized MoE Kernel Integration

This integration enables high-performance Mixture of Experts (MoE) kernels in axolotl, providing significant speedups for MoE model training.

## Features

- **Contiguous Grouped GEMM**: Optimized matrix multiplication for expert computations
- **Memory Optimization**: Token sorting for better memory coalescence
- **Auto-tuning**: Triton-based kernels with automatic performance tuning
- **Multi-model Support**: Works with Mixtral, Qwen3-MoE, and Qwen2-MoE models

## Performance Benefits

- Eliminates slow for-loop expert access pattern
- Improves memory coalescence by sorting tokens by expert assignment
- Reduces kernel launch overhead through grouped operations
- Better GPU utilization through optimized tiling strategies

## Configuration

Add the following to your axolotl configuration:

```yaml
# Enable optimized MoE kernels
moe_kernels: true

# Optional: Configure group size (default: 128)
moe_group_size: 128

# Optional: Use persistent kernels (default: true)
moe_persistent_kernel: true

# Optional: Specify models to patch (auto-detected by default)
moe_kernel_models:
  - mixtral
  - qwen3_moe
```

## Supported Models

- **Mixtral** (all variants)
- **Qwen3-MoE**
- **Qwen2-MoE**
- **DeepSeek-V3** (671B MoE with 256 experts)

The integration automatically detects the model type and applies appropriate patches.

## Requirements

- PyTorch >= 2.0
- Triton
- CUDA-capable GPU
- Transformers library with MoE model support

## Example Usage

```python
# In your training script or configuration
cfg = {
    'model_type': 'mixtral',
    'moe_kernels': True,
    'moe_group_size': 128,
    # ... other config options
}
```

## Implementation Notes

The integration patches the forward methods of MoE blocks at model load time, replacing the default implementation with optimized kernels. This happens transparently without requiring changes to the model code.

### Key Components

- `triton_kernels.py`: Core Triton kernel implementations (in `axolotl.kernels.moe`)
- `plugin.py`: Model patching logic and integration hooks
- `args.py`: Configuration schema

## Limitations

- Currently supports forward pass and weight gradients
- Input gradient computation not yet implemented
- Requires contiguous tensor inputs
- Best performance with power-of-2 group sizes

## Credits

Kernel implementations based on PyTorch TorchTitan's contiguous grouped GEMM approach.
