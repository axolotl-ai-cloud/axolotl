# MOE Kernels Plugin

This plugin integrates optimized Mixture-of-Experts (MOE) kernels from TorchTitan into Axolotl, providing enhanced performance for MOE model training and inference.

## Features

- **Optimized Triton kernels** for token permutation and indexing
- **Symmetric memory operations** for distributed MOE communication  
- **Token dispatch and combine** kernels for efficient expert routing
- **Configurable alignment** and kernel parameters
- **DeepSeek V3 support** with extensible architecture for other models

## Usage

### Configuration

Add the MOE kernels configuration to your Axolotl config:

```yaml
# Enable MOE kernels optimization
moe_kernels_enabled: true

# Models to apply MOE kernels to
moe_kernels_models:
  - deepseek_v3

# Kernel configuration
moe_kernels_group_size_m: 128
moe_kernels_persistent_kernel: true
moe_kernels_use_triton: true
moe_kernels_use_symmetric_memory: true

# Register the plugin
plugins:
  - moe_kernels.plugin.MoeKernelsPlugin
```

### Programmatic Usage

```python
from axolotl.integrations.moe_kernels.plugin import apply_moe_kernel_patches

# Apply patches before model loading
apply_moe_kernel_patches(
    models=["deepseek_v3"],
    group_size_m=128,
    persistent_kernel=True,
    use_triton=True,
    use_symmetric_memory=True,
)
```

## Architecture

The plugin consists of several key components:

### Core Kernels

1. **`indices.py`** - Triton kernels for generating permutation indices and token alignment
2. **`dispatch.py`** - Token dispatcher for distributing tokens to experts across ranks
3. **`combine.py`** - Token combiner for gathering expert outputs back to original order

### Plugin Integration

- **`plugin.py`** - Main plugin class implementing Axolotl's BasePlugin interface
- **`args.py`** - Configuration arguments for the plugin

### Key Functions

- `generate_permute_indices()` - Creates optimized token permutation indices
- `TokenDispatcher` - Handles token distribution to experts
- `TokenCombiner` - Combines expert outputs back to tokens
- `apply_moe_kernel_patches()` - Applies optimizations to supported models

## Supported Models

Currently supported:
- **DeepSeek V3** - Full optimization support

Extensible architecture allows adding support for other MOE models.

## Performance Benefits

- **Faster token routing** through optimized Triton kernels
- **Reduced memory overhead** with symmetric memory operations
- **Better GPU utilization** with persistent kernels
- **Efficient distributed communication** for multi-GPU setups

## Requirements

- CUDA-capable GPU
- PyTorch with Triton support
- Distributed training setup (for symmetric memory features)

## Testing

Run the test suite to verify installation:

```bash
python test_moe_kernels_plugin.py
```

The test suite verifies:
- Plugin initialization and configuration
- Kernel module imports and functionality
- Triton kernel execution (GPU vs CPU consistency)
- Integration with Axolotl's plugin system

## Development

The plugin follows Axolotl's plugin architecture:

1. **`get_input_args()`** - Returns configuration class path
2. **`pre_model_load()`** - Applies patches before model loading

To extend support to new models:

1. Add model-specific patching logic in `plugin.py`
2. Create model-specific optimization functions
3. Update the `apply_moe_kernel_patches()` function
4. Add tests for the new model support

## Troubleshooting

### Common Issues

1. **CUDA not available**: Some optimizations require GPU support
2. **Triton import errors**: Ensure PyTorch installation includes Triton
3. **Model not supported**: Check `moe_kernels_models` configuration
4. **Distributed setup**: Symmetric memory features require proper distributed initialization

### Debug Mode

Enable debug logging to see optimization details:

```python
import logging
logging.getLogger('axolotl.integrations.moe_kernels.plugin').setLevel(logging.DEBUG)
```

## Contributing

When contributing new kernels or model support:

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new functionality
3. Update documentation with new features
4. Ensure compatibility with both single-GPU and distributed setups