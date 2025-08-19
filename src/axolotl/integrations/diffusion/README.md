# Diffusion LM Training Plugin for Axolotl

This plugin enables diffusion language model training using the LLaDA (Large Language
And Diffusion Assistant) approach within the Axolotl framework.

## Overview

LLaDA is a diffusion-based approach to language model training that uses:
- **Random token masking** during training instead of next-token prediction
- **Bidirectional attention** to allow the model to see the full context
- **Importance weighting** based on masking probabilities for stable training

This approach can lead to more robust language models with better understanding of
bidirectional context.

## Installation

The plugin is included with Axolotl. To use it, simply add the plugin configuration to
your training config.

## Quickstart

### Basic Configuration

Add the following to your Axolotl configuration YAML:

```yaml
# Enable diffusion LM training plugin
plugins:
  - axolotl.integrations.diffusion.DiffusionPlugin

# Diffusion-specific configuration
noise_schedule: linear  # or "cosine"
min_mask_ratio: 0.1
max_mask_ratio: 0.9
num_diffusion_steps: 128
eps: 1e-3
importance_weighting: true
mask_token_id: 128002

# Sample generation (optional)
generate_samples: true
generation_interval: 100
num_generation_samples: 3
generation_steps: 128
generation_temperature: 0.0
generation_max_length: 100

# Model configuration
base_model: meta-llama/Llama-3.2-1B
model_type: llama

# Standard Axolotl configuration
datasets:
  - path: your_dataset
    ...

# Other config
sequence_len: 1024
micro_batch_size: 8
gradient_accumulation_steps: 4
learning_rate: 3e-4
```

## Supported Models

Currently supported base model types:
- **Llama** (meta-llama/Llama-*, etc.) - Uses `LlamaForDiffusionLM`
- **Mistral** (mistralai/Mistral-*, etc.) - Uses `MistralForDiffusionLM`

The plugin automatically creates custom model classes that inherit from the base model
while adding diffusion training capabilities. This provides full compatibility with
HuggingFace's ecosystem for saving, loading, and inference.

## How It Works

### Custom Model Architecture

The plugin creates custom model classes (`LlamaForDiffusionLM`, `MistralForDiffusionLM`) that inherit from
standard HuggingFace models. During training, these models:

1. **Apply forward diffusion process**: Randomly mask tokens based on sampled timesteps
2. **Use bidirectional attention**: Override causal attention with full bidirectional attention
3. **Compute diffusion loss**: Calculate loss only on masked tokens with optional importance weighting

### Random Masking
During training, tokens are randomly masked based on a sampled timestep:
- Sample timestep `t` uniformly from [0, 1]
- Calculate masking probability: `p = (1 - eps) * t + eps`
- Randomly mask tokens with probability `p`

### Bidirectional Attention
The models override causal attention with bidirectional attention:
- Creates 4D attention masks allowing all-to-all attention
- Maintains proper padding and sample packing masks
- Compatible with standard HuggingFace attention implementations

### Diffusion Loss

Loss is computed only on masked tokens with (optional) importance weighting:

```python
loss = sum(cross_entropy(pred, target) / p_mask) / total_tokens
```

### Model Loading and Saving

The custom models work seamlessly with HuggingFace's AutoModel system:

```python
from transformers import AutoModel, AutoConfig

# Load a diffusion model
model = AutoModel.from_pretrained("path/to/diffusion/model", trust_remote_code=True)

# Save a diffusion model
model.save_pretrained("path/to/save/diffusion/model")
```

During inference, the models behave like standard causal language models.

## Sample Generation

When `generate_samples: true`, the plugin generates samples during training:

```
Sample 1:
   Original (45 tokens): The quick brown fox jumps over the lazy dog...
   Masked (18/45 tokens, 40.0%): The [MASK] [MASK] fox [MASK] over [MASK] lazy [MASK]...
   Generated: The quick brown fox jumps over the lazy dog...
```

Samples are logged to console and wandb (if enabled).

## Metrics and Monitoring

The plugin adds several metrics to track diffusion training:

- `train/loss`: Weighted diffusion loss
- `train/accuracy`: Accuracy on masked tokens
- `train/mask_ratio`: Average fraction of tokens masked
- `train/num_masked_tokens`: Number of tokens masked
- `train/avg_p_mask`: Average masking probability
- `train/ce_loss`: Unweighted cross-entropy loss
- `train/importance_weight_avg`: Average importance weight

## Benefits of Custom Model Approach

✅ **Type Safety**: Full IDE support and type checking  
✅ **HuggingFace Integration**: Works with AutoModel, Hub, pipelines  
✅ **Maintainability**: Clean architecture, no monkey patching  
✅ **Ecosystem Compatibility**: Standard save/load, PEFT support  
✅ **Testing**: Easier to test and debug  

## Limitations

- **Model Support**: Currently limited to Llama and Mistral architectures
- **Flash Attention**: Not yet optimized for flash attention
- **Inference Speed**: Bidirectional attention is slower than causal for generation

## References

- [LLaDA Paper](https://arxiv.org/abs/2404.10406)
- [Axolotl Documentation](https://docs.axolotl.ai/)
