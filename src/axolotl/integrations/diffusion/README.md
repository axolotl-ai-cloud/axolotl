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
    type: completion  # or conversation

sequence_len: 1024
micro_batch_size: 8
gradient_accumulation_steps: 4
learning_rate: 3e-4
```

## Supported Models

Any models that support 4D attention masks should work out of the box. If not, please
create an [issue](https://github.com/axolotl-ai-cloud/axolotl/issues)!

## How It Works

### Random Masking
During training, tokens are randomly masked based on a sampled timestep:
- Sample timestep `t` uniformly from [0, 1]
- Calculate masking probability: `p = (1 - eps) * t + eps`
- Randomly mask tokens with probability `p`

### Bidirectional Attention
The plugin uses native 4D attention masks to:
- Enable bidirectional attention without patches
- Allow all tokens to attend to all other tokens
- Maintain proper padding masks
- Work with modern `transformers` models out of the box

### Diffusion Loss

Loss is computed only on masked tokens with (optional) importance weighting:

```
loss = sum(cross_entropy(pred, target) / p_mask) / total_tokens
```

## Performance Tips

### Memory Optimization
- Bidirectional attention uses more memory than causal attention
- Consider reducing `micro_batch_size` if you encounter OOM errors
- Consider using gradient checkpointing, torch.compile,

### Training Stability
- Start with `noise_schedule: linear` for more predictable behavior
- Enable `importance_weighting: true` for better gradient scaling

### Convergence
- Monitor the `diffusion_loss` and `diffusion_accuracy` metrics
- Expect different loss curves compared to standard language modeling

## Sample Generation

When `generate_samples: true`, the plugin generates samples during training:

```
📝 Sample 1:
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

## Limitations

- No flash attention support

## References

- [LLaDA Paper](https://arxiv.org/abs/2404.10406)
- [Axolotl Documentation](https://github.com/OpenAccess-AI-Collective/axolotl)
