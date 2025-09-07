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

# Diffusion-specific configuration (prefixed)
diffusion_noise_schedule: linear  # or "cosine"
diffusion_min_mask_ratio: 0.1
diffusion_max_mask_ratio: 0.9
diffusion_num_diffusion_steps: 128
diffusion_eps: 1e-3
diffusion_importance_weighting: true
# For non-Llama tokenizers, set this to a valid id (e.g., pad/eos)
diffusion_mask_token_id: 128002

# Sample generation during training (optional)
diffusion_generate_samples: true
diffusion_generation_interval: 100
diffusion_num_generation_samples: 3
diffusion_generation_steps: 128
diffusion_generation_temperature: 0.0
diffusion_generation_max_length: 100

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

Any models that support 4D attention masks should work out of the box. If not, please
create an [issue](https://github.com/axolotl-ai-cloud/axolotl/issues) or open a
[PR](https://github.com/axolotl-ai-cloud/axolotl/compare)!

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

```python
loss = sum(cross_entropy(pred, target) / p_mask) / total_tokens
```

## Sample Generation

When `generate_samples: true`, the plugin generates samples during training:

```
Sample 1:
   Original (45 tokens): The quick brown fox jumps over the lazy dog...
   Masked (18/45 tokens, 40.0%): The [MASK] [MASK] fox [MASK] over [MASK] lazy [MASK]...
   Generated: The quick brown fox jumps over the lazy dog...
```

Samples are logged to console and wandb (if enabled).

## Inference

Diffusion inference is integrated into the standard Axolotl CLI. Use the same config
you trained with and run:

```
axolotl inference --config path/to/your-config.yaml
```

Interactive controls (prefix the prompt with commands):
- `:complete N` → completion mode with N new masked tokens appended (default 64)
- `:mask R` → random masking mode with target mask ratio R in [0.0, 1.0]

Example session:

```
================================================================================
Commands:
:complete N -> completion mode with N tokens (default 64)
:mask R     -> random masking with ratio R (0.0–1.0)
================================================================================
Give me an instruction (Ctrl + D to submit):

:mask 0.4 The quick brown fox jumps over the lazy dog

Masked (40.0%):
The [MASK] brown [MASK] jumps over the [MASK] dog

Generated:
The quick brown fox jumps over the lazy dog
```

Notes:
- The CLI renders per‑token correctness in color (green/red/dim) against the original
  sequence for masked positions.
- Inference reuses `diffusion_mask_token_id` from your config. It will not add new
  tokens or resize embeddings during inference. If the id is missing or invalid, it
  falls back to the tokenizer's `unk_token_id`.

## Metrics and Monitoring

The plugin adds (or modifies) several metrics to track diffusion training:

- `train/loss`: Weighted diffusion loss
- `train/accuracy`: Accuracy on masked tokens
- `train/mask_ratio`: Average fraction of tokens masked
- `train/num_masked_tokens`: Number of tokens masked
- `train/avg_p_mask`: Average masking probability
- `train/ce_loss`: Unweighted cross-entropy loss
- `train/importance_weight_avg`: Average importance weight

## Limitations

- No flash attention support
- No RL training support

## References

- [LLaDA Paper](https://arxiv.org/abs/2404.10406)
- [Axolotl Documentation](https://docs.axolotl.ai/)
