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
# For non-Llama tokenizers, set this to a valid id (e.g., pad/eos)
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

After training, you can run lightweight reverse-diffusion on your model using the
helper script `scripts/diffusion_infer.py`.

Examples:

- Provide your own texts:

```
python scripts/diffusion_infer.py \
  --model /path/to/checkpoint \
  --texts "The quick brown fox" "In a hole in the ground" \
  --steps 32 --max-length 64 --num-samples 2
```

- Sample from a dataset:

```
python scripts/diffusion_infer.py \
  --model /path/to/checkpoint \
  --dataset mhenrichsen/alpaca_2k_test \
  --dataset-field text --steps 32 --num-samples 2
```

Key options:
- `--steps`: diffusion steps (lower for faster, e.g., 32)
- `--num-samples`: number of samples to generate
- `--mask-token-id`: token used for masking. Defaults to a Llama-3.2 id (128002). For
  other models, prefer an existing reserved special token (e.g., a token containing
  "reserved"/"mask"), or `unk_token_id`. Avoid using `pad`/`eos` if possible.

Note: During training, if `mask_token_id` is unset or out-of-range for the tokenizer's
vocabulary, Axolotl tries to auto-select a suitable id in this order:
reserved special → other additional special → unk → pad → eos → vocab_size-1 → 0.

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
- [Axolotl Documentation](https://docs.axolotl.ai/)
