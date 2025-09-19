# Diffusion LM Training Plugin for Axolotl

This plugin enables diffusion language model training using an approach inspired by
LLaDA (Large Language Diffusion Models) within Axolotl.

## Overview

LLaDA is a diffusion-based approach to language model training that uses:
- **Random token masking** during training instead of next-token prediction
- **Bidirectional attention** to allow the model to attend to the full context
- **Importance weighting** based on masking probabilities for stable training

This approach can lead to more robust language models with better understanding of
bidirectional context.

## Installation

The plugin is included with Axolotl. See our
[installation docs](https://docs.axolotl.ai/docs/installation.html).

## Quickstart

Train with an example config (Llama‑3.2 1B):
   - Pretrain: `axolotl train examples/llama-3/diffusion-3.2-1b-pretrain.yaml`
   - SFT: `axolotl train examples/llama-3/diffusion-3.2-1b-sft.yaml`

### Basic Configuration

You can also modify your existing configs to enable / customize diffusion training.

Add the following to your Axolotl config:

```yaml
# Enable diffusion LM training plugin
plugins:
  - axolotl.integrations.diffusion.DiffusionPlugin
```

And, configure the nested `diffusion` block (defaults shown):

```yaml
diffusion:
  noise_schedule: linear  # or "cosine"
  min_mask_ratio: 0.1
  max_mask_ratio: 0.9
  num_diffusion_steps: 128
  eps: 1e-3
  importance_weighting: true

  # Mask token (training auto-adds if missing, avoid pad/eos)
  mask_token_str: "<|diffusion_mask|>"
  # Or use an existing special token id (e.g., 128002 for Llama-3.x)
  # mask_token_id: 128002

  # Sample generation during training (optional)
  generate_samples: true
  generation_interval: 100
  num_generation_samples: 3
  generation_steps: 128
  generation_temperature: 0.0
  generation_max_length: 100
```

## Supported Models

Any models that support 4D attention masks should work out of the box. If not, please
create an [issue](https://github.com/axolotl-ai-cloud/axolotl/issues) or open a
[PR](https://github.com/axolotl-ai-cloud/axolotl/compare)!

## How It Works

### Random Masking
During training, tokens are randomly masked:
- Sample timestep `t` uniformly from [0, 1]
- Calculate masking probability: `p = (1 - eps) * t + eps`
- Randomly mask tokens with probability `p`

### Diffusion Loss

Loss is computed only on masked tokens with (optional) importance weighting:

```python
loss = sum(cross_entropy(pred, target) / p_mask) / total_tokens
```

## Sample Generation

When `diffusion.generate_samples: true`, the plugin generates samples during training:

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
axolotl inference path/to/your-config.yaml
```

Optionally, pass `--gradio` to use a simple web interface.

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
The quick brown fox jumps over the loud dog
```

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
- [API reference for plugin](https://docs.axolotl.ai/docs/api/integrations.diffusion.args.html#axolotl.integrations.diffusion.args)
