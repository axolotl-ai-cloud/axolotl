# Kernels Integration

MoE (Mixture of Experts) kernels speed up training for MoE layers and reduce VRAM costs. In transformers v5, `batched_mm` and `grouped_mm` were integrated as built-in options via the `experts_implementation` config kwarg:

```python
class ExpertsInterface(GeneralInterface):
    _global_mapping = {
        "batched_mm": batched_mm_experts_forward,
        "grouped_mm": grouped_mm_experts_forward,
    }
```

In our custom integration, we add support for **ScatterMoE**, which is even more efficient and faster than `grouped_mm`.

## Usage

Add the following to your axolotl YAML config:

```yaml
plugins:
  - axolotl.integrations.kernels.KernelsPlugin

use_kernels: true
use_scattermoe: true
```

**Important:** Do not set `experts_implementation` in your config as they are incompatible.

## How It Works

The `KernelsPlugin` runs before model loading and:

1. Registers the ScatterMoE kernel from the [`axolotl-ai-co/scattermoe`](https://huggingface.co/axolotl-ai-co/scattermoe) Hub repo.
2. Patches the model's `SparseMoeBlock` forward method with the optimized ScatterMoE implementation.

This works for any MoE model in transformers that uses a `SparseMoeBlock` class (Mixtral, Qwen2-MoE, OLMoE, etc.).

## Limitations

ScatterMoE uses a softmax -> topk routing, so results may be different for some model arch as baseline (GPT-OSS, GLM_MOE_DSA).

## Note on MegaBlocks

We tested [MegaBlocks](https://huggingface.co/kernels-community/megablocks) but were unable to ensure numerical accuracy, so we did not integrate it. It was also incompatible with many newer model architectures in transformers.
