# Kernels Integration

MoE (Mixture of Experts) kernels speed up training for MoE layers and reduce VRAM costs. In transformers v5, `batched_mm` and `grouped_mm` were integrated as built-in options via the `experts_implementation` config kwarg:

```python
class ExpertsInterface(GeneralInterface):
    _global_mapping = {
        "batched_mm": batched_mm_experts_forward,
        "grouped_mm": grouped_mm_experts_forward,
    }
```

In our custom integration, we add support for **ScatterMoE** and **SonicMoE**, which are more efficient and faster than `grouped_mm`.

## Usage

Add the following to your axolotl YAML config:

```yaml
plugins:
  - axolotl.integrations.kernels.KernelsPlugin

use_kernels: true

# Choose one (mutually exclusive):
use_scattermoe: true
# OR
# use_sonicmoe: true
```

### SonicMoE installation

```bash
pip install git+https://github.com/Dao-AILab/sonic-moe@3e3f36eeefc324b10042db22921bfe9ab53ed2d1
```

**Important:** Setting `experts_implementation` is incompatible with custom kernel options.

## How It Works

The `KernelsPlugin` runs before model loading and:

### ScatterMoE
1. Registers the ScatterMoE kernel from the local `libs/scattermoe_lora` package (includes fused LoRA support via Triton kernels).
2. Patches the model's `SparseMoeBlock` forward method with the optimized ScatterMoE implementation.

### SonicMoE
1. Resolves the model's MoE block class(es) from `constants.py`.
2. Patches the forward method with SonicMoE's optimized kernels and registers a weight converter for the interleaved gate/up projection format.
3. Supports both softmax->topk and sigmoid->topk routing strategies.

Both paths use the shared `resolve_moe_block_classes` utility in `constants.py` for model-type-to-class resolution.

## Supported Models

See `constants.py` for the full list of supported model types (Qwen2-MoE, Qwen3-MoE, OLMoE, Mixtral, DeepSeek-V3, GLM-MoE, MiniMax, etc.).

## Limitations

ScatterMoE uses a softmax -> topk routing, so results may be different for some model architectures as baseline (GPT-OSS, GLM_MOE_DSA).

SonicMoE supports both softmax->topk and sigmoid->topk routing, covering a wider range of architectures.

## Note on MegaBlocks

We tested [MegaBlocks](https://huggingface.co/kernels-community/megablocks) but were unable to ensure numerical accuracy, so we did not integrate it. It was also incompatible with many newer model architectures in transformers.
