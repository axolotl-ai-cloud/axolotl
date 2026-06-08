# Kernels Integration

MoE (Mixture of Experts) kernels speed up training for MoE layers and reduce VRAM costs. Transformers v5 introduced a uniform dispatch point for the per-expert grouped GEMMs via the `experts_implementation` config kwarg:

```python
class ExpertsInterface(GeneralInterface):
    _global_mapping = {
        "batched_mm": batched_mm_experts_forward,
        "grouped_mm": grouped_mm_experts_forward,
        "sonicmoe":   sonicmoe_experts_forward,   # upstream HF integration
    }
```

Axolotl registers two additional implementations into this same global registry: **ScatterMoE** (Triton, runs on any CUDA GPU) and a LoRA-aware **SonicMoE** variant (CUTLASS / cute-DSL, Hopper or newer). Routing — softmax/sigmoid top-k, group selection, shared experts, bias correction, etc. — stays in each model's `SparseMoEBlock`, where transformers handles all per-architecture variation. Axolotl only swaps the experts forward.

## Usage

Add the following to your axolotl YAML config:

```yaml
plugins:
  - axolotl.integrations.kernels.KernelsPlugin

use_kernels: true

# Choose one (mutually exclusive):
use_scattermoe: true
# OR
use_sonicmoe: true
```

`experts_implementation` is auto-set to `scattermoe` / `sonicmoe` from the kernel flag, but you can override to `eager` / `batched_mm` / `grouped_mm` to compare against the transformers reference implementations.

### SonicMoE installation

**Prerequisites:**
- NVIDIA Hopper (H100/H200) or Blackwell (B200/GB200/B300) GPU
- CUDA 12.9+ (13.0+ for B300)
- PyTorch 2.7+
- For B300: Triton 3.6.x

The sonic-moe kernel ships through the HF [`kernels`](https://github.com/huggingface/kernels) package. Transformers v5.8+ auto-fetches a prebuilt kernel from [`kernels-community/sonic-moe`](https://huggingface.co/kernels-community/sonic-moe) on first use:

```bash
pip install kernels "nvidia-cutlass-dsl==4.4.2"
```

**Note:** Blackwell support is in upstream beta. On Blackwell GPUs Axolotl automatically sets `USE_QUACK_GEMM=1` to enable the Blackwell kernels.

## How It Works

The `KernelsPlugin` runs once before model loading and:

1. Calls `register_scattermoe_experts()` or `register_sonicmoe_experts()`, which inserts the kernel forward into `transformers.integrations.moe.ALL_EXPERTS_FUNCTIONS`.
2. Sets `cfg.experts_implementation` to the matching name.
3. When the model loads, transformers' `@use_experts_implementation` decorator on each model's `Experts` class reads `config._experts_implementation` and dispatches to our registered forward.

That's the entire integration — there is no per-architecture SparseMoEBlock monkey-patch, no per-model routing code, and no weight-layout conversion. As new MoE models adopt the decorator upstream they immediately benefit from both kernels.

## LoRA Support

Both kernels train PEFT adapters on `gate_up_proj` / `down_proj` (and `gate` for the router) end-to-end:

- **ScatterMoE** fuses the LoRA `B @ A` product into the per-expert grouped GEMM via custom Triton kernels (`parallel_linear_lora`). No extra materialization pass.
- **SonicMoE** materializes `W_eff = W + scaling * (B @ A)` per expert inside a custom `MoELoRAMaterialize` `autograd.Function` and passes the effective weight into the CUTLASS kernel. Backward decomposes `dW_eff` into `dA` and `dB` via the chain rule, so LoRA parameters train without modifying the kernel.

Both paths detect PEFT `ParamWrapper` on individual expert parameters (`target_parameters` API) and unwrap them before dispatch.

## Model Support

Any model whose `Experts` class is decorated with `@use_experts_implementation` upstream works automatically. As of transformers 5.8 this includes (verified):

| Model Type        | ScatterMoE | SonicMoE |
|-------------------|:---------:|:--------:|
| `mixtral`         |    Yes    |   Yes    |
| `qwen2_moe`       |    Yes    |   Yes    |
| `qwen3_moe`       |    Yes    |   Yes    |
| `qwen3_5_moe`     |    Yes    |   Yes    |
| `olmoe`           |    Yes    |   Yes    |
| `mistral4`        |    Yes    |   Yes    |
| `glm_moe_dsa`     |    Yes    |   Yes    |
| `deepseek_v3`     |    Yes    |   Yes    |
| `minimax_m2`      |    Yes    |   Yes    |
| `ernie4_5_moe`    |    Yes    |   Yes    |
| `hunyuan_v1_moe`  |    Yes    |   Yes    |
| `gemma4_text`     |    Yes    |   Yes    |
| `gpt_oss`         |    Yes    |   Yes    |

`gpt_oss` carries the decorator with `is_concatenated=False, is_transposed=True, has_bias=True` and uses a sigmoid-GLU activation with clamping. Both forwards read these flags off `self` and dispatch accordingly: the ScatterMoE forward handles the transposed/interleaved/biased layout and clamped sigmoid-GLU via its Triton path (no weight transpose, interleaved gate/up, per-expert bias folded into the grouped GEMM); the SonicMoE forward uses the upstream CUTLASS kernel.

### Blackwell (sm_120) note

The SonicMoE CUTLASS kernel (`kernels-community/sonic-moe`) does not currently run on consumer Blackwell (sm_120) — its bundled quack `GemmSm120` predates the `concat_layout` arg the dispatcher passes. On sm_120, `use_sonicmoe` with a standard-layout model transparently falls back to the ScatterMoE Triton path, which runs there. `gpt_oss` on sm_120 should use `use_scattermoe` directly (bf16 base; MXFP4 weights dequantize on the fly as with other MXFP4 models — fused MXFP4 for `gpt_oss` is not yet wired).

## Feature comparison

| Feature                          | ScatterMoE | SonicMoE |
|----------------------------------|:----------:|:--------:|
| Kernel backend                   | Triton     | CUTLASS / cute-DSL |
| GPU requirement                  | Any CUDA   | Hopper+ |
| LoRA path                        | Fused in Triton kernel | `MoELoRAMaterialize` + custom autograd |
| LoRA overhead                    | Lower (fused) | Higher (materialization pass) |
| Selective expert dequantization  | Yes (~97% memory savings) | No |
| Weight format                    | Standard `[E, 2*I, H]` | Standard `[E, 2*I, H]` (concat layout, no interleave) |

## Note on MegaBlocks

We tested [MegaBlocks](https://huggingface.co/kernels-community/megablocks) but were unable to ensure numerical accuracy, so we did not integrate it. It was also incompatible with many newer model architectures in transformers.
