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
uv pip install kernels "nvidia-cutlass-dsl==4.6.0" "apache-tvm-ffi>=0.1.10,<0.2"
```

`apache-tvm-ffi` is an undeclared runtime dependency of `nvidia-cutlass-dsl` 4.6.0 (absent from its `Requires-Dist`, so pip will not pull it); `<0.1.10` breaks `cute.compile`, so pin it explicitly.

**Note:** Blackwell support is in upstream beta. On Blackwell GPUs Axolotl automatically sets `USE_QUACK_GEMM=1` to enable the Blackwell kernels.


## How It Works

The `KernelsPlugin` runs once before model loading and:

1. Calls `register_scattermoe_experts()` or `register_sonicmoe_experts()`, which inserts the kernel forward into `transformers.integrations.moe.ALL_EXPERTS_FUNCTIONS`.
2. Sets `cfg.experts_implementation` to the matching name.
3. When the model loads, transformers' `@use_experts_implementation` decorator on each model's `Experts` class reads `config._experts_implementation` and dispatches to our registered forward.

That's the entire integration — there is no per-architecture SparseMoEBlock monkey-patch, no per-model routing code, and no weight-layout conversion. As new MoE models adopt the decorator upstream they immediately benefit from both kernels.

## BF16 LoRA Support

Both kernels train PEFT adapters on `gate_up_proj` / `down_proj` (and `gate` for the router) end-to-end:

- **ScatterMoE** fuses the LoRA `B @ A` product into the per-expert grouped GEMM via custom Triton kernels (`parallel_linear_lora`). No extra materialization pass.
- **SonicMoE** materializes `W_eff = W + scaling * (B @ A)` per expert inside a custom `MoELoRAMaterialize` `autograd.Function` and passes the effective weight into the CUTLASS kernel. Backward decomposes `dW_eff` into `dA` and `dB` via the chain rule, so LoRA parameters train without modifying the kernel.

Both paths detect PEFT `ParamWrapper` on individual expert parameters (`target_parameters` API) and unwrap them before dispatch.

### ScatterMoE NVFP4 (W4A16) LoRA

Train LoRA on ModelOpt NVFP4 checkpoint via ScatterMoE. Routed experts are dequantized to bf16 (W4A16).

Requires:
- CUDA GPU with Triton.
- `qwen3_moe`, `qwen3_next`, `deepseek_v4`, `glm_moe_dsa`, and `gemma4_text`

**Tip:** in our tests, the Triton `dequant` path below is currently faster end to end than the ScatterMoE NVFP4 (if the arch is supported).

### SonicMoE NVFP4 (W4A4) LoRA

Train LoRA on ModelOpt NVFP4 checkpoint via Quack (e.g. `nvidia/Qwen3-30B-A3B-NVFP4`).

Requires:
- Blackwell SM100 for W4A4, others for W4A16
- `qwen3_moe` / `qwen3_next`

Install the pinned quack kernels (other versions untested):

```bash
uv pip install "quack-kernels==0.6.1" "nvidia-cutlass-dsl==4.6.0" "apache-tvm-ffi>=0.1.10,<0.2"
```

`AXOLOTL_SONICMOE_NVFP4_BACKEND` picks the expert GEMM (unset = auto):

| Backend | Compute | Runs on |
|---------|---------|---------|
| `fp4_cute` | native W4A4 tensor cores (quack) | Blackwell B200/GB200 (SM100/110) |
| `dequant` | dequant to bf16 (W4A16) | any CUDA GPU with Triton |

Advanced tuning knobs (fused up-proj, per-tensor-scale fold, fp8 DeepGEMM backward) are exposed as other `AXOLOTL_SONICMOE_NVFP4_*` env vars; defaults are correct for normal training.

**Tip:** in our tests, the Triton `dequant` path is currently faster (and lower mem) end to end than the `fp4_cute` path for < 100B param MoE model. However, `fp4_cute` should overtake when expert matmul become more dominant cost.

## Model Support

Any model whose `Experts` class is decorated with `@use_experts_implementation` upstream works automatically. As of transformers 5.8 this includes (verified). The `bf16` columns are base (unquantized) support; the `NVFP4` columns mark which kernel trains LoRA on a ModelOpt NVFP4 checkpoint of that arch:

| Model Type        | ScatterMoE (bf16) | SonicMoE (bf16) | NVFP4 (ScatterMoE) | NVFP4 (SonicMoE) |
|-------------------|:---------:|:--------:|:---:|:---:|
| `mixtral`         |    Yes    |   Yes    |  -  |  -  |
| `qwen2_moe`       |    Yes    |   Yes    |  -  |  -  |
| `qwen3_moe`       |    Yes    |   Yes    | Yes | Yes |
| `qwen3_next`      |    Yes    |   Yes    | Yes | Yes |
| `qwen3_5_moe`     |    Yes    |   Yes    |  -  |  -  |
| `olmoe`           |    Yes    |   Yes    |  -  |  -  |
| `mistral4`        |    Yes    |   Yes    |  -  |  -  |
| `glm_moe_dsa`     |    Yes    |   Yes    | Yes |  -  |
| `deepseek_v3`     |    Yes    |   Yes    |  -  |  -  |
| `minimax_m2`      |    Yes    |   Yes    |  -  |  -  |
| `ernie4_5_moe`    |    Yes    |   Yes    |  -  |  -  |
| `hunyuan_v1_moe`  |    Yes    |   Yes    |  -  |  -  |
| `gemma4_text`     |    Yes    |   Yes    | Yes |  -  |
| `gpt_oss`         |    Yes    |   Yes    |  -  |  -  |

NVFP4 for `deepseek_v4` is supported via ScatterMoE with `use_dsv4_kernels` (its own fused-kernel path), so it is not a row above.

`gpt_oss` carries the decorator with `is_concatenated=False, is_transposed=True, has_bias=True` and uses a sigmoid-GLU activation with clamping. Both forwards read these flags off `self` and dispatch accordingly: the ScatterMoE forward handles the transposed/interleaved/biased layout and clamped sigmoid-GLU via its Triton path (no weight transpose, interleaved gate/up, per-expert bias folded into the grouped GEMM); the SonicMoE forward uses the upstream CUTLASS kernel.

### Blackwell (sm_120) note

`use_sonicmoe` runs on consumer Blackwell (sm_120) when the loaded `sonic-moe` kernel bundles quack 0.6.1 (on nvidia-cutlass-dsl 4.6.0). The upstream `kernels-community/sonic-moe` prebuilt may still bundle quack 0.3.11 (no sm_120 GEMM) until the rebuild lands; point at a quack 0.6.1 build or use `use_scattermoe`. NVFP4 experts on sm_120 take the dequant path (no native W4A4: `fp4_cute` is SM100/SM110-only).

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
