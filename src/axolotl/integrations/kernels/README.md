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
use_sonicmoe: true
```

**Important:** Setting `experts_implementation` to `batched_mm` or `grouped_mm` is incompatible with custom kernel options. The exception is `experts_implementation: scattermoe`, which is used for models like Gemma 4 that embed MoE directly in the decoder layer (no SparseMoeBlock) and dispatch through the transformers `ExpertsInterface`.

### SonicMoE installation

**Prerequisites:**
- NVIDIA Hopper (H100, H200) or Blackwell (B200, GB200) GPU
- CUDA 12.9+ (13.0+ for B300)
- PyTorch 2.7+ (2.9.1 recommended)
- For B300: Triton 3.6.0

```bash
pip install --ignore-requires-python --no-deps "sonic-moe @ git+https://github.com/Dao-AILab/sonic-moe.git@116e2df0a41874f77fa0ad269ce7df3f0cfcb956" && pip install nvidia-cutlass-dsl==4.4.0 quack-kernels==0.2.5
```

See the [SonicMoE installation guide](https://github.com/Dao-AILab/sonic-moe?tab=readme-ov-file#-installation) for the latest prerequisite details.

**Note:** Blackwell support is in upstream beta. On Blackwell GPUs, Axolotl automatically sets `USE_QUACK_GEMM=1` to enable the Blackwell kernels.

## How It Works

The `KernelsPlugin` runs before model loading and:

### ScatterMoE
1. Registers the ScatterMoE kernel from the local `libs/scattermoe_lora` package (includes fused LoRA support via Triton kernels).
2. Patches the model's `SparseMoeBlock` forward method with the optimized ScatterMoE implementation via the HF `kernels` library.

### SonicMoE
1. Resolves the model's MoE block class(es) from `constants.py`.
2. Patches the forward method with SonicMoE's optimized CUTLASS kernels and registers a weight converter for the interleaved gate/up projection format.
3. Supports pluggable routing strategies (see routing table below).

Both paths use the shared `resolve_moe_block_classes` utility in `constants.py` for model-type-to-class resolution.

## Model Support Matrix

Most models use the **SwiGLU** activation (`silu(gate) * up`). Gemma 4 uses **GEGLU** (`gelu(gate) * up`). ScatterMoE supports any gated activation (activation is applied in Python between kernel calls). SonicMoE supports SwiGLU, GEGLU, and REGLU via its `ActivationType` enum.

### Routing strategies

| Routing Strategy | Description | ScatterMoE | SonicMoE |
|---|---|:---:|:---:|
| softmax → topk | Softmax over experts, select top-K, optional renormalization | Yes | Yes |
| softmax → group selection → topk | Softmax, select top groups (sum of top-2 per group), topk from selected groups, renorm + scaling | No | Yes |
| sigmoid → topk (with groups) | Sigmoid + bias correction, group-based masking, topk from masked scores, weights from original sigmoid | Yes | Yes |
| sigmoid → topk (no groups) | Sigmoid + bias correction, straight topk (n_group=1) | Yes | Yes |
| softmax → bias correction → topk | Softmax, bias via `gate.moe_statics`, topk, gather from original probs, clamp-based renorm | No | Yes |
| softmax → group_limited_greedy | Softmax, group selection (max per group), topk, scale only (no renorm) | No | Yes |
| softmax → topk via gate.wg | Softmax, gate weight at `gate.wg.weight` (not `gate.weight`), always renormalize | No | Yes |
| softmax → topk + per_expert_scale | RMSNorm → scale → proj → softmax → topk → renorm → per-expert learned scales | Yes | Yes |
| fused topk → softmax | Routing + expert computation fused in a single kernel | No | Planned |

### Per-model support

| Model Type | Architecture | Routing | ScatterMoE | SonicMoE |
|---|---|---|:---:|:---:|
| `qwen2_moe` | Qwen2-MoE | softmax → topk | **Yes** | **Yes** |
| `qwen3_moe` | Qwen3-MoE | softmax → topk | **Yes** | **Yes** |
| `qwen3_5_moe` | Qwen3.5-MoE | softmax → topk | **Yes** | **Yes** |
| `qwen3_5_moe_text` | Qwen3.5-MoE (VLM text) | softmax → topk | **Yes** | **Yes** |
| `qwen3_next` | Qwen3-Next | softmax → topk | **Yes** | **Yes** |
| `qwen3_vl_moe` | Qwen3-VL-MoE | softmax → topk | **Yes** | **Yes** |
| `qwen3_omni_moe` | Qwen3-Omni (Thinker + Talker) | softmax → topk | **Yes** | **Yes** |
| `olmoe` | OLMoE | softmax → topk | **Yes** | **Yes** |
| `mixtral` | Mixtral | softmax → topk | **Yes** | **Yes** |
| `minimax` | MiniMax | softmax → topk | **Yes** | **Yes** |
| `mistral4` | Mistral 4 | softmax → group → topk | No | **Yes** |
| `glm_moe_dsa` | GLM-MoE DSA (GLM 5) | sigmoid → topk (groups) | **Yes** | **Yes** |
| `deepseek_v3` | DeepSeek-V3 | sigmoid → topk (groups) | **Yes** | **Yes** |
| `glm4_moe` | GLM4-MoE | sigmoid → topk (groups) | **Yes** | **Yes** |
| `glm4_moe_lite` | GLM4-MoE Lite (GLM 4.7 Flash) | sigmoid → topk (groups) | **Yes**\* | **Yes** |
| `glm4v_moe` | GLM4v-MoE | sigmoid → topk (groups) | **Yes** | **Yes** |
| `minimax_m2` | MiniMax M2 | sigmoid → topk (no groups) | **Yes** | **Yes** |
| `ernie4_5_moe` | ERNIE 4.5 MoE | softmax → bias → topk | No | **Yes** |
| `deepseek_v2` | DeepSeek-V2 | softmax → group_limited_greedy | No | **Yes** |
| `hunyuan_v1_moe` | HunYuan V1 MoE | softmax → topk (gate.wg) | No | **Yes** |
| `gemma4_text` | Gemma 4 (26B-A4B) | softmax → topk + per_expert_scale | **Yes**\*\* | **Yes**\*\* |
| `gpt_oss` | GPT-OSS | fused topk → softmax | No | Planned |

\* `glm4_moe_lite` with ScatterMoE may have issues — see Limitations.

\*\* Gemma 4 uses `experts_implementation: scattermoe` path (registered via `ExpertsInterface`) instead of SparseMoeBlock patching, since Gemma 4 embeds MoE directly in its decoder layer (no separate SparseMoeBlock). See the [Gemma 4 section](#gemma-4) below.

### Feature comparison

| Feature | ScatterMoE | SonicMoE |
|---|:---:|:---:|
| Kernel backend | Triton | CUTLASS |
| GPU requirement | Any CUDA | Hopper (H100/H200) or Blackwell (B200+) |
| LoRA approach | Fused in Triton kernel | Runtime materialization + custom autograd |
| LoRA overhead | Lower (fused computation) | Higher (per-forward materialization) |
| Gate/router LoRA | Yes | Yes |
| Expert LoRA | Yes (fused) | Yes (materialized) |
| Shared expert LoRA | Yes (standard PEFT) | Yes (standard PEFT) |
| Selective expert dequantization | Yes (~97% memory savings) | No |
| Weight format | Transposed `[E, hidden, 2*inter]` | Interleaved gate/up `[2*I, H, E]` |
| torch.compile routing | No | Yes (optional) |

## Shared Expert Handling

Both kernels handle shared experts identically. Shared expert attribute names are detected in order of priority:

1. `shared_expert` (Qwen2-MoE)
2. `shared_experts` (GLM-MoE, DeepSeek-V3)
3. `shared_mlp` (HunYuan V1 MoE)

If `shared_expert_gate` exists, sigmoid gating is applied to the shared expert contribution before adding it to the routed output. PEFT wraps shared expert linear layers with standard LoRA — no special handling is needed.

## Gemma 4

Gemma 4 (e.g. `google/gemma-4-26B-A4B`) has a unique hybrid MoE architecture:

- **No SparseMoeBlock**: MoE is embedded directly in the decoder layer alongside a dense MLP. Both run in parallel and their outputs are summed.
- **Custom router** (`Gemma4TextRouter`): RMSNorm → learned scale → linear projection → softmax → top-k → renormalization → per-expert learned scales.
- **GEGLU activation**: Uses `gelu_pytorch_tanh` (not SiLU/SwiGLU like most other MoE models).
- **128 experts, top-k=8** for the 26B-A4B variant.

Because there is no SparseMoeBlock class to patch, Gemma 4 uses a different integration path: we register `"scattermoe"` as a custom implementation in the transformers `ExpertsInterface`, and set `experts_implementation: scattermoe` in the config. The `@use_experts_implementation` decorator on `Gemma4TextExperts` then dispatches to our ScatterMoE kernel automatically. The router is untouched — it runs as-is.

## Limitations

- **ScatterMoE + GLM4-MoE Lite**: ScatterMoE does not work reliably for GLM 4.7 Flash (`glm4_moe_lite`).
- **Non-SwiGLU activations**: Neither kernel supports MoE architectures with non-SwiGLU expert activations (e.g., GPT-OSS uses a custom GLU variant).
- **GPT-OSS**: Deferred — requires transposed weight layout `[E, H, 2*I]`, expert biases, and custom GLU activation. A dedicated forward path is needed.
- **FSDP + fused gate LoRA (SonicMoE)**: The fused topk→softmax path materializes a local tensor when LoRA delta is present to avoid DTensor + Tensor mixing under FSDP.

## Note on MegaBlocks

We tested [MegaBlocks](https://huggingface.co/kernels-community/megablocks) but were unable to ensure numerical accuracy, so we did not integrate it. It was also incompatible with many newer model architectures in transformers.
