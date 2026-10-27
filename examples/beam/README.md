# Finetune Beam with Axolotl

[Beam](https://huggingface.co/reflection/Beam-1.0-501B-A23B) is a 501B-parameter mixture-of-experts model (23B active parameters per token) from Reflection, built for software engineering and long-horizon agentic tasks.

This guide shows how to fine-tune it with Axolotl.

Thanks to the team at Reflection for giving us early access to prepare for this release.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Pick a config and run (sized for one 8xH200 node):

    ```bash
    axolotl train examples/beam/beam-qlora-fsdp-staged.yaml
    ```

Available configs:

| Config | Approach |
|---|---|
| `beam-qlora-fsdp-staged.yaml` | QLoRA, FSDP2 + CPU-staged NF4 loader, attention + expert LoRA (data-parallel) |
| `beam-qlora-mp.yaml` | QLoRA, `device_map` model parallel, attention-only |
| `beam-qlora-experts-mp.yaml` | QLoRA, `device_map` model parallel, + expert LoRA |
| `beam-tiny-smoke.yaml` | Tiny random-weight model for fast iteration |

## Smoke test on a tiny model

`beam-tiny-smoke.yaml` points at a local beam model for testing and verification purposes.

```bash
uv run examples/beam/build_tiny_model.py
axolotl train examples/beam/beam-tiny-smoke.yaml
```

## Tips

- **`device_map` configs** run as one process (`--num-processes 1`) and need `gpu_memory_limit` set to spread layers across GPUs. Use `adamw_torch`: `adamw_torch_4bit` fails once adapters span GPUs.
- Keep the router (`mlp.gate`) out of `lora_target_parameters`; adapting it destabilizes routing.
- Run a full finetune by removing `adapter: qlora`, `load_in_4bit: true`, and
  `quantize_moe_experts: true`.
- **Tool-calling data:** every `tool` message needs a `name` (the called function); `tool_call_id` alone fails to render.
- Read more on loading your own dataset in the [docs](https://docs.axolotl.ai/docs/dataset_loading.html).

## Expert-parallel LoRA on B200 (bf16, NVFP4 and NF4)

### Prerequisites

The B200 configs assume the Blackwell kernel stack below; the `axolotlai/axolotl-cloud`
`-blackwell` image tags ship these kernels (see the [Docker docs](https://docs.axolotl.ai/docs/docker.html)),
or install them into an existing CUDA 13 environment:

| Config key | Package | Install |
|---|---|---|
| `expert_parallel_backend: deep_ep` | [DeepEP](https://github.com/deepseek-ai/DeepEP) v1.2.1 built for sm_100 | build per the [expert-parallel integration README](https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/integrations/expert_parallel/README.md#installation) (Blackwell section). Without it, set `expert_parallel_backend: torch`; the all-to-all dispatch needs nothing extra and ran about 25% slower on our smoke test |
| `use_sonicmoe: true` / `nvfp4_merge_aware: true` | [quack-kernels](https://pypi.org/project/quack-kernels/) ≥ 0.6.5 on nvidia-cutlass-dsl ≥ 4.7 | `uv pip install "quack-kernels[cu13]"` |
| `attn_implementation: flash_attention_4` | [flash-attn-4](https://pypi.org/project/flash-attn-4/) (pre-release; pulls quack-kernels and the cutlass DSL) | `uv pip install --pre "flash-attn-4[cu13]"`. Or keep `flash_attention_2`, which resolves to the `kernels-community/flash-attn2` hub kernel (has an sm_100 build) and is ~2x slower at Beam's shape |

The kernels plugin checks the installed cutlass DSL against quack's declared range at startup and
fails fast with the fix if they disagree.

### Configs

| File | Layout | Notes |
|---|---|---|
| `beam-lora-ep-b200.yaml` | 2 nodes x 8 B200: EP=8 inside each node, dense weights FSDP-sharded over the same ranks, HSDP replication across nodes | bf16 base, LoRA on attention + fused experts, DeepEP + ScatterMoE, flash attention 4, cut cross entropy |
| `beam-nvfp4-lora-b200.yaml` | 1 node x 8 B200, EP=8 | ModelOpt NVFP4 checkpoint, SonicMoE experts with merge-aware LoRA (`nvfp4_merge_aware`), so `merge-lora` reproduces the trained model |
| `beam-nf4-qlora-b200.yaml` | 1 node x 8 B200, EP=8 | bf16 checkpoint quantized to bitsandbytes NF4 at load by the CPU-staged loader (attention, dense MLP and the fused experts); ScatterMoE chunked-dequant experts |
| `autotune/b200.json` | measured Triton autotune winners for both paths | `autotune_cache_path` seeds the kernels from a copy and keeps recording |
| `autotune/h100.json` | H100 winners for the bf16 ScatterMoE forward at Beam's width (EP=2 and EP=4 on a random real-width model) | same tile as B200: `BLOCK_M 128, BLOCK_N 64, BLOCK_K 64`, 3 or 4 stages within 5% of each other |

The bf16 config pins the ScatterMoE LoRA tiles with `autotune_profiles` (measured on B200 and H100,
identical across every routed-row bucket) and enables `autotune_neighborhood: 1`, so no training or eval
step benchmarks kernel configs while expert-parallel peers wait. It also carries the B200 quack profiles for
`use_sonicmoe: true` (EP=8, `sonicmoe_ep_compact_routing` with `highwater` row padding): pins for the
forward and backward expert GEMMs at training sizes, measured per routed-pair bucket from 98k to 1M rows
(the pinned configs were within about 2% of each other wherever both ran; every other tile was at least 3%
slower and non-persistent scheduling up to 22% slower on the weight gradients), and narrowed searches
for everything else. The pins match Beam's expert shapes at EP=8 (32 local experts); another EP size or
GPU falls through to the narrowed or full search.

## Related Resources

- [Beam 1.0 on Hugging Face](https://huggingface.co/collections/reflection/beam-10)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
