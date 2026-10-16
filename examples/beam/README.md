# Finetune Beam with Axolotl

[Beam](https://huggingface.co/reflection/Beam-1.0-501B-A23B) is a 501B-parameter mixture-of-experts model (23B active parameters per token) from Reflection, built for software engineering and long-horizon agentic tasks.

This guide shows how to fine-tune it with Axolotl.

Thanks to the team at Reflection for giving us early access to prepare for this release.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Upgrade Transformers to v5.20.1 or later, which adds Beam:

    ```bash
    uv pip install "transformers>=5.20.1"
    ```

3. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

4. Pick a config and run (sized for one 8xH200 node):

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

| File | Layout | Notes |
|---|---|---|
| `beam-lora-ep-b200.yaml` | 2 nodes x 8 B200: EP=8 inside each node, dense weights FSDP-sharded over the same ranks, HSDP replication across nodes | bf16 base, LoRA on attention + fused experts, DeepEP + ScatterMoE, flash attention 4, cut cross entropy |
| `beam-nvfp4-lora-b200.yaml` | 1 node x 8 B200, EP=8 | ModelOpt NVFP4 checkpoint, SonicMoE experts with merge-aware LoRA (`nvfp4_merge_aware`), so `merge-lora` reproduces the trained model |
| `beam-nf4-qlora-b200.yaml` | 1 node x 8 B200, EP=8 | bf16 checkpoint quantized to bitsandbytes NF4 at load by the CPU-staged loader (attention, dense MLP and the fused experts); ScatterMoE chunked-dequant experts |
| `autotune/b200.json` | measured Triton autotune winners for both paths | `autotune_cache_path` seeds the kernels from a copy and keeps recording |

The bf16 and NF4 configs pin the ScatterMoE LoRA tiles with `autotune_profiles` (measured on B200, identical
across every routed-row bucket) and enable `autotune_neighborhood: 1`, so no training or eval
step benchmarks kernel configs while expert-parallel peers wait.

## Related Resources

- [Reflection HF](https://huggingface.co/reflection)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
