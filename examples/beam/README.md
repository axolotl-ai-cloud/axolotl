# Finetune Beam with Axolotl

[Beam](https://huggingface.co/reflectionai/beam) is a ~501B-total / ~23B-active
Mixture-of-Experts model from Reflection AI (`BeamForCausalLM`, `model_type: beam`).
Experts are stored as fused 3D `nn.Parameter` tensors, which makes the LoRA
targeting a little different from a typical model (see below).

This guide shows how to fine-tune it with Axolotl.

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

`beam-tiny-smoke.yaml` points at a local `./beam-tiny` — a tiny, random-weight
model built from Beam's own config, so the full pipeline (loader, NF4 quant, LoRA
targeting) runs end to end in about a minute without the 501B download. Build it
once:

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

src = "reflectionai/beam"
cfg = AutoConfig.from_pretrained(src)
# Shrink every size knob the config exposes; unknown fields are skipped.
for field, value in {
    "num_hidden_layers": 2,
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 512,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "first_k_dense_replace": 1,
}.items():
    if hasattr(cfg, field):
        setattr(cfg, field, value)

AutoModelForCausalLM.from_config(cfg).save_pretrained("./beam-tiny")
AutoTokenizer.from_pretrained(src).save_pretrained("./beam-tiny")
```

Then `axolotl train examples/beam/beam-tiny-smoke.yaml`.

## LoRA targeting (experts)

Experts are ~98% of the weights, so `quantize_moe_experts: true` (NF4) is what
makes QLoRA fit. At runtime the experts are **fused 3D `nn.Parameter` tensors**,
not linear modules, so:

- Target them with **`lora_target_parameters`** (`mlp.experts.gate_up_proj`,
  `mlp.experts.down_proj`), not `lora_target_modules`.
- **`lora_dropout` must be `0`** — PEFT wraps 3D params in `ParamWrapper`, which
  rejects nonzero dropout.
- **Do not set `lora_target_linear: true`** — it is rejected alongside
  `quantize_moe_experts`, and `gate_proj` is ambiguous here (it names both the
  attention output gate and the dense MLP).
- Leave the router out of `target_parameters`; adapting it destabilizes routing.

Set `transformer_layer_cls_to_wrap: BeamDecoderLayer` to match the checkpoint.

## Tips

- **Optimizer:** use `adamw_torch`. `adamw_torch_4bit` raises once `device_map`
  spreads adapters across GPUs; for the expert config, `adamw_bnb_8bit` cuts
  optimizer-state VRAM if you are tight.
- **FSDP2 staged loader:** keep `state_dict_type: SHARDED_STATE_DICT`. With
  `quantize_moe_experts`, `FULL_STATE_DICT` crashes on the first checkpoint save;
  the final adapter is gathered and saved normally regardless.
- **`device_map` configs** have no data parallelism and must launch with
  `--num-processes 1`.
- **Sample packing** is supported, but the model's `ssmax` reads `position_ids`,
  so it needs the `flash_attention_2` varlen path (which resets positions per
  document). On `sdpa` the packed batch keeps running positions across the packed
  samples, so `ssmax` mis-scales and the packed loss diverges from the unpacked
  run — use `flash_attention_2` when packing.
- Run a full finetune by removing `adapter: qlora`, `load_in_4bit: true`, and
  `quantize_moe_experts: true`.
- Read more on loading your own dataset in the [docs](https://docs.axolotl.ai/docs/dataset_loading.html).

## Expert-parallel LoRA on B200 (bf16, NVFP4 and NF4)

### Prerequisites

The H200 configs above need only Axolotl and Cut Cross Entropy. The B200 configs assume the
Blackwell kernel stack below; the `axolotlai/axolotl-cloud` `-blackwell` image tags ship all of
it (see the [Docker docs](https://docs.axolotl.ai/docs/docker.html)), or install it into an
existing CUDA 13 environment:

| Config key | Package | Install |
|---|---|---|
| `expert_parallel_backend: deep_ep` | [DeepEP](https://github.com/deepseek-ai/DeepEP) v1.2.1 built for sm_100 | build per the [expert-parallel integration README](../../src/axolotl/integrations/expert_parallel/README.md#installation) (Blackwell section). Without it, set `expert_parallel_backend: torch`; the all-to-all dispatch needs nothing extra and ran about 25% slower on our smoke test |
| `use_sonicmoe: true` / `nvfp4_merge_aware: true` | [quack-kernels](https://pypi.org/project/quack-kernels/) ≥ 0.6.5 on nvidia-cutlass-dsl ≥ 4.7 | `uv pip install "quack-kernels[cu13]"` |
| `attn_implementation: flash_attention_4` | [flash-attn-4](https://pypi.org/project/flash-attn-4/) (pre-release; pulls quack-kernels and the cutlass DSL) | `uv pip install --pre "flash-attn-4[cu13]"`. Or keep `flash_attention_2`, which resolves to the `kernels-community/flash-attn2` hub kernel (has an sm_100 build) and is ~2x slower at Beam's shape |
| `base_model: reflectionai/beam*` | transformers ≥ 5.20.1 (`BeamForCausalLM`) | newer than Axolotl's pin while it catches up: `uv pip install "transformers>=5.20.1"` |

The kernels plugin checks the installed cutlass DSL against quack's declared range at startup and
fails fast with the fix if they disagree.

| File | Layout | Notes |
|---|---|---|
| `beam-lora-ep-b200.yaml` | 2 nodes x 8 B200: EP=8 inside each node, dense weights FSDP-sharded over the same ranks, HSDP replication across nodes | bf16 base, LoRA on attention + fused experts, DeepEP + ScatterMoE, flash attention 4, cut cross entropy |
| `beam-nvfp4-lora-b200.yaml` | 1 node x 8 B200, EP=8 | ModelOpt NVFP4 checkpoint, SonicMoE experts with merge-aware LoRA (`nvfp4_merge_aware`), so `merge-lora` reproduces the trained model |
| `beam-nf4-qlora-b200.yaml` | 1 node x 8 B200, EP=8 | bf16 checkpoint quantized to bitsandbytes NF4 at load by the CPU-staged loader (attention, dense MLP and the fused experts); ScatterMoE chunked-dequant experts |
| `autotune/b200.json` | measured Triton autotune winners for both paths | `autotune_cache_path` seeds the kernels from a copy and keeps recording |
| `autotune/h100.json` | H100 winners for the bf16 ScatterMoE forward at Beam's width (EP=2 and EP=4 on a random real-width model) | same tile as B200: `BLOCK_M 128, BLOCK_N 64, BLOCK_K 64`, 3 or 4 stages within 5% of each other |

The bf16 and NF4 configs pin the ScatterMoE LoRA tiles with `autotune_profiles` (measured on B200 and H100,
identical across every routed-row bucket) and enable `autotune_neighborhood: 1`, so no training or eval
step benchmarks kernel configs while expert-parallel peers wait.
