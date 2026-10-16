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
cfg = AutoConfig.from_pretrained(
    src,
    num_hidden_layers=2,
    # one entry per layer: a sliding layer, then the full-attention layer
    layer_types=["sliding_attention", "full_attention"],
    mlp_layer_types=["dense", "sparse"],
    hidden_size=256,
    intermediate_size=512,
    moe_intermediate_size=256,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    num_experts=8,
    num_experts_per_tok=2,
    n_group=1,
    topk_group=1,
)

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
- Run a full finetune by removing `adapter: qlora`, `load_in_4bit: true`, and
  `quantize_moe_experts: true`.
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
