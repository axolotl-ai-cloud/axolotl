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
  document). Verify loss parity before relying on it; on `sdpa`, packing is wrong.
- Run a full finetune by removing `adapter: qlora`, `load_in_4bit: true`, and
  `quantize_moe_experts: true`.
- Read more on loading your own dataset in the [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
