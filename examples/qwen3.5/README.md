# Finetune Qwen3.5 with Axolotl

[Qwen3.5](https://huggingface.co/collections/Qwen/qwen35) is a hybrid architecture model series combining Gated DeltaNet linear attention with standard Transformer attention. All Qwen3.5 models are early-fusion vision-language models: dense variants use `Qwen3_5ForConditionalGeneration` and MoE variants use `Qwen3_5MoeForConditionalGeneration`.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Install FLA for sample packing support with the Gated DeltaNet linear attention layers:
  ```bash
  pip3 uninstall -y causal-conv1d && pip3 install flash-linear-attention==0.4.1
  ```
  > FLA is required when `sample_packing: true`. Without it, training raises a `RuntimeError` on packed sequences. Vision configs use `sample_packing: false` so FLA is optional there.

4. Pick any config from the table below and run:

    ```bash
    axolotl train examples/qwen3.5/<config>.yaml
    ```

Available configs:

| Config | Model | Type | Peak VRAM |
|---|---|---|---|
| `9b-lora-vision.yaml` | Qwen3.5-9B | Vision+text LoRA, single GPU | — |
| `9b-fft-vision.yaml` | Qwen3.5-9B | Vision+text FFT, single GPU | ~61 GiB |
| `27b-qlora.yaml` | Qwen3.5-27B | Dense, text-only QLoRA | ~47 GiB |
| `27b-fft.yaml` | Qwen3.5-27B | Dense, text-only FFT (vision frozen) | ~53 GiB |
| `27b-qlora-fsdp.yaml` | Qwen3.5-27B | Dense, text-only QLoRA + FSDP2 | — |
| `35b-a3b-moe-qlora.yaml` | Qwen3.5-35B-A3B | MoE, text-only QLoRA | — |
| `35b-a3b-moe-qlora-fsdp.yaml` | Qwen3.5-35B-A3B | MoE, text-only QLoRA + FSDP2 | — |
| `122b-a10b-moe-qlora.yaml` | Qwen3.5-122B-A10B | MoE, text-only QLoRA | — |
| `122b-a10b-moe-qlora-fsdp.yaml` | Qwen3.5-122B-A10B | MoE, text-only QLoRA + FSDP2 | — |

### Gated DeltaNet Linear Attention

Qwen3.5 interleaves standard attention with Gated DeltaNet linear attention layers. To apply LoRA to them, add to `lora_target_modules`:

```yaml
lora_target_modules:
  # ... standard projections ...
  - linear_attn.in_proj_qkv
  - linear_attn.in_proj_z
  - linear_attn.out_proj
```

### Routed Experts (MoE)

To apply LoRA to routed expert parameters, add `lora_target_parameters`:

```yaml
lora_target_parameters:
  - mlp.experts.gate_up_proj
  - mlp.experts.down_proj
#  - mlp.gate.weight  # router
```

### Shared Experts (MoE)

Shared experts use `nn.Linear` (unlike routed experts which are 3D `nn.Parameter` tensors), so they can be targeted via `lora_target_modules`. To also train shared expert projections alongside attention, uncomment `gate_up_proj` and `down_proj` in `lora_target_modules`:

```yaml
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  # Add gate_up_proj and down_proj to also target shared experts (nn.Linear):
  # - gate_up_proj
  # - down_proj
```

Use `lora_target_parameters` (see [Routed Experts](#routed-experts-moe) above) to target routed experts separately.

### TIPS

- For inference hyp, please see the respective model card details.
- You can run a full finetuning of smaller configs by removing `adapter: qlora` and `load_in_4bit: true`. See [Multi-GPU](#optimization-guides) below.
- Read more on loading your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).
- For **multimodal** finetuning, set `processor_type: AutoProcessor`, `skip_prepare_dataset: true`, and `remove_unused_columns: false` as shown in `9b-lora-vision.yaml`.

## Optimization Guides

- [Optimizations Guide](https://docs.axolotl.ai/docs/optimizations.html)

## Related Resources

- [Qwen3.5 Blog](https://qwenlm.github.io/blog/qwen3.5/)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
