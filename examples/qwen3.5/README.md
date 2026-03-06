# Finetune Qwen3.5 with Axolotl

[Qwen3.5](https://huggingface.co/collections/Qwen/qwen35-68452f3bc6e4b7cfb4e1c803) is a hybrid architecture model series combining Gated DeltaNet linear attention with standard Transformer attention. Models from 7B onwards are early-fusion vision-language models (`Qwen3_5ForConditionalGeneration`), meaning vision and text tokens are processed through the same transformer stack. The 2B variant is text-only.

Available configs:

| Config | Model | Type |
|---|---|---|
| `27b-qlora.yaml` | Qwen3.5-27B | Dense VLM, text-only path |
| `35b-a3b-moe-qlora.yaml` | Qwen3.5-35B-A3B | MoE, text-only path |
| `122b-a10b-moe-qlora.yaml` | Qwen3.5-122B-A10B | MoE, text-only path |
| `7b-lora-vision.yaml` | Qwen3.5-7B | Vision+text (multimodal) |

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Install FLA for sample packing support with the Gated DeltaNet linear attention layers:
```bash
pip3 uninstall -y causal-conv1d && pip3 install flash-linear-attention==0.4.1
```
> FLA is required when `sample_packing: true`. Without it, training raises a `RuntimeError` on packed sequences. Vision configs use `sample_packing: false` so FLA is optional there.

4. Run a finetuning example:

```bash
# Dense 27B text-only (QLoRA, ~47 GiB VRAM with sample packing)
axolotl train examples/qwen3.5/27b-qlora.yaml

# MoE 35B-A3B text-only (QLoRA)
axolotl train examples/qwen3.5/35b-a3b-moe-qlora.yaml

# MoE 122B-A10B text-only (QLoRA)
axolotl train examples/qwen3.5/122b-a10b-moe-qlora.yaml

# 7B vision+text (LoRA, multimodal dataset)
axolotl train examples/qwen3.5/7b-lora-vision.yaml
```

### TIPS

- For inference, you can experiment with `temperature: 0.7`, `top_p: 0.8`, `top_k: 20`, and `min_p: 0`.
- You can run a full finetuning by removing `adapter: qlora` and `load_in_4bit: true`. See [Multi-GPU](#optimization-guides) below.
- Read more on loading your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).
- For **multimodal** finetuning, set `processor_type: AutoProcessor`, `skip_prepare_dataset: true`, and `remove_unused_columns: false` as shown in `7b-lora-vision.yaml`.
- The Gated DeltaNet linear attention layers (`linear_attn.*`) can optionally be added to `lora_target_modules` — they are commented out by default.

## Optimization Guides

- [Optimizations Guide](https://docs.axolotl.ai/docs/optimizations.html)

## Related Resources

- [Qwen3.5 Blog](https://qwenlm.github.io/blog/qwen3.5/)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
