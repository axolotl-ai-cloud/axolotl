# Finetune Qwen3.8-Flash-Next with Axolotl

[Qwen3.8-Flash-Next](https://huggingface.co/collections/Qwen/qwen38-flash-next) is a large hybrid-attention MoE model combining Gated DeltaNet linear attention with periodic full attention, a sparse-attention indexer, and multi-token prediction, plus a vision tower for image-text-to-text inputs.

This guide shows how to fine-tune the text path with Axolotl using QLoRA.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Run the finetuning example:

```bash
axolotl train examples/qwen3.8-flash-next/qwen3.8-flash-next-qlora.yaml
```

## TIPS

- This config trains on text-only conversation data; no `pixel_values` are supplied, so the vision tower is not exercised.
- Uncomment the `linear_attn.*` entries in `lora_target_modules` to also adapt the Gated DeltaNet projections.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

- [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
- [LoRA Optimizations](https://docs.axolotl.ai/docs/lora_optims.html)

## Related Resources

- [Qwen3.8-Flash-Next Collection](https://huggingface.co/collections/Qwen/qwen38-flash-next)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
