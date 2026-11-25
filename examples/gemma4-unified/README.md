# Finetune Google's Gemma 4 Unified with Axolotl

[Gemma 4 Unified](https://huggingface.co/google/gemma-4-12B-it) is the **encoder-free** multimodal member of the [Gemma 4](https://huggingface.co/collections/google/gemma-4) family; no vision tower and no audio tower. Raw image patches and 16 kHz waveform frames are projected directly into the language model through lightweight `LayerNorm/RMSNorm → Linear` pipelines. The text backbone is the standard Gemma 4 decoder (mixed sliding/global attention, `global_head_dim=512`, optional KV sharing).

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Run the finetuning example:

```bash
# Text LoRA (1x96GB @ ~27.76 GiB)
axolotl train examples/gemma4-unified/12b-text-lora.yaml

# Vision LoRA (1x96GB @ ~26.5 GiB)
axolotl train examples/gemma4-unified/12b-vision-lora.yaml
```

## Limitations

- **Attention**: FA2 (max head_dim=256) / FA4 (max head_dim=128) cannot serve `global_head_dim=512` on their own. For `sample_packing: true`, use `flex_attention` or `gemma4_hybrid_attn_impl: true` (with `flash_attention` varlen for sliding-window layers + block-diagonal `sdpa` for global layers, packing-safe).
- **lora_target_linear**: incompatible for multimodal. Use `lora_target_modules` as seen in vision example.

### TIPS

- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The text dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).
- The multimodal dataset format follows the OpenAI multi-content Messages format as seen [here](https://docs.axolotl.ai/docs/multimodal.html#dataset-format).

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Related Resources

- [Gemma 4 Blog](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12B/)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
