# Finetune Mistral Small 4 with Axolotl

Mistral Small 4 is a 119B parameter MoE vision-language model from MistralAI found on HuggingFace at [Mistral-Small-4-119B-2602-HF](https://huggingface.co/mistralai/Mistral-Small-4-119B-2602-HF).

Thanks to the team at MistralAI for giving us early access to prepare for this release.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage

3. Run one of the example configs:

```bash
# text-only
axolotl train examples/mistral4/qlora-text.yml
axolotl train examples/mistral4/fft-text.yml

# vision
axolotl train examples/mistral4/qlora-vision.yml
axolotl train examples/mistral4/fft-vision.yml
```

## Tips

- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The text dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Related Resources

- [MistralAI Mistral Small 4 Blog](https://mistral.ai/news/mistral-4)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
