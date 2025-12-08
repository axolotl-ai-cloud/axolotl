# Finetune Allenai's Olmo 3 with Axolotl

[Olmo 3](https://huggingface.co/collections/allenai/olmo-3) are a family of 7B and 32B models open source models trained by The Allen Institute for Artificial Intelligence.

This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Run the finetuning example:

    ```bash
    axolotl train examples/olmo3/olmo3-7b-qlora.yaml
    ```

Let us know how it goes. Happy finetuning! 🚀

### TIPS

- The example config can be re-used for Olmo and Olmo 2.
- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Related Resources

- [Olmo 3 Blog](https://allenai.org/blog/olmo3)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
