# Finetune ArceeAI's Trinity with Axolotl

[Trinity](https://huggingface.co/collections/arcee-ai/trinity) is a family of open weight MoE models trained by Arcee.ai.

This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

## Getting started

1. Install Axolotl following the main from the [installation guide](https://docs.axolotl.ai/docs/installation.html#sec-edge-build).

2. Run the finetuning example:

    ```bash
    axolotl train examples/trinity/trinity-nano-preview-qlora.yaml
    ```

This config uses about 24.9 GiB VRAM.

Let us know how it goes. Happy finetuning! 🚀

### TIPS

- For inference, the official Arcee.ai team recommends `top_p: 0.75`, `temperature: 0.15`, `top_k: 50`, and `min_p: 0.06`.
- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Related Resources

- [Trinity Blog](https://www.arcee.ai/blog/the-trinity-manifesto)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
