# Finetune Xiaomi's MiMo with Axolotl

[MiMo](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL) is a family of models trained from scratch for reasoning tasks, incorporating **Multiple-Token Prediction (MTP)** as an additional training objective for enhanced performance and faster inference. Pre-trained on ~25T tokens with a three-stage data mixture strategy and optimized reasoning pattern density.

This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Run the finetuning example:

    ```bash
    axolotl train examples/mimo/mimo-7b-qlora.yaml
    ```

This config uses about 17.2 GiB VRAM. Let us know how it goes. Happy finetuning! 🚀

### Tips

- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Limitations

**Cut Cross Entropy (CCE)**: Currently not supported. We plan to include CCE support for MiMo in the near future.

## Related Resources

- [MiMo Paper](https://arxiv.org/abs/2505.07608)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
