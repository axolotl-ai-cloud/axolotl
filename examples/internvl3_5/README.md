# Finetune OpenGV's InternVL with Axolotl

[InternVL 3.5](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF) is a family of powerful vision-language models supporting dynamic resolution and multi-image understanding by OpenGV. It features a ViT-style vision encoder and strong language model backbone for tasks like visual question answering, OCR, and scene text understanding.

This guide shows how to fine-tune it with Axolotl.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install `timm` for vision model support:

    ```bash
    pip install timm==1.0.17
    ```

3. Run the finetuning example:

    ```bash
    axolotl train examples/internvl3_5/internvl3_5-8b-qlora.yml
    ```

Let us know how it goes. Happy finetuning! 🚀

### Tips

- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the multi-modal format as seen [here](https://docs.axolotl.ai/docs/multimodal.html#dataset-format).

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Limitations

**Cut Cross Entropy (CCE)**: Currently not supported. We plan to include CCE support for InternVL in the near future.

## Related Resources

- [InternVL Paper](https://huggingface.co/papers/2508.18265)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
