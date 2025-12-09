# Finetune SmolVLM2 with Axolotl

[SmolVLM2](https://huggingface.co/collections/HuggingFaceTB/smolvlm2-smallest-video-lm-ever-67ab6b5e84bf8aaa60cb17c7) are a family of lightweight, open-source multimodal models from HuggingFace designed to analyze and understand video, image, and text content.

These models are built for efficiency, making them well-suited for on-device applications where computational resources are limited. Models are available in multiple sizes, including 2.2B, 500M, and 256M.

This guide shows how to fine-tune SmolVLM2 models with Axolotl.

## Getting Started

1.  Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

    Here is an example of how to install from pip:
    ```bash
    # Ensure you have a compatible version of Pytorch installed
    pip3 install packaging setuptools wheel ninja
    pip3 install --no-build-isolation 'axolotl[flash-attn]>=0.12.0'
    ```

2. Install an extra dependency:

    ```bash
    pip3 install num2words==0.5.14
    ```

3.  Run the finetuning example:

    ```bash
    # LoRA SFT (1x48GB @ 6.8GiB)
    axolotl train examples/smolvlm2/smolvlm2-2B-lora.yaml
    ```

## TIPS

- **Dataset Format**: For video finetuning, your dataset must be compatible with the multi-content Messages format. For more details, see our documentation on [Multimodal Formats](https://docs.axolotl.ai/docs/multimodal.html#dataset-format).
- **Dataset Loading**: Read more on how to prepare and load your own datasets in our [documentation](https://docs.axolotl.ai/docs/dataset_loading.html).

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Related Resources

- [SmolVLM2 Blog](https://huggingface.co/blog/smolvlm2)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
