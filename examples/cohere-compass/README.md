# Finetune North Micro Vision with Axolotl

[North Micro Vision Instruct](https://huggingface.co/CohereLabs/North-Micro-Vision-instruct-preview) is a 2.4B parameters (a 2B language decoder plus a 400M SigLIP-shaped vision encoder), with native-resolution image support. It is built for fine-tuning and document understanding, and is strongest on structured data extraction, scientific figure understanding, and document Q&A.

This guide shows how to fine-tune it with Axolotl's multimodal SFT path.

Thanks to the team at CohereLabs for giving us early access to prepare for this release.

## Getting Started

1. Install Axolotl following the main from the [installation guide](https://docs.axolotl.ai/docs/installation.html#sec-edge-build).

2. Ensure `uv pip install transformers>=5.15.0`.

3. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

4. Run one of the finetuning examples:

    ```bash
    # 5.1 GiB VRAM, adapters on the language decoder
    axolotl train examples/cohere-compass/north-micro-vision-qlora.yaml

    # 5.2 GiB VRAM, adapters on the decoder, vision tower and patch mergers
    axolotl train examples/cohere-compass/north-micro-vision-qlora-vision.yaml

    # 21.2 GiB VRAM
    axolotl train examples/cohere-compass/north-micro-vision-fft.yaml
    ```

Let us know how it goes. Happy finetuning! 🚀

### TIPS

- Liger Kernels rope and FLCE are not supported.
- For inference, Cohere recommends `temperature: 1.0` and `top_p: 0.95`.
- Images are processed at native resolution, so token count scales with pixel count and `sequence_len` does not bound it. Cap it with the image processor's `max_pixels`:

    ```yaml
    processor_kwargs:
      max_pixels: 1003520  # 28*28*1280, the default longest edge
    ```

- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Related Resources

- [North Micro Vision Instruct](https://huggingface.co/CohereLabs/North-Micro-Vision-instruct-preview)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl multimodal docs](https://docs.axolotl.ai/docs/multimodal.html)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
