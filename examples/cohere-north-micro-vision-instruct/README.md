# Finetune North Micro Vision Instruct with Axolotl

[North Micro Vision Instruct](https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct) is a 2.4B-parameter open-weight vision-language model comprising a 2B language model and a custom-trained 400M native-resolution vision encoder. Its compact scale and broad image-understanding capabilities make it a practical foundation for task-specific customization.

This guide shows how to fine-tune the model using Axolotl's multimodal supervised fine-tuning (SFT) support. For architecture, training, and benchmark details, read the [North Micro Vision technical blog post](https://huggingface.co/blog/CohereLabs/meet-north-micro-vision-instruct).

Thanks to the Cohere team for providing early access ahead of the release.

## Getting Started

1. Install Axolotl from main following the [installation guide](https://docs.axolotl.ai/docs/installation.html#sec-edge-build).

2. Install Transformers. This model lands in v5.16, which is not released yet, so install the 5.16 dev build from source for now:

    ```bash
    uv pip install "transformers @ git+https://github.com/huggingface/transformers.git"
    ```

    Once v5.16 is released, `uv pip install "transformers>=5.16.0"` is enough.

3. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

4. Run one of the finetuning examples:

    ```bash
    # 5.1 GiB VRAM, adapters on the language decoder
    axolotl train examples/cohere-north-micro-vision-instruct/qlora.yaml

    # 5.2 GiB VRAM, adapters on the decoder, vision tower and projector
    axolotl train examples/cohere-north-micro-vision-instruct/qlora-vision.yaml

    # 21.2 GiB VRAM
    axolotl train examples/cohere-north-micro-vision-instruct/fft.yaml
    ```

Let us know how it goes. Happy finetuning! 🚀

### TIPS

- Liger Kernels RMSNorm, RoPE and FLCE are not supported.
- For Transformers inference, use the model's generation defaults: `temperature=0.7`, `top_p=0.8`, and `top_k=20`.
- Public vLLM support is coming soon. For vLLM inference, use `temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, and `repetition_penalty=1.0`.
- Images are processed at native resolution, so token count scales with pixel count and `sequence_len` does not bound it. Cap it with the image processor's `max_pixels`:

    ```yaml
    processor_kwargs:
      max_pixels: 1003520  # total pixels
    ```

- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Related Resources

- [North Micro Vision Instruct](https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct)
- [North Micro Vision technical blog post](https://huggingface.co/blog/CohereLabs/meet-north-micro-vision-instruct)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
