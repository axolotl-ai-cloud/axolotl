# Finetune Liquid Foundation Models 2 (LFM2) with Axolotl

[Liquid Foundation Models 2 (LFM2)](https://huggingface.co/collections/LiquidAI/lfm2-686d721927015b2ad73eaa38) are a family of small, open-weight models from [Liquid AI](https://www.liquid.ai/) focused on quality, speed, and memory efficiency. Liquid AI released text-only [LFM2](https://huggingface.co/collections/LiquidAI/lfm2-686d721927015b2ad73eaa38) and text+vision [LFM2-VL](https://huggingface.co/collections/LiquidAI/lfm2-vl-68963bbc84a610f7638d5ffa) models.

LFM2 features a new hybrid Liquid architecture with multiplicative gates, short-range convolutions, and grouped query attention, enabling fast training and inference.

This guide shows how to fine-tune both the LFM2 and LFM2-VL models with Axolotl.

Thanks to the team at LiquidAI for giving us early access to prepare for these releases.

## Getting Started

1.  Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

    Here is an example of how to install from pip:
    ```bash
    # Ensure you have a compatible version of Pytorch installed
    pip3 install packaging setuptools wheel ninja
    pip3 install --no-build-isolation 'axolotl[flash-attn]>=0.12.0'
    ```

2.  Run one of the finetuning examples below.

    **LFM2**
    ```bash
    # FFT SFT (1x48GB @ 25GiB)
    axolotl train examples/LiquidAI/lfm2-350m-fft.yaml
    ```

    **LFM2-VL**
    ```bash
    # LoRA SFT (1x48GB @ 2.7GiB)
    axolotl train examples/LiquidAI/lfm2-vl-lora.yaml
    ```

    **LFM2-MoE**
    ```bash
    pip install git+https://github.com/huggingface/transformers.git@0c9a72e4576fe4c84077f066e585129c97bfd4e6

    # LoRA SFT (1x48GB @ 16.2GiB)
    axolotl train examples/LiquidAI/lfm2-8b-a1b-lora.yaml
    ```

### TIPS

- **Installation Error**: If you encounter `ImportError: ... undefined symbol ...` or `ModuleNotFoundError: No module named 'causal_conv1d_cuda'`, the `causal-conv1d` package may have been installed incorrectly. Try uninstalling it:
  ```bash
  pip uninstall -y causal-conv1d
  ```

- **Dataset Loading**: Read more on how to load your own dataset in our [documentation](https://docs.axolotl.ai/docs/dataset_loading.html).
- **Dataset Formats**:
  - For LFM2 models, the dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).
  - For LFM2-VL models, Axolotl follows the multi-content Messages format. See our [Multimodal docs](https://docs.axolotl.ai/docs/multimodal.html#dataset-format) for details.

## Optimization Guides

- [Optimizations Guide](https://docs.axolotl.ai/docs/optimizations.html)

## Related Resources

- [LFM2 Blog](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)
- [LFM2-VL Blog](https://www.liquid.ai/blog/lfm2-vl-efficient-vision-language-models)
- [LFM2-MoE Blog](https://www.liquid.ai/blog/lfm2-8b-a1b-an-efficient-on-device-mixture-of-experts)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
