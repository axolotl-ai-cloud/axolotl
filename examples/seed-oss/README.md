# Finetune ByteDance's Seed-OSS with Axolotl

[Seed-OSS](https://huggingface.co/collections/ByteDance-Seed/seed-oss-68a609f4201e788db05b5dcd) are a series of 36B parameter open source models trained by ByteDance's Seed Team.

This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

## Getting started

1.  Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

    Here is an example of how to install from pip:
    ```bash
    # Ensure you have a compatible version of Pytorch installed
    pip3 install packaging setuptools wheel ninja
    pip3 install --no-build-isolation 'axolotl[flash-attn]>=0.12.0'

    # Install Cut Cross Entropy
    python scripts/cutcrossentropy_install.py | sh
    ```

2. Run the finetuning example:

```bash
axolotl train examples/seed-oss/seed-oss-36b-qlora.yaml
```

This config uses about 27.7 GiB VRAM.

Let us know how it goes. Happy finetuning! 🚀

### TIPS

- For inference, the official Seed Team recommends `top_p=0.95` and `temperature=1.1`.
- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Related Resources

- [ByteDance Seed Website](https://seed.bytedance.com/)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
