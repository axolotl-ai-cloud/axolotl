# Finetune ArceeAI's AFM with Axolotl

[Arcee Foundation Models (AFM)](https://huggingface.co/collections/arcee-ai/afm-45b-68823397c351603014963473) are a family of 4.5B parameter open weight models trained by Arcee.ai.

This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

Thanks to the team at Arcee.ai for using Axolotl in supervised fine-tuning the AFM model.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html). You need to install from main as AFM is only on nightly or use our latest [Docker images](https://docs.axolotl.ai/docs/docker.html).

    Here is an example of how to install from main for pip:

```bash
# Ensure you have Pytorch installed (Pytorch 2.6.0 min)
git clone https://github.com/axolotl-ai-cloud/axolotl.git
cd axolotl

pip3 install packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation -e '.[flash-attn]'

# Install CCE https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy
python scripts/cutcrossentropy_install.py | sh
```

2. Run the finetuning example:

```bash
axolotl train examples/arcee/afm-4.5b-qlora.yaml
```

This config uses about 7.8GiB VRAM.

Let us know how it goes. Happy finetuning! 🚀

### TIPS

- For inference, the official Arcee.ai team recommends `top_p: 0.95`, `temperature: 0.5`, `top_k: 50`, and `repeat_penalty: 1.1`.
- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

- [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
- [LoRA Optimizations](https://docs.axolotl.ai/docs/lora_optims.html)

## Related Resources

- [AFM Blog](https://docs.arcee.ai/arcee-foundation-models/introduction-to-arcee-foundation-models)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
