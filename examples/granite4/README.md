# Finetune IBM's Granite 4.0 with Axolotl

[Granite 4.0](https://huggingface.co/collections/ibm-granite/granite-40-language-models) are a family of open source models trained by IBM Research.

This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html). You need to install from main as Granite4 is only on nightly or use our latest [Docker images](https://docs.axolotl.ai/docs/docker.html).

    Here is an example of how to install from main for pip:

```bash
# Ensure you have Pytorch installed (Pytorch 2.7.1 min)
git clone https://github.com/axolotl-ai-cloud/axolotl.git
cd axolotl

pip3 install packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation -e '.[flash-attn]'

# Install CCE https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy
python scripts/cutcrossentropy_install.py | sh
```

2. Run the finetuning example:

```bash
axolotl train examples/granite4/granite-4.0-tiny-fft.yaml
```

This config uses about 40.8GiB VRAM.

Let us know how it goes. Happy finetuning! 🚀

### TIPS

- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

### Limitation

Adapter finetuning does not work at the moment. It would error with

```bash
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4096x3072 and 1x1179648)
```

In addition, if adapter training works, `lora_target_linear: true` will not work due to:
```bash
ValueError: Target module GraniteMoeHybridParallelExperts() is not supported.
```

## Optimization Guides

- [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
- [LoRA Optimizations](https://docs.axolotl.ai/docs/lora_optims.html)

## Related Resources

- [Granite Docs](https://www.ibm.com/granite/docs/models/granite)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
