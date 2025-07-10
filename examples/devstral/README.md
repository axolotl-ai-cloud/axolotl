# Finetune Devstral with Axolotl

Devstral Small is a 24B parameter opensource model from MistralAI found on HuggingFace [Devstral-Small-2505](https://huggingface.co/mistralai/Devstral-Small-2505). This guide shows how to fine-tune it with Axolotl with multi-turn conversations with proper masking.

The model was fine-tuned ontop of [Mistral-Small-3.1](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Base-2503) without the vision layer and has a context of upto 128k tokens.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html). You need to install from main as Devstral is only on nightly or use our latest [Docker images](https://docs.axolotl.ai/docs/docker.html).

    Here is an example of how to install from main for pip:

```bash
# Ensure you have Pytorch installed (Pytorch 2.6.0+)
git clone https://github.com/axolotl-ai-cloud/axolotl.git
cd axolotl

pip3 install packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation -e '.[flash-attn]'

# Install the latest mistral-common from source
pip3 uninstall mistral-common
pip3 install git+https://github.com/mistralai/mistral-common.git@039465d

```

2. Run the finetuning example:

```bash
axolotl train examples/devstral/devstral-small-qlora.yml
```

This config uses about 21GB VRAM.

Let us know how it goes. Happy finetuning! 🚀

### TIPS

- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

- [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
- [LoRA Optimizations](https://docs.axolotl.ai/docs/lora_optims.html)
- [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy)
- [Liger Kernel](https://docs.axolotl.ai/docs/custom_integrations.html#liger-kernels)

## Limitations

We only support the `mistral-common` tokenizer for Supervised Fine-tuning at the moment and for `type: chat_template` only.

In addition, we do not support overriding tokens yet.

## Related Resources

- [MistralAI Devstral Blog](https://mistral.ai/news/devstral)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)


## Future Work

- Add parity to Preference Tuning, RL, Multi-modal, etc.
- Add parity to other tokenizer configs like overriding tokens.
