# Finetune Devstral with Axolotl

Devstral Small is a 24B parameter opensource model from MistralAI found on HuggingFace [Devstral-Small-2505](https://huggingface.co/mistralai/Devstral-Small-2505) and [Devstral-Small-2507](https://huggingface.co/mistralai/Devstral-Small-2507). `Devstral-Small-2507` is the latest version of the model and has [function calling](https://mistralai.github.io/mistral-common/usage/tools/) support.

This guide shows how to fine-tune it with Axolotl with multi-turn conversations with proper masking.

The model was fine-tuned ontop of [Mistral-Small-3.1](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Base-2503) without the vision layer and has a context of up to 128k tokens.

Thanks to the team at MistralAI for giving us early access to prepare for this release.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

    Here is an example of how to install from pip:

```bash
# Ensure you have Pytorch installed (Pytorch 2.6.0 min)
pip3 install packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation 'axolotl[flash-attn]>=0.12.0'
```

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage

```bash
python scripts/cutcrossentropy_install.py | sh
```

3. Run the finetuning example:

```bash
axolotl train examples/devstral/devstral-small-qlora.yml
```

This config uses about 21GB VRAM.

Let us know how it goes. Happy finetuning! 🚀

### TIPS

- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).
- Learn how to use function calling with Axolotl at [docs](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#using-tool-use).

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
- [MistralAI Devstral 1.1 Blog](https://mistral.ai/news/devstral-2507)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)


## Future Work

- Add parity to Preference Tuning, RL, Multi-modal, etc.
- Add parity to other tokenizer configs like overriding tokens.
