# Finetune Magistral Small with Axolotl

Magistral Small is a 24B parameter opensource model from MistralAI found on HuggingFace at [2506](https://huggingface.co/mistralai/Magistral-Small-2506), [2507](https://huggingface.co/mistralai/Magistral-Small-2507) (see [Thinking](#thinking)), and [2509](https://huggingface.co/mistralai/Magistral-Small-2509) (see [Vision](#vision)). This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

MistralAI has also released a proprietary medium-sized version called Magistral Medium.

Thanks to the team at MistralAI for giving us early access to prepare for these releases.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

    Here is an example of how to install from pip:

```bash
# Ensure you have Pytorch installed (Pytorch 2.7.0 min)
pip3 install packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation 'axolotl[flash-attn]>=0.12.0'
```

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage

```bash
python scripts/cutcrossentropy_install.py | sh
```

3. Run the finetuning example:

```bash
axolotl train examples/magistral/magistral-small-qlora.yaml
```

This config uses about 24GB VRAM.

Let us know how it goes. Happy finetuning! 🚀

### Thinking

MistralAI has released their [2507](https://huggingface.co/mistralai/Magistral-Small-2507) model with thinking capabilities, enabling Chain-of-Thought reasoning with explicit thinking steps.

📚 **[See the Thinking fine-tuning guide →](./think/README.md)**

### Vision

MistralAI has released their [2509](https://huggingface.co/mistralai/Magistral-Small-2509) model with vision capabilities.

📚 **[See the Vision fine-tuning guide →](./vision/README.md)**

### Tips

- We recommend adding the same/similar SystemPrompt that the model is tuned for. You can find this within the repo's files titled `SYSTEM_PROMPT.txt`.
- For inference, the official MistralAI team recommends `top_p: 0.95` and `temperature: 0.7` with `max_tokens: 40960`.
- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The text dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

- [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
- [LoRA Optimizations](https://docs.axolotl.ai/docs/lora_optims.html)

## Limitations

We only support the `mistral-common` tokenizer for Supervised Fine-tuning at the moment and for `type: chat_template` only.

In addition, we do not support overriding tokens yet.

## Related Resources

- [MistralAI Magistral Blog](https://mistral.ai/news/magistral/)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)


## Future Work

- Add parity to Preference Tuning, RL, etc.
- Add parity to other tokenizer configs like overriding tokens.
