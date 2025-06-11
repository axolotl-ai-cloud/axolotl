# Finetune Magistral Small with Axolotl

Magistral Small is a 24B parameter opensource model from MistralAI found on [HuggingFace](https://huggingface.co/mistralai/Magistral-Small-2506). This guide shows how to fine-tune it with Axolotl with multi-turn conversations with proper masking.

MistralAI has also released a proprietary medium-sized version called Magistral Medium.

Thanks to the team at MistralAI for giving us early access to prepare for this release.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html). You need to install from main as Magistral is only on nightly or use our latest [Docker images](https://docs.axolotl.ai/docs/docker.html).

    Here is an example of how to install from main for pip:

```bash
# Ensure you have Pytorch installed (we recommend Pytorch 2.6.0)
git clone https://github.com/axolotl-ai-cloud/axolotl.git
cd axolotl

pip3 install packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation -e '.[flash-attn,mistral]'
```

2. Download the example config:

```bash
axolotl fetch examples
```

3. Run the finetuning example:

```bash
axolotl train examples/magistral/magistral-small-qlora.yaml
```

Let us know how it goes. Happy finetuning! 🚀

### TIPS

- For inference, the official MistralAI team recommends `top_p: 0.95` and `temperature: 0.7` with `max_tokens: 40960`.
- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format is the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

- [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
- [LoRA Optimizations](https://docs.axolotl.ai/docs/lora_optims.html)

## Limitations

We only support the `mistral-common` tokenizer for Supervised Fine-tuning at the moment and for `type: chat_template` only.

The tokenizer does not work with `dataset.map` with multiprocessing, so we had to disable it. In addition, we do not support overriding tokens yet.

## Related Resources

- [Magistral Blog](https://mistral.ai/news/magistral/)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)


## Future Work

- Add parity to Preference Tuning, RL, Multi-modal, etc.
- Add parity to other tokenizer configs like overriding tokens.
