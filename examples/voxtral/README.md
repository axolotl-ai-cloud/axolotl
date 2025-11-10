# Finetune Voxtral with Axolotl

Voxtral is a [3B](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)/[24B](https://huggingface.co/mistralai/Voxtral-Small-24B-2507) parameter opensource model from MistralAI found on HuggingFace. This guide shows how to fine-tune it with Axolotl.

Thanks to the team at MistralAI for giving us early access to prepare for this release.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

    Here is an example of how to install from pip:

```bash
# Ensure you have Pytorch installed (Pytorch 2.6.0 min)
pip3 install packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation 'axolotl[flash-attn]>=0.12.0'
```

2. Please install the below.

```bash
# audio
pip3 install librosa==0.11.0
pip3 install 'mistral_common[audio]==1.8.3'

# Install CCE https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy
python scripts/cutcrossentropy_install.py | sh
```

3. Download sample dataset files

```bash
# for text + audio only
wget https://huggingface.co/datasets/Nanobit/text-audio-2k-test/resolve/main/En-us-African_elephant.oga
```

4. Run the finetuning example:

```bash
# text only
axolotl train examples/voxtral/voxtral-mini-qlora.yml

# text + audio
axolotl train examples/voxtral/voxtral-mini-audio-qlora.yml
```

These configs use about 4.8 GB VRAM.

Let us know how it goes. Happy finetuning! 🚀

### TIPS

- For inference, the official MistralAI team recommends `temperature: 0.2` and `top_p: 0.95` for audio understanding and `temperature: 0.0` for transcription.
- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The text dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).
- The multimodal dataset format follows the OpenAI multi-content Messages format as seen [here](https://docs.axolotl.ai/docs/multimodal.html#dataset-format).


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
