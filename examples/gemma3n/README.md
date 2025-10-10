# Finetune Gemma-3n with Axolotl

Gemma-3n is a family of multimodal models from Google found on [HuggingFace](https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4). This guide shows how to fine-tune it with Axolotl.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

    Here is an example of how to install from pip:

```bash
# Ensure you have Pytorch installed (Pytorch 2.6.0 min)
pip3 install packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation 'axolotl[flash-attn]>=0.12.0'
```

2. In addition to Axolotl's requirements, Gemma-3n requires:

```bash
pip3 install timm==1.0.17

# for loading audio data
pip3 install librosa==0.11.0
```

3. Download sample dataset files

```bash
# for text + vision + audio only
wget https://huggingface.co/datasets/Nanobit/text-vision-audio-2k-test/resolve/main/African_elephant.jpg
wget https://huggingface.co/datasets/Nanobit/text-vision-audio-2k-test/resolve/main/En-us-African_elephant.oga
```

4. Run the finetuning example:

```bash
# text only
axolotl train examples/gemma3n/gemma-3n-e2b-qlora.yml

# text + vision
axolotl train examples/gemma3n/gemma-3n-e2b-vision-qlora.yml

# text + vision + audio
axolotl train examples/gemma3n/gemma-3n-e2b-vision-audio-qlora.yml
```

Let us know how it goes. Happy finetuning! 🚀

WARNING: The loss and grad norm will be much higher than normal. We suspect this to be inherent to the model as of the moment. If anyone would like to submit a fix for this, we are happy to take a look.

### TIPS

- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The text dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).
- The multimodal dataset format follows the OpenAI multi-content Messages format as seen [here](https://docs.axolotl.ai/docs/multimodal.html#dataset-format).

## Optimization Guides

- [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
- [LoRA Optimizations](https://docs.axolotl.ai/docs/lora_optims.html)

## Related Resources

- [Gemma 3n Blog](https://ai.google.dev/gemma/docs/gemma-3n)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
