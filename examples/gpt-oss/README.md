# Finetune OpenAI's GPT-OSS with Axolotl

[GPT-OSS](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4) are a family of open-weight MoE models trained by OpenAI, released in August 2025. There are two variants: a 20B and 120B.

This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html). You need to install from main as GPT-OSS is only on nightly or use our latest [Docker images](https://docs.axolotl.ai/docs/docker.html).

    Here is an example of how to install from main for pip:

```bash
# Ensure you have Pytorch installed (Pytorch 2.6.0 min)
git clone https://github.com/axolotl-ai-cloud/axolotl.git
cd axolotl

pip3 install packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation -e '.[flash-attn]'
```

2. Choose one of the following configs below for training the 20B model.

```bash
# LoRA SFT linear layers & 2 experts (1x48GB)
# (only linear layers -> ~44GiB)
axolotl train examples/gpt-oss/gpt-oss-20b-sft-lora-singlegpu.yaml

# FFT SFT with offloading (2x24GB, ~21GiB/GPU)
axolotl train examples/gpt-oss/gpt-oss-20b-fft-fsdp2-offload.yaml

# FFT SFT (8x48gb, ~36GiB/GPU)
axolotl train examples/gpt-oss/gpt-oss-20b-fft-fsdp2.yaml
```

Note: 120B coming soon!

### TIPS

- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

- [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)

## Related Resources

- [GPT-OSS Blog](https://openai.com/index/introducing-gpt-oss/)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
