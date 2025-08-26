# Finetune ByteDance's Seed-OSS with Axolotl

[Seed-OSS](https://huggingface.co/collections/arcee-ai/afm-45b-68823397c351603014963473) are a series of 36B parameter open source models trained by ByteDance's Seed Team.

This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html). You need to install from main as Seed-OSS is only on nightly or use our latest [Docker images](https://docs.axolotl.ai/docs/docker.html).

    Here is an example of how to install from main for pip:

```bash
# Ensure you have Pytorch installed (Pytorch 2.6.0 min)
git clone https://github.com/axolotl-ai-cloud/axolotl.git
cd axolotl

pip3 install packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation -e '.[flash-attn]'
```

2. Install their transformers PR:

```bash
pip3 uninstall -y transformers

pip3 install git+https://github.com/huggingface/transformers.git@56d68c6706ee052b445e1e476056ed92ac5eb383
```

3. Run the finetuning example:

```bash
axolotl train examples/seed-oss/seed-oss-36b-qlora.yaml
```

This config uses about (XYZ) GiB VRAM.

Let us know how it goes. Happy finetuning! 🚀

### TIPS

- For inference, the official Arcee.ai team recommends `top_p=0.95` and `temperature=1.1`.
- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

- [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
- [LoRA Optimizations](https://docs.axolotl.ai/docs/lora_optims.html)

## Related Resources

- [ByteDance Seed Website](https://seed.bytedance.com/)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
