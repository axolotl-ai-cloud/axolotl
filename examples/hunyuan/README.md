# Finetune HunYuan with Axolotl

Tencent released a family of opensource models called HunYuan with varying parameter scales of 0.5B, 1.8B, 4B, and 7B scale for both Pre-trained and Instruct variants. The models can be found at [HuggingFace](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7). This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html). You need to install from main as HunYuan is only on nightly or use our latest [Docker images](https://docs.axolotl.ai/docs/docker.html).

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
axolotl train examples/hunyuan/hunyuan-v1-dense-qlora.yaml
```

This config uses about 4.7 GB VRAM.

Let us know how it goes. Happy finetuning! 🚀

### Dataset

HunYuan Instruct models can choose to enter a slow think or fast think pattern. For best performance on fine-tuning their Instruct models, your dataset should be adjusted to match their pattern.

```python
# fast think pattern
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "/no_think What color is the sun?" },
    {"role": "assistant", "content": "<think>\n\n</think>\n<answer>\nThe sun is yellow.\n</answer>"}
]

# slow think pattern
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "/no_think What color is the sun?" },
    {"role": "assistant", "content": "<think>\nThe user is asking about the color of the sun. I need to ...\n</think>\n<answer>\nThe sun is yellow.\n</answer>"}
]
```

### TIPS

- For inference, the official Tencent team recommends

```json

{
  "do_sample": true,
  "top_k": 20,
  "top_p": 0.8,
  "repetition_penalty": 1.05,
  "temperature": 0.7
}

```

- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

- [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
- [LoRA Optimizations](https://docs.axolotl.ai/docs/lora_optims.html)

## Related Resources

- [Tencent HunYuan Blog](https://hunyuan.tencent.com/)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
