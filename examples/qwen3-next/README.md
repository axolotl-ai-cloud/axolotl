# Finetune Qwen3-Next with Axolotl

[Qwen3-Next](https://huggingface.co/collections/Qwen/qwen3-next-68c25fd6838e585db8eeea9d) represents the next-generation foundation models optimized for extreme context length and large-scale parameter efficiency. The series introduces architectural innovations including Hybrid Attention (Gated DeltaNet + Gated Attention), High-Sparsity MoE with 1:50 activation ratio, and Multi-Token Prediction for enhanced performance and inference acceleration.

This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html). You need to install from main as Qwen3-Next is only on nightly or use our latest [Docker images](https://docs.axolotl.ai/docs/docker.html).

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

2. Install Qwen3-Next transformers commit
```bash
pip3 uninstall -y transformers && pip3 install "git+https://github.com/huggingface/transformers.git@b9282355bea846b54ed850a066901496b19da654"
```

3. Install FLA for improved performance
```bash
pip3 uninstall -y causal-conv1d && pip3 install flash-linear-attention==0.3.2
```

4. Run the finetuning example:

```bash
axolotl train examples/qwen3-next/qwen3-next-80b-a3b-qlora.yaml
```

This config uses about 45.62 GiB VRAM.

Let us know how it goes. Happy finetuning! 🚀

### TIPS

- For inference, you can experiment with `temperature: 0.7`, `top_p: 0.8`, `top_k: 20`, and `min_p: 0`.
- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config. See [Multi-GPU](#optimization-guides) section below.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

- [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
- [LoRA Optimizations](https://docs.axolotl.ai/docs/lora_optims.html)

## Related Resources

- [Qwen3-Next Blog](https://qwenlm.github.io/blog/qwen3_next/)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
