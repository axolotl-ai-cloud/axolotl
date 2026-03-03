# Finetune Qwen3-Next with Axolotl

[Qwen3-Next](https://huggingface.co/collections/Qwen/qwen3-next-68c25fd6838e585db8eeea9d) represents the next-generation foundation models optimized for extreme context length and large-scale parameter efficiency. The series introduces architectural innovations including Hybrid Attention (Gated DeltaNet + Gated Attention), High-Sparsity MoE with 1:50 activation ratio, and Multi-Token Prediction for enhanced performance and inference acceleration.

This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Install FLA for improved performance
```bash
pip3 uninstall -y causal-conv1d && pip3 install flash-linear-attention==0.4.1
```

4. Run the finetuning example:

```bash
axolotl train examples/qwen3-next/qwen3-next-80b-a3b-qlora.yaml
```

This config uses about ~47 GiB (no target experts) and ~71GiB (target experts) VRAM.

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
