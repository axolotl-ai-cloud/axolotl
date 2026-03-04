# Finetune Z.ai's GLM-4.7-Flash with Axolotl

[GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) is a 30B-A3B MoE model by Z.ai.

This guide shows how to fine-tune it with Axolotl.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Run the finetuning example:

```bash
# QLoRA
# - no target experts (1x48GB @ ~24GiB/GPU)
# - target experts (1x48GB @ ~34GiB/GPU)
axolotl train examples/glm47-flash/qlora.yaml

# QLoRA FSDP2 no target experts (2x48GB @ ~29GiB/GPU)
axolotl train examples/glm47-flash/qlora_fsdp.yaml
```

```bash
# LoRA
# - no target experts (1x48GB @ ~35GiB/GPU)
# - target experts (1x48GB @ OOM. Projected ~45-50GiB/GPU)
axolotl train examples/glm47-flash/lora.yaml

# LoRA FSDP2 no target experts (2x48GB @ ~43GiB/GPU)
axolotl train examples/glm47-flash/lora_fsdp.yaml
```

### MoE Expert Quantization & Expert LoRA

This model quantize expert weights on load. To learn about expert quantization, expert LoRA targeting, and related limitations, see the [MoE Expert Quantization](https://docs.axolotl.ai/docs/expert_quantization.html) docs.

## Limitations

- **lora_target_linear**: Incompatible for this model.
- **LoRA kernels**: Incompatible with this model due to non-standard attention projections (DSA). Must be explicitly disabled (`lora_*_kernel: false`).


### TIPS

- For inference, the official Z.ai team recommends these default settings (most tasks):
  - `temperature: 1.0`
  - `top_p: 0.95`
  - `max_new_tokens: 131072`
- You can run a full finetuning by removing `adapter: qlora`, `load_in_4bit: true`, and `quantize_moe_experts: true` from the config. This is heavy, so we have not tested this.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Related Resources

- [GLM-4.7-Flash on HuggingFace](https://huggingface.co/zai-org/GLM-4.7-Flash)
- [GLM-4.7 Blog](https://z.ai/blog/glm-4.7)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
