# Finetune Google's Gemma 4 with Axolotl

[Gemma 4](https://huggingface.co/collections/google/gemma-4) is a family of multimodal models from Google. This guide covers how to train them with Axolotl.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Run the finetuning example:

```bash
# 26B MoE QLoRA (1x80GB @ ~50 GiB)
axolotl train examples/gemma4/26b-a4b-moe-qlora.yaml

# 31B Dense QLoRA (1x80GB @ ~44 GiB)
axolotl train examples/gemma4/31b-qlora.yaml

# 31B Dense QLoRA Flex Attn (1x80GB @ ~26 GiB)
axolotl train examples/gemma4/31b-qlora-flex.yaml
```

### MoE Expert Quantization & Expert LoRA (26B-A4B only)

The 26B-A4B config uses ScatterMoE kernels via the transformers `ExpertsInterface` and quantizes expert weights on load. To learn about expert quantization, expert LoRA targeting, and related limitations, see the [MoE Expert Quantization](https://docs.axolotl.ai/docs/expert_quantization.html) docs.

## Flex Attention

Reduce ~40% VRAM (at the cost of up to half throughput) by setting the below (shown in `examples/gemma4/31b-qlora-flex.yaml`):

```yaml
torch_compile: true
flex_attention: true
```

This works for both the MoE and Dense model.

## Limitations

- **Flash Attention**: FA2 (max head_dim=256) and FA4 (max head_dim=128) cannot support Gemma 4's `global_head_dim=512`. Use SDP or flex attention instead.
- **LoRA kernels**: Not supported due to KV-sharing layers.
- **lora_target_linear**: Incompatible for multimodal models — use `lora_target_modules` with a regex to restrict LoRA to the text backbone.

### TIPS

- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- You can run full finetuning by removing `adapter: qlora`, `load_in_4bit: true`, and `quantize_moe_experts: true` from the config. This is heavy and has not been tested.

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Related Resources

- [Gemma 4 Blog](https://huggingface.co/blog/gemma4)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
