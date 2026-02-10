# Finetune Z.ai's GLM-4.7-Flash with Axolotl

[GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) is a 30B-A3B MoE model.

This guide shows how to fine-tune it with Axolotl.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage

3. Run the finetuning example:

```bash
axolotl train examples/glm4.7-flash/glm4.7-flash-qlora.yaml
```

This config uses about X GiB VRAM.

Let us know how it goes. Happy finetuning! 🚀

### TIPS

- For inference, the official Z.ai team recommends `top_p: 0.95`, `temperature: 1.0`, and `max_new_tokens: 131072`.
- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
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
