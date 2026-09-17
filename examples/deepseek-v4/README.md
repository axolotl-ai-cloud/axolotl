# Finetune DeepSeek-V4-Flash with Axolotl

[DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) is a sparse MoE model with NVFP4 experts and head_dim 512 eager-only attention.

This guide trains the MoE experts (LoRA on the 3D expert parameters) on the NVFP4 checkpoint [nvidia/DeepSeek-V4-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4) (~168GB).

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy).

3. Run the finetuning example:

```bash
# 2xB200 (sm100): ~140GB/GPU, ~130 tok/s, train_loss ~1.09
axolotl train examples/deepseek-v4/v4-flash-nvfp4-lora.yaml
```

Let us know how it goes. Happy finetuning! 🚀

### TIPS

- For single GPU, remove FSDP block. It would require >=180GB GPU.
- On cloud, if using volume mount, pointing the HF cache (via `HF_HOME`) at a local disk keeps weight load fast.
- Train the experts only. Do not add attention or module LoRA on a `use_dsv4_kernels` run: it is unsupported and breaks the experts-only FSDP2 invariant (data-independent backward collectives across ranks).
- Keep `attn_implementation: eager` so the dsv4 kernels plugin owns the attention path.
- `sample_packing` attends within the sliding window across packed documents, exactly as plain eager attention would. Drop packing if your data needs strict cross-document isolation.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).

## Optimization Guides

- [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
- [LoRA Optimizations](https://docs.axolotl.ai/docs/lora_optims.html)

## Related Resources

- [DeepSeek-V4-Flash on HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
