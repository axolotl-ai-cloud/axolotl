# Finetune Xiaomi's MiMo with Axolotl

[MiMo](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL) is a family of models trained from scratch for reasoning tasks, incorporating **Multiple-Token Prediction (MTP)** as an additional training objective for enhanced performance and faster inference. Pre-trained on ~25T tokens with a three-stage data mixture strategy and optimized reasoning pattern density.

This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Run the finetuning example:

    ```bash
    axolotl train examples/mimo/mimo-7b-qlora.yaml
    ```

This config uses about 17.2 GiB VRAM. Let us know how it goes. Happy finetuning! 🚀

### MiMo-V2.6-Flash

[MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) is a 309B-A15B MoE model. Axolotl loads its text decoder into the native MiMo-V2-Flash modules and dequantizes the FP8/MXFP4 checkpoint to bf16 on load, so plan for ~620 GB of bf16 weights sharded across the FSDP group (e.g. 8x H200). With `cpu_ram_efficient_loading: true`, rank 0 materializes the full ~620 GB model in host RAM before sharding, so that node needs well over 620 GB of CPU memory.

```bash
axolotl train examples/mimo/mimo-v2.6-flash-lora-fsdp.yaml
```

- Text only: the vision, audio and MTP weights are not loaded.
- Requires `flash_attention_4` (or `eager`); SDPA and Flash Attention 2 cannot handle its attention sinks and head dims.
- Saved checkpoints use the bf16 native layout and reload in Axolotl as-is.

### Tips

- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Limitations

**Cut Cross Entropy (CCE)**: Currently not supported. We plan to include CCE support for MiMo in the near future.

**LoRA kernels**: Not supported for MiMo-V2.6.

## Related Resources

- [MiMo Paper](https://arxiv.org/abs/2505.07608)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
