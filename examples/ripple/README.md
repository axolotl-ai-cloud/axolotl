# Finetune Ricursive's Ripple with Axolotl

[Ripple-65B Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) is a 65B dense model by
Ricursive Intelligence with up to 1M context. Every fifth layer is global attention without RoPE; the
others are 1024-token sliding-window layers with RoPE. Attention outputs are gated, values carry a
residual from the input embeddings, and the FFN activations are clipped.

The checkpoints ship inference-only remote code, so Axolotl loads a training-ready copy in its place.
`trust_remote_code: true` is required.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Run the finetuning example:

    ```bash
    # QLoRA, single 80GB GPU
    axolotl train examples/ripple/ripple-65b-qlora.yaml

    # bf16 LoRA, FSDP2 across 8x80GB GPUs
    axolotl train examples/ripple/ripple-65b-lora-fsdp.yaml
    ```

### Tips

- `sample_packing: true` is supported with `flash_attention_2` and `sdpa`.
- For the pretrain-only checkpoint, set `base_model: ricursiveAI/Ripple-65B-Base-Preview`.
- The chat template wraps an assistant message's `reasoning_content` in a `<think>` block; map yours
  with `field_thinking`, as the examples do for `thinking`.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Limitations

- Liger kernels and the LoRA QKV/O/MLP kernels do not cover this architecture: they would drop the
  value residual and the FFN clipping.
- These examples use the BF16 checkpoint; the NVFP4 checkpoint is not covered yet.

## Related Resources

- [Ripple Launch Blog](https://www.ricursive.com/ripple)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
