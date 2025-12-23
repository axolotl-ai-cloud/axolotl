# Finetune MoonshotAI's Kimi Linear with Axolotl

[Kimi Linear](https://huggingface.co/collections/moonshotai/kimi-linear-a3b) is a MoE model (48B total, 3B active) by MoonshotAI using a hybrid linear attention architecture to achieve a 1M token context length. It uses Kimi Delta Attention (KDA), a refined version of Gated DeltaNet that reduces KV cache size by up to 75% and boosts decoding throughput by up to 6x for long contexts.

This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

**Note:** Axolotl uses experimental training code for Kimi Linear as their original modeling code is inference-only.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install CCE via [docs](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy)

3. Run the finetuning example:

    ```bash
    axolotl train examples/kimi-linear/kimi-48b-lora.yaml
    ```

This config uses about 98.7GiB VRAM.

Let us know how it goes. Happy finetuning!

### TIPS

- Kimi Linear requires `trust_remote_code: true`.
- You can run a full finetuning by removing the `adapter: lora` and `load_in_8bit: true`.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html)
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template)

## Optimization Guides

See 👉 [docs](https://docs.axolotl.ai/docs/optimizations.html).

## Limitations

This is not yet compatible with MoE kernels from transformers v5.

## Related Resources

- [Kimi Linear Paper](https://huggingface.co/papers/2510.26692)
- [Kimi Linear GitHub](https://github.com/MoonshotAI/Kimi-Linear)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
