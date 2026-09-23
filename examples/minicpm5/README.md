# Finetune MiniCPM5 with Axolotl

[MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) is an open source model from OpenBMB. It uses the Llama architecture, so Axolotl's Llama optimizations (flash attention, sample packing, Cut Cross Entropy, Liger) apply.

This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Run the finetuning example:

    ```bash
    axolotl train examples/minicpm5/lora-2b.yml
    axolotl train examples/minicpm5/fft-2b.yml
    ```

Let us know how it goes. Happy finetuning! 🚀

### Turn terminator

The tokenizer's `eos_token` is `</s>`, but the chat template ends each turn with `<|im_end|>`. Both configs set `eot_tokens` so `<|im_end|>` is trained and the model learns to stop:

```yaml
chat_template: tokenizer_default
eot_tokens:
  - "<|im_end|>"
```

### TIPS

- The chat template adds an empty `<think>` block to assistant turns without `reasoning_content`; it is masked out of the loss. Verify with `axolotl preprocess examples/minicpm5/lora-2b.yml --debug`.
- To train on reasoning traces, put them in `reasoning_content` on the assistant message ([docs](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template)).
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Related Resources

- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
