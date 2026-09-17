# Finetune MiniCPM5 with Axolotl

[MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) ships as a stock `LlamaForCausalLM` (`model_type: llama`, 42 layers, GQA 16/2, 128K context), so every Llama code path in Axolotl applies: flash attention, sample packing, Cut Cross Entropy and Liger kernels.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) and [Liger](https://docs.axolotl.ai/docs/custom_integrations.html#liger-kernels); both configs enable them.

3. Run a config:

    ```bash
    axolotl train examples/minicpm5/lora-2b.yml
    axolotl train examples/minicpm5/fft-2b.yml
    ```

| Config | Type |
|---|---|
| `lora-2b.yml` | LoRA on all attention + MLP projections, sample packing |
| `fft-2b.yml` | Full fine-tune, sample packing |

## Chat template and turn terminator

The tokenizer's `eos_token` is `</s>`, but the bundled chat template ends every turn with `<|im_end|>` (the model's `generation_config` lists both as stop tokens). Axolotl trains the turn terminator it finds after each assistant span, and by default looks for `eos_token`, so without an override `<|im_end|>` is masked out and the model never learns to stop. Both configs set:

```yaml
chat_template: tokenizer_default
eot_tokens:
  - "<|im_end|>"
```

The template also inserts an empty `<think>\n\n</think>\n\n` block into every assistant turn that has no `reasoning_content`. That block is template scaffolding and stays masked; only the assistant content and `<|im_end|>` are trained. Verify with:

```bash
axolotl preprocess examples/minicpm5/lora-2b.yml --debug
```

To train on reasoning traces, put them in `reasoning_content` on the assistant message (see the [chat_template docs](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template)).

## Related Resources

- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
