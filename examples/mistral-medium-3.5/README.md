# Finetune Mistral Medium 3.5 with Axolotl

Mistral Medium 3.5 is a 128B parameter dense multimodal model from MistralAI that unifies instruct, reasoning, and agentic capabilities into a single model.
It shares the `mistral3` architecture (dense, YaRN RoPE, 256k context) with Ministral 3 and supports the same `reasoning_effort` toggle as Mistral Small 4.

Thanks to the team at MistralAI for giving us early access to prepare for this release.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. (Text config only) Install [Flash Attention 4](https://docs.axolotl.ai/docs/attention.html#flash-attention-4) on Hopper/Blackwell.

4. Run one of the example configs:

    ```bash
    # text-only
    axolotl train examples/mistral-medium-3.5/qlora-text.yml  # ~83.1 GiB

    # text + vision
    # wget https://huggingface.co/datasets/Nanobit/text-vision-2k-test/resolve/main/African_elephant.jpg
    axolotl train examples/mistral-medium-3.5/qlora-vision.yml  # ~80.3 GiB
    ```

Note: vision training does not currently work with Flash Attention 4.

## Reasoning Effort

The chat template supports a `reasoning_effort` variable to control the model's reasoning depth:

- `"none"` — instruct mode (default)
- `"high"` — reasoning mode with explicit thinking steps

Pass it via `chat_template_kwargs` under your dataset config:

```yaml
datasets:
  - path: your/dataset
    type: chat_template
    chat_template_kwargs:
      reasoning_effort: high
```

## Thinking Support

The chat template supports a `thinking` content type in assistant messages for training on reasoning traces (rendered as `[THINK]...[/THINK]` blocks).

To use thinking datasets, add the `thinking` mapping via `message_property_mappings`:

```yaml
datasets:
  - path: your/thinking-dataset
    type: chat_template
    message_property_mappings:
      role: role
      content: content
      thinking: thinking
    chat_template_kwargs:
      reasoning_effort: high
```

See the [Magistral thinking guide](../magistral/think/README.md) for dataset format details.

## Tips

- For smaller experiments on the same architecture, see [`examples/ministral3`](../ministral3/README.md) (Ministral 3, 3B).
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The text dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).
- The vision model requires multi-modal dataset format as documented [here](https://docs.axolotl.ai/docs/multimodal.html#dataset-format).

## Related Resources

- [MistralAI Mistral Medium 3.5 Blog](https://mistral.ai/news/mistral-medium-3-5)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
