# Finetune Mistral Small 4 with Axolotl

Mistral Small 4 is a 119B parameter (6.5B active) multimodal MoE model from MistralAI that unifies instruct, reasoning, and coding capabilities into a single model. It is available on HuggingFace at [Mistral-Small-4-119B-2603](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603).

Thanks to the team at MistralAI for giving us early access to prepare for this release.

## Getting started

Note: Training this model requires weights in BF16 which we will link to later.
Users interested in training can convert / descale the existing FP8 weights.

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage

3. Install transformers from main

  ```bash
  pip install git+https://github.com/huggingface/transformers.git
  ```

4. Run one of the example configs:

  ```bash
  # text-only
  axolotl train examples/mistral4/qlora-text.yml  # no experts ~69 GiB, experts ~93 GiB
  axolotl train examples/mistral4/fft-text.yml

  # text + vision
  # run: wget https://huggingface.co/datasets/Nanobit/text-vision-2k-test/resolve/main/African_elephant.jpg
  axolotl train examples/mistral4/qlora-vision.yml  # no experts ~68 GiB
  axolotl train examples/mistral4/fft-vision.yml
  ```

Note: FFT configs provided as reference. Please adjust hyperparameters as needed.

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

- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The text dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).
- The vision model requires multi-modal dataset format as documented [here](https://docs.axolotl.ai/docs/multimodal.html#dataset-format).

## Related Resources

- [MistralAI Mistral Small 4 Blog](https://mistral.ai/news/mistral-small-4)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
