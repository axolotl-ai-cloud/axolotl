# Finetune Ministral with Axolotl

Ministral is a family of openweight models from MistralAI found on [HuggingFace](mistralai/Ministral-8B-Instruct-2410). This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Run the finetuning example:

    ```bash
    axolotl train examples/ministral/ministral-small-qlora.yaml
    ```

This config uses about 8.76 GiB VRAM.

Let us know how it goes. Happy finetuning! 🚀

### Tips

- We recommend adding the same/similar SystemPrompt that the model is tuned for. You can find this within the repo's files titled `SYSTEM_PROMPT.txt`.
- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The text dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Limitations

We only support the `mistral-common` tokenizer for Supervised Fine-tuning at the moment and for `type: chat_template` only.

In addition, we do not support overriding tokens yet.

## Related Resources

- [MistralAI Ministral Blog](https://mistral.ai/news/ministraux)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)


## Future Work

- Add parity to Preference Tuning, RL, etc.
- Add parity to other tokenizer configs like overriding tokens.
