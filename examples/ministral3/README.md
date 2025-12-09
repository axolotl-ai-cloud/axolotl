# Finetune Ministral3 with Axolotl

Ministral3 is a family of open-weight models from MistralAI found on [HuggingFace](https://huggingface.co/collections/mistralai/ministral-3). This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

Please see [Thinking](#thinking) and [Vision](#vision) for their respective fine-tuning.

Thanks to the team at MistralAI for giving us early access to prepare for these releases.

Note: This is still experimental given it is based on transformers v5 RC.

## Getting started

1. Install Axolotl from source following the [installation guide](https://docs.axolotl.ai/docs/installation.html#sec-edge-build).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Swap to the Axolotl transformers v5 branch

    ```bash
    cp examples/ministral3/ministral3-3b-qlora.yaml ministral3-3b-qlora.yaml

    git fetch
    git checkout transformers-v5

    # Install packages for transformers v5
    pip install -e .
    ```

4. Run the fine-tuning:

    ```bash
    axolotl train ministral3-3b-qlora.yaml
    ```

Let us know how it goes. Happy finetuning! 🚀


### Tips

- We recommend adding the same/similar SystemPrompt that the model is tuned for. You can find this within the repo's files titled `SYSTEM_PROMPT.txt`.
- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The text dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

### Thinking

Ministral3 2512 model supports thinking capabilities, enabling Chain-of-Thought reasoning with explicit thinking steps.

📚 **[See the Thinking fine-tuning guide →](./think/README.md)**

### Vision

Ministral3 2512 model also supports vision capabilities.

📚 **[See the Vision fine-tuning guide →](./vision/README.md)**

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Limitations

We only support the `mistral-common` tokenizer for Supervised Fine-tuning at the moment and for `type: chat_template` only.

In addition, we do not support overriding tokens yet.

## Related Resources

- [MistralAI Mistral3 Blog](https://mistral.ai/news/mistral-3)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)


## Future Work

- Add parity to Preference Tuning, RL, etc.
- Add parity to other tokenizer configs like overriding tokens.
