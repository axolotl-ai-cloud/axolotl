# Finetune Katanemo's Plano-Orchestrator with Axolotl

[Plano-Orchestrator](https://huggingface.co/collections/katanemo/plano-orchestrator) is a family of 4B and 30B-A3B routing and orchestration models designed for multi-agent systems. It analyzes user intent and conversation context to make precise routing decisions, excelling at multi-turn context understanding, multi-intent detection, and context-dependent routing.

This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Run the finetuning example:

    ```bash
    axolotl train examples/plano/plano-4b-qlora.yaml
    ```

This config uses about 5.1 GiB VRAM. Let us know how it goes. Happy finetuning! 🚀

### Orchestration Prompt

Plano-Orchestrator uses a specific orchestration prompt format for routing/agent decisions. Please check the [official model card](https://huggingface.co/katanemo/Plano-Orchestrator-4B) for proper prompt formatting and the `ORCHESTRATION_PROMPT` template.

### Tips

- To use the larger [Plano-Orchestrator-30B-A3B](https://huggingface.co/katanemo/Plano-Orchestrator-30B-A3B) MoE model, simply change `base_model: katanemo/Plano-Orchestrator-30B-A3B` in the config and enable multi-GPU training if needed.
- You can run a full finetuning by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Related Resources

- [Plano GitHub](https://github.com/katanemo/plano)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
