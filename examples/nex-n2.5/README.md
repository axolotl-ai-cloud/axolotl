# Finetune Nex-N2.5 with Axolotl

[Nex-N2.5-mini](https://huggingface.co/nex-agi/Nex-N2.5-mini) is an open source model from Nex AGI built on the Qwen3.5-35B-A3B architecture (hybrid Gated DeltaNet + attention MoE with vision). Everything in [examples/qwen3.5](../qwen3.5/README.md) applies here; only the chat template differs.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Install FLA for sample packing support with the Gated DeltaNet linear attention layers:

    ```bash
    uv pip uninstall causal-conv1d && uv pip install flash-linear-attention==0.4.1
    ```

4. Run the finetuning example:

    ```bash
    axolotl train examples/nex-n2.5/mini-qlora.yaml
    axolotl train examples/nex-n2.5/mini-vision-lora.yaml
    ```

Let us know how it goes. Happy finetuning! 🚀

### Chat template

Use `chat_template: tokenizer_default`, not `qwen3_5`. Nex's template renders a `<think>` block on every assistant turn (not only the last) and allows system messages mid-conversation, so `qwen3_5` would mask multi-turn data incorrectly. Empty `<think>` blocks are masked out; the assistant content and `<|im_end|>` are trained.

### TIPS

- For LoRA targets on the DeltaNet layers and routed/shared experts, see the [Qwen3.5 README](../qwen3.5/README.md#gated-deltanet-linear-attention).
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).
- For **multimodal** finetuning, set `processor_type: AutoProcessor`, `skip_prepare_dataset: true`, and `remove_unused_columns: false` as shown in `mini-vision-lora.yaml`.

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Related Resources

- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
