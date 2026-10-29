# Finetune Mistral Large 4 with Axolotl

Mistral Large 4 is a 1T parameter (52B active) multimodal MoE model from MistralAI with a 1M token context. The NVFP4 checkpoint is available on HuggingFace at [Mistral-Large-4-1T-A52B-NVFP4](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B-NVFP4).

Thanks to the team at MistralAI for giving us early access to prepare for this release.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Install transformers from main

    ```bash
    uv pip install git+https://github.com/huggingface/transformers.git
    ```

4. Run the example config:

    ```bash
    # text-only, 8xB200
    axolotl train examples/mistral-large-4/nvfp4-lora-text.yml
    ```

## Reasoning Effort

Set the reasoning depth with top-level `chat_template_kwargs`: `"low"`, `"medium"` or `"high"` (default).

```yaml
chat_template_kwargs:
  reasoning_effort: medium
```

## Thinking Support

An assistant message's `reasoning_content` is rendered as a `[THINK]...[/THINK]` block and trained on every turn. Map a different key with `field_thinking` in your dataset config.

```json
{"messages": [
  {"role": "user", "content": "What is 2+2?"},
  {"role": "assistant", "reasoning_content": "2+2=4.", "content": "4"}
]}
```

## Tips

- Only `sdpa` and `eager` attention are supported.
- The example trains merge-aware NVFP4 LoRA (`nvfp4_merge_aware`) so `axolotl merge-lora` keeps the adapter in the NVFP4 checkpoint. See [NVFP4 LoRA](https://docs.axolotl.ai/docs/nvfp4_lora.html).
- `expert_parallel_backend: deep_ep` needs [DeepEP](https://github.com/deepseek-ai/DeepEP) v1.2.1, built per the [expert-parallel README](https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/integrations/expert_parallel/README.md#installation). Without it, set `expert_parallel_backend: torch` (1.3-2.2x slower in our tests).
- On cloud, if using a volume mount, pointing the HF cache (via `HF_HOME`) at a local disk keeps weight load fast.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The text dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

- [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
- [LoRA Optimizations](https://docs.axolotl.ai/docs/lora_optims.html)

## Related Resources

- [Mistral Large 4 NVFP4 on HuggingFace](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B-NVFP4)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
