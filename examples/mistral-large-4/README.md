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

4. (Optional) For vision only:

    ```bash
    uv pip install 'mistral-common[opencv]>=1.13.0'
    ```

5. Run the example config:

    ```bash
    # text-only, 8xB200
    axolotl train examples/mistral-large-4/nvfp4-lora-text.yml

    # text + vision, 8xB200
    # wget https://huggingface.co/datasets/Nanobit/text-vision-2k-test/resolve/main/African_elephant.jpg
    axolotl train examples/mistral-large-4/nvfp4-lora-vision.yml
    ```

6. Merge the adapter into the checkpoint. The merged model keeps the NVFP4/FP8 formats and is written to `<output_dir>/merged`:

    ```bash
    axolotl merge-lora examples/mistral-large-4/nvfp4-lora-text.yml --lora-model-dir="./outputs/mistral-large-4-nvfp4-lora"
    ```

PS: We're still working on this and will add more configs in coming days.
Due to limited compute, we can only do minor testing, so please let us know if you find any issues!

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
- Only the routed experts (merge-aware NVFP4) and the bf16 `kv_a_proj_with_mqa` / `kv_b_proj` are merge-exact LoRA targets. `q_proj`, `o_proj`, `shared_experts` and the layer-0 dense MLP are FP8 in the checkpoint: LoRA there trains in bf16, but the merge re-rounds them to FP8 and drops most of the update (about 55% in our tests), so the configs leave them out.
- The HF repo does not ship a `config.json` yet, so the configs set `base_model_config` to `examples/mistral-large-4/config` (run from the axolotl root or after `axolotl fetch examples`) and load the tokenizer from the repo. Drop `base_model_config`, `tokenizer_config` and `processor_config` once the repo adds one.
- The example trains merge-aware NVFP4 LoRA (`nvfp4_merge_aware`) so `axolotl merge-lora` keeps the adapter in the NVFP4 checkpoint. See [NVFP4 LoRA](https://docs.axolotl.ai/docs/nvfp4_lora.html).
- `expert_parallel_backend: deep_ep` needs [DeepEP](https://github.com/deepseek-ai/DeepEP) v1.2.1, built per the [expert-parallel README](https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/integrations/expert_parallel/README.md#installation). Without it, set `expert_parallel_backend: torch` (1.3-2.2x slower in our tests).
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
