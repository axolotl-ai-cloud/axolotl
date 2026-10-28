# Finetune Mistral Large 4 with Axolotl

Mistral Large 4 is a 1T parameter (52B active) multimodal MoE model from MistralAI with Multi-head Latent Attention, DeepSeek Sparse Attention and a 1M token context. The NVFP4 checkpoint is available on HuggingFace at [Mistral-Large-4-1T-A52B-NVFP4](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B-NVFP4), and the FP8 checkpoint at [Mistral-Large-4-1T-A52B](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B).

Thanks to the team at MistralAI for giving us early access to prepare for this release.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage

3. Install transformers from main

  ```bash
  uv pip install git+https://github.com/huggingface/transformers.git
  ```

4. Run the example config:

  ```bash
  # text-only LoRA on the NVFP4 checkpoint, 8xB200
  axolotl train examples/mistral-large-4/nvfp4-qlora-text.yml
  ```

NVFP4 needs Blackwell GPUs (e.g. B200, GB200, GB300).

## Reasoning Effort

The chat template supports a `reasoning_effort` variable to control the model's reasoning depth: `"low"`, `"medium"` or `"high"` (default).

Pass it via top-level `chat_template_kwargs`:

```yaml
chat_template_kwargs:
  reasoning_effort: medium
```

## Thinking Support

The chat template renders a `reasoning_content` field on assistant messages as a `[THINK]...[/THINK]` block, and keeps it on every assistant turn, so multi-turn reasoning traces are trained as-is.

```json
{"messages": [
  {"role": "user", "content": "What is 2+2?"},
  {"role": "assistant", "reasoning_content": "2+2=4.", "content": "4"}
]}
```

If your dataset stores the reasoning under a different key, map it with `field_thinking` under your dataset config.

## Tips

- Only `sdpa` and `eager` attention are supported. The sparse attention indexer selects up to 2048 keys per query, so up to `sequence_len: 2048` attention is dense; beyond that it follows the indexer's selection, as in inference. Keep `sequence_len` at or below 2048 when using `sample_packing` for now.
- The indexer runs without gradients and is not a LoRA target.
- The routed experts stay in NVFP4. The FP8 projections (attention `q_proj`/`o_proj`, shared experts, the dense first layer) are dequantized to bf16 at load.
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
