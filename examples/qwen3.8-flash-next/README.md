# Finetune Qwen3.8-Flash-Next with Axolotl

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is a 176.94B multimodal MoE model (512 routed experts, 10 active, plus a shared expert) built on a hybrid attention stack of 36 Gated DeltaNet layers and 12 Qwen Sparse Attention layers, with a 51.2B-parameter n-gram PLE embedding. It loads as `model_type: qwen4_exp`, a different architecture from Qwen3.8-27B, which is `qwen3_5`.

This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Install `torchvision`. The model loads a video processor even when your dataset is text-only.

    ```bash
    uv pip install torchvision
    ```

4. Run the finetuning example:

    ```bash
    # QLoRA (1x B300 @ ~120 GiB w offload, else ~216 GiB)
    axolotl train examples/qwen3.8-flash-next/qlora.yaml

    # Vision + text QLoRA (1x B300 @ ~103 GiB w offload, else ~197 GiB)
    axolotl train examples/qwen3.8-flash-next/vision-qlora.yaml
    ```

    ```bash
    # NVFP4 MoE-LoRA (1x B300 @ ~200 GiB)
    axolotl train examples/qwen3.8-flash-next/nvfp4-lora.yaml

    # bake the adapter back into a plain NVFP4 checkpoint. --lora-model-dir defaults to
    # output_dir, so pass it explicitly when the adapter lives elsewhere
    axolotl merge-lora examples/qwen3.8-flash-next/nvfp4-lora.yaml \
      --lora-model-dir ./outputs/qwen3.8-flash-next-nvfp4-lora
    ```

Let us know how it goes. Happy finetuning! 🚀

### Gated DeltaNet Linear Attention

36 of the 48 layers are Gated DeltaNet. Its projections are split rather than fused as in Qwen3-Next, so target them via:

```yaml
lora_target_modules:
  - linear_attn.in_proj_qkv
  - linear_attn.in_proj_z
  - linear_attn.in_proj_b
  - linear_attn.in_proj_a
  - linear_attn.out_proj
```

## Limitations

| Feature | Status |
|---|---|
| `attn_implementation` | `sdpa` only. |
| `lora_target_linear` | Incompatible. It expands to the QSA indexer projection incorrectly. |
| LoRA kernels | Unsupported |
| Liger | Unsupported due to incompatible kernels |
| `sdpa_varlen` | Unsupported |
| Full finetuning | Untested. The bf16 weights alone are 329.6 GiB. |

### TIPS

- `ple_cpu_offload: true` saves 95.4 GiB of VRAM by keeping the n-gram table in host RAM for some minor throughput tradeoff.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

- [Optimizations Guide](https://docs.axolotl.ai/docs/optimizations.html)

## Related Resources

- [Qwen3.8-Flash-Next on HuggingFace](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
