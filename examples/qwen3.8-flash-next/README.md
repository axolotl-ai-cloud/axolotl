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
    # QLoRA (1x B300 @ ~207 GiB)
    axolotl train examples/qwen3.8-flash-next/qlora.yaml

    # Vision + text LoRA
    axolotl train examples/qwen3.8-flash-next/vision-lora.yaml
    ```

    ```bash
    # NVFP4 MoE-LoRA
    axolotl train examples/qwen3.8-flash-next/nvfp4-lora.yaml

    # bake the adapter back into a plain NVFP4 checkpoint. --lora-model-dir defaults to
    # output_dir, so pass it explicitly when the adapter lives elsewhere
    axolotl merge-lora examples/qwen3.8-flash-next/nvfp4-lora.yaml \
      --lora-model-dir ./outputs/qwen3.8-flash-next-nvfp4-lora
    ```

Let us know how it goes. Happy finetuning! 🚀

### Checkpoints

| Repo | Size | Notes |
|---|---|---|
| `Qwen/Qwen3.8-Flash-Next` | 360.0 GB | bf16, used by `qlora.yaml` and `vision-lora.yaml` |
| `Qwen/Qwen3.8-Flash-Next-FP8` | 185.5 GB | inference only |
| `Inferact/Qwen3.8-Flash-Next-NVFP4` | 182.7 GB | third-party, used by `nvfp4-lora.yaml` |

The licence is `qwen-community-1.0`, not Apache-2.0.

### MoE Expert Quantization & Expert LoRA

`quantize_moe_experts: true` is required for QLoRA. The experts are fused 3D `nn.Parameter` tensors that bitsandbytes cannot reach on its own, so without it most of the model stays in bf16 and the run OOMs with no warning. To learn about expert quantization, expert LoRA targeting, and related limitations, see the [MoE Expert Quantization](https://docs.axolotl.ai/docs/expert_quantization.html) docs.

Expert LoRA at `lora_r: 16` over 512 experts x 48 layers accounts for almost all of the trainable params, giving a ~11.2 GB adapter. Drop `lora_target_parameters` for a ~30 MB adapter that leaves the experts untrained.

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

- **attn_implementation**: `sdpa` only. The QSA layers overlay the indexer's per-query top-k selection onto the causal mask and hand the combined 4D mask to the attention interface. Flash attention (2/3/4) only ever receives a 2D padding mask, so the selection cannot reach it, and flex attention receives a `BlockMask`, which the indexer cannot read. This is not a head-shape limit: `head_dim=256` is inside FA2's range and is a dedicated FA4 kernel shape. Upstream sets `_supports_flash_attn = False` on every sparse-indexer model (`deepseek_v32`, `glm_moe_dsa`, `qwen4_exp`).
- **lora_target_linear**: Incompatible for this model. It expands to the QSA indexer projection, which receives no gradient, so that adapter would silently never train.
- **LoRA kernels**: Must be explicitly disabled (`lora_*_kernel: false`).
- **Liger**: Not supported. `sdpa_varlen` is also unsupported, and not needed: packing isolation comes from the 4D block-diagonal mask, not from varlen.
- Full finetuning is impractical on a single node: the bf16 weights alone are 329.6 GiB.

### TIPS

- `ple_cpu_offload: true` keeps the 51.2B n-gram table (95.4 GiB) in host RAM, cutting peak VRAM
  from 206.8 to 111.4 GiB at a ~5% throughput cost, with identical loss. The model gathers only
  the rows each token needs. Needs ~100 GB of free system RAM, and 4-bit or 8-bit loading.
- Sample packing is supported and enabled in `qlora.yaml`. Axolotl's "0.0 loss with sdpa in bf16" warning fires on any non-SM90 GPU, so it also fires on Blackwell; a packed 4096 run on B300 trains normally.
- QSA only engages past 2048 tokens, since each query selects 512 blocks of 4. Below that it is a no-op and Axolotl short-circuits the indexer.
- Set `lora_dropout: 0` whenever `lora_target_parameters` is set.
- For inference hyperparameters, please see the model card.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Related Resources

- [Qwen3.8-Flash-Next on HuggingFace](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
