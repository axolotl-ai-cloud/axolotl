# Finetune Z.ai's GLM-5.3-Flash with Axolotl

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is a ~321B multimodal MoE model by Z.ai (288 routed experts, 8 active, plus a shared expert). It loads as `model_type: glm5_next`, a different architecture from GLM-4.7-Flash.

This guide shows how to fine-tune it with Axolotl.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Run the finetuning example:

    ```bash
    # QLoRA FSDP2 (8x H100 80GB)
    axolotl train examples/glm53-flash/qlora_fsdp.yaml

    # LoRA FSDP2 (8x H200)
    axolotl train examples/glm53-flash/lora_fsdp.yaml
    ```

The examples load the `-BF16` checkpoint, since bitsandbytes cannot quantize the FP8 release.

### MoE Expert Quantization & Expert LoRA

This model quantizes expert weights on load. To learn about expert quantization, expert LoRA targeting, and related limitations, see the [MoE Expert Quantization](https://docs.axolotl.ai/docs/expert_quantization.html) docs.

## Limitations

| Feature | Status |
|---|---|
| `attn_implementation` | `sdpa` or `eager` only. |
| `sample_packing` | Requires `flash-linear-attention`. |
| `lora_target_linear` | Incompatible. It also targets the vision tower and the no-grad DSA indexer. |
| LoRA kernels | Unsupported |
| Cut Cross Entropy | Unsupported |
| `sdpa_varlen` | Unsupported |
| Full finetuning | Untested. |

### TIPS

- The DSA layers use `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, and the KDA linear-attention layers use `q_proj`, `k_proj`, `v_proj`. Both use `o_proj`.
- Avoid `gate_proj`/`up_proj`/`down_proj` in `lora_target_modules`, since they also match the vision tower.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

- [Optimizations Guide](https://docs.axolotl.ai/docs/optimizations.html)

## Related Resources

- [GLM-5.3-Flash on HuggingFace](https://huggingface.co/zai-org/GLM-5.3-Flash)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
