# Finetune IFM's K2-Horizon with Axolotl

[K2-Horizon](https://huggingface.co/collections/IFM/k2-horizon) is a fully open decoder-only family by
IFM (the LLM360 team) with a native 512K context window. The dense sizes are 0.9B, 3.7B, 7B and 32B;
[MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B) and
[375B-A23B](https://huggingface.co/IFM/K2-Horizon-375B-A23B) add sparse MoE layers whose attention also
routes its value projection through a set of experts (MoVA).

Axolotl trains K2-Horizon with the modeling code shipped in the checkpoints, so `trust_remote_code: true`
is required.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Run the finetuning example:

    ```bash
    # Dense 7B, LoRA
    axolotl train examples/k2-horizon/k2-horizon-7b-lora.yaml

    # MoVA-36B-A4B, QLoRA + FSDP2
    axolotl train examples/k2-horizon/k2-horizon-mova-36b-a4b-qlora-fsdp.yaml
    ```

### Tips

- For the MoE checkpoints, prefer an explicit `lora_target_modules` list over `lora_target_linear: true`
  to keep the adapter count manageable; note MoVA attention has `v_experts` instead of `v_proj`.
- Use `lora_target_modules` or `lora_target_linear: true` to target the MoE layers; the experts are
  plain `nn.Linear` modules.
- `sample_packing: true` is supported.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Limitations

- Cut Cross Entropy, Liger kernels and the LoRA QKV/O kernels do not cover this architecture.
- The MoE experts are a per-expert `nn.Linear` loop, not the transformers v5 fused `Experts` layout, so
  the ScatterMoE / SonicMoE kernels do not apply.
- Use FSDP rather than DeepSpeed ZeRO-3 for the MoE checkpoints: the expert loop only runs the experts
  a batch routes to, which can hang ZeRO-3's per-module parameter gathering.
- `output_router_logits` is off in the published configs, so no load-balancing auxiliary loss is added
  unless you enable it via `overrides_of_model_config`.

## Related Resources

- [K2-Horizon Blog](https://ifm.ai/blog/k2/)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
