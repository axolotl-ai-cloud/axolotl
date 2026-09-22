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
    axolotl train examples/k2-horizon/k2-horizon-7b-lora.yaml
    ```

### Tips

- For the MoE checkpoints, prefer an explicit `lora_target_modules` list over `lora_target_linear: true`
  to keep the adapter count manageable; note MoVA attention has `v_experts` instead of `v_proj`.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Limitations

- Cut Cross Entropy, Liger kernels and the LoRA QKV/O kernels are not supported.
- Use FSDP rather than DeepSpeed ZeRO-3 for the MoE checkpoints.
- The load-balancing auxiliary loss is off by default; enable `output_router_logits` via
  `overrides_of_model_config` if you want it.

## Related Resources

- [K2-Horizon Blog](https://ifm.ai/blog/k2/)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
