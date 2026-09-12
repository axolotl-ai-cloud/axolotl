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

- Every size shares one `model_type` (`k2_horizon`), so the example config works for the other dense
  checkpoints by changing `base_model`.
- The MoE checkpoints keep their experts as plain `nn.Linear` modules, so `lora_target_linear: true`
  and `load_in_4bit: true` reach them the ordinary way. Prefer an explicit `lora_target_modules` list of
  the attention projections to keep the adapter count manageable: a MoVA layer has `q_proj`, `k_proj`,
  `o_proj` and a `v_experts` list instead of `v_proj`.
- Sample packing is isolated per document off `position_ids`, the same path in-tree transformers
  models use.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Limitations

- Cut Cross Entropy, Liger kernels and the LoRA QKV/O kernels do not cover this architecture; Axolotl
  rejects those flags for `k2_horizon` with the reason.
- The MoE forward loops over the experts a batch actually routes to, so under DeepSpeed ZeRO-3 ranks
  can disagree on which expert parameters to gather. Use FSDP for the MoE checkpoints.
- `output_router_logits` is off in the published configs, so no load-balancing auxiliary loss is added
  unless you enable it via `overrides_of_model_config`.

## Related Resources

- [K2-Horizon Blog](https://ifm.ai/blog/k2/)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
