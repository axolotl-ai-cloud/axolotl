# Finetune inclusionAI's Ling 3.0 with Axolotl

[Ling 3.0](https://huggingface.co/collections/inclusionAI/ling-30) is a hybrid linear-attention MoE
family by inclusionAI. Most layers use Kimi Delta Attention (KDA) and every `layer_group_size`-th
one uses multi-latent attention (MLA), so the KV cache stays small at long context; the FFNs are
fine-grained MoE with a shared expert and sigmoid `noaux_tc` routing.

| Model | Total / active | Experts | Layers | KDA : MLA |
|-------|----------------|---------|--------|-----------|
| [Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny) | 7.9B / 1.3B | 128 | 24 | 18 : 6 |
| [Ling-3.0-flash](https://huggingface.co/inclusionAI/Ling-3.0-flash) | 124B / 5.1B | 512 | 42 (+1 MTP) | 35 : 7 |

**Note:** Axolotl trains Ling 3.0 with its own modeling code — the published remote code is
inference-only. See [Limitations](#limitations) for what differs.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Run the finetuning example:

    ```bash
    axolotl train examples/ling3/ling-3.0-tiny-lora.yaml
    ```

   Ling-3.0-flash needs multiple GPUs; its config quantizes both the linear layers and the expert
   tensors:

    ```bash
    axolotl train examples/ling3/ling-3.0-flash-qlora.yaml
    ```

Measured on a single H100 80GB at `sequence_len: 4096` and `micro_batch_size: 1`:

| Config | Peak VRAM (torch reserved) | Peak VRAM (device) | Speed |
|--------|---------------------------|--------------------|-------|
| `ling-3.0-tiny-lora.yaml` | 25.0 GiB | 29.0 GiB | 12.3 s/step |
| `ling-3.0-flash-qlora.yaml` | does not fit on 80 GiB | — | — |

Ling-3.0-flash as configured needs more than one 80 GiB card. The 4-bit weights plus the expert
adapters settle at 75.9 GiB, and each MoE layer then builds a bf16 LoRA delta for its fused
`gate_up_proj` — `512 x 2560 x 1536` in bf16, 3.75 GiB — so the first forward runs out of memory.
Dropping `lora_target_parameters` to train the attention projections alone fits in 71.3 GiB reserved
(73.3 GiB device) at 45 s/step.

### Tips

- Ling 3.0 requires `trust_remote_code: true`; Axolotl serves the model classes from its own copy.
- The two layer types expose different projections. `q_proj`/`k_proj`/`v_proj`/`o_proj` reach the
  KDA layers, `kv_a_proj_with_mqa`/`kv_b_proj`/`dense` reach the MLA layers. How MLA projects the
  query depends on the checkpoint: Ling-3.0-tiny sets `q_lora_rank`, so it has `q_a_proj`/`q_b_proj`,
  while Ling-3.0-flash leaves it null and reuses the plain `q_proj` name.
- Experts are stored as fused 3D tensors, so LoRA cannot target them as linear layers. Reach them
  with `lora_target_parameters: [gate_up_proj, down_proj]`, and add `quantize_moe_experts: true` to
  keep them off the VRAM budget. Leave `lora_target_parameters` out to train the attention
  projections only — the flash config sets both, the tiny config trains attention alone.
- Sample packing is isolated per document off `position_ids`: Axolotl drops the packed attention
  mask, so the MLA layers get their block-diagonal mask from position ids and the KDA layers get
  the matching `cu_seqlens`.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Limitations

- The remote code published with the checkpoints is inference-only, so Axolotl loads the in-tree
  copy instead: the published version hardcodes the eager attention interface (so
  `flash_attention_2` would train without a causal mask), builds masks with the pre-v5 helpers, and
  discards the padding mask before the linear-attention kernels.
- Ling-3.0-flash ships one multi-token-prediction layer. It is not built for training, so its
  weights are ignored on load and absent from a full-parameter save. LoRA runs are unaffected.
- Ling-3.0-flash caps the SwiGLU activation on its last few layers (`expert_swiglu_limit_list`,
  `share_expert_swiglu_limit_list`). The published modeling code never reads those keys, so neither
  does Axolotl — training matches the reference implementation, but check the serving stack before
  relying on it. Axolotl logs a warning when a checkpoint sets them.
- `context_parallel_size > 1` is rejected: ring attention shards the sequence across ranks and
  nothing hands a KDA layer's recurrent state to the next rank, so each would restart the recurrence
  from zero.
- Cut Cross Entropy, Liger kernels and the LoRA MLP/QKV kernels do not cover this architecture.
- `kda_safe_gate` is applied outside the kernel: `fla-core` 0.4.1 has no fused equivalent, so
  Axolotl evaluates `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))` in plain PyTorch.
- `fla-core` 0.4.1's KDA backward kernel faults with an illegal memory access when Triton compiles
  it at 3.6 or 3.7 — the version `torch` 2.12.1 and 2.13.0 pin. Only the backward is affected, so a
  run loads and evaluates cleanly and then dies inside step 1; `triton==3.5.1` compiles it correctly.

## Related Resources

- [Ling 3.0 Blog](https://huggingface.co/collections/inclusionAI/ling-30)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
