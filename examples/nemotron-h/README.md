# Nemotron-H (nvidia/NVIDIA-Nemotron-3-* and NVIDIA-Nemotron-3.5-*)

Hybrid Mamba2 / Attention / MoE architecture (`model_type: nemotron_h`).

| Model | Total params | Active params | Layers |
|---|---|---|---|
| NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | 120B | ~12B | 88 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 30B | ~3B | — |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | 30B | ~3B | 52 |

## Requirements

```bash
pip install mamba-ssm causal-conv1d   # fast Mamba2 CUDA kernels
```

`sample_packing` and `context_parallel_size > 1` need these kernels: only they
take the `seq_idx` that resets SSM state at packed-sample boundaries, while the
transformers torch fallback drops it silently. Axolotl probes the scan that is
actually in use on the first packed batch and stops rather than mixing state
across samples.

## Architecture notes

- Three block types per layer: **Mamba2** (selective SSM), **Attention** (sparse), **MoE** (mixture-of-experts).
- Only ~12 out of 88 blocks are attention layers (120B variant).
- MLP activation is `relu2` via `mlp_hidden_act` (not the usual `hidden_act`).
- Nemotron-3.5 ships its block pattern as `layers_block_type` (Nemotron-3 used the
  `hybrid_override_pattern` string) and adds multi-token-prediction weights, which
  transformers ignores on load.

## LoRA kernel patches

Attention LoRA kernels support the attention blocks under `NemotronHBlock.mixer`.
Mamba and MoE blocks are skipped by attention patching:

```yaml
lora_qkv_kernel: true
lora_o_kernel: true
lora_mlp_kernel: true    # dense/shared up_proj + down_proj adapters only
```

The MLP option fuses dense/shared-expert LoRA projections and ReLU² activation
forward/backward. Include `up_proj` and `down_proj` in `lora_target_modules` to
adapt these modules. Attention-only examples leave this option disabled.

## MoE expert weights

NemotronH experts store `up_proj` and `down_proj` as 3D `nn.Parameter` tensors
(shape `[num_experts, out_dim, in_dim]`), **not** `nn.Linear` modules — there is no
`gate_proj`. To fine-tune them alongside attention, use `lora_target_parameters`
instead of `lora_target_modules`:

```yaml
lora_target_parameters:
  - up_proj
  - down_proj
```

## Limitations

- **Routed experts**: `lora_mlp_kernel` only patches dense/shared modules. Routed 3D expert tensors retain their existing PEFT or configured MoE backend; enabling dense MLP kernels does not optimize these routed expert tensors.
- **Sample packing / context parallelism**: requires `mamba-ssm` and `causal-conv1d`; see Requirements above.
