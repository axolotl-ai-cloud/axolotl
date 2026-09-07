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

All three LoRA Triton kernel patches must be disabled:

```yaml
lora_qkv_kernel: false   # attention lives in NemotronHBlock.mixer, not layer.self_attn
lora_o_kernel: false     # same reason
lora_mlp_kernel: false   # relu2 (mlp_hidden_act) is not supported by lora_mlp_kernel
```

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

- **MoE Triton kernels**: `lora_mlp_kernel` is not supported for NemotronH's MoE expert layers. The expert weights are 3D `nn.Parameter` tensors (not `nn.Linear`), which the Triton kernel does not support. Keep `lora_mlp_kernel: false`.
- **Sample packing / context parallelism**: requires `mamba-ssm` and `causal-conv1d`; see Requirements above.
