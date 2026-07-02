# sonicmoe NVFP4 (W4A4) smoke scripts

Correctness gates for the SM100 grouped gated NVFP4 GEMM described in
`SONICMOE_NVFP4_LORA.md` (Appendix B). Run them in order on a B200/GB200 pod
(SM100/SM110); smokes 1-3 are standalone (no axolotl install), smoke 4 imports
the repo's `axolotl` package by path (torchao required, part of axolotl deps).

## Pod setup

```bash
# torch >= 2.8 with cu12x/cu13 (needs torch.float4_e2m1fn_x2)
pip install "quack-kernels @ git+https://github.com/Dao-AILab/quack.git@f4f54db"
# nvidia-cutlass-dsl is pulled in by quack; verify with check_env
python check_env.py
```

The driver composes against quack internals verified at commit `f4f54db0`
(v0.5.3); install that exact ref, a pip release may drift.

## Run order (each isolates one new thing)

```bash
python smoke_01_upstream_dense.py   # env + upstream ground truth + host-prep parity
python smoke_02_dense_gated.py      # THE composition: GemmGatedSm100(sf_vec_size=16), dense
python smoke_03_varlen_gated.py     # grouped varlen_m + per-expert weights + folded pts
python smoke_04_end_to_end_lora.py  # seam: MoE-LoRA fwd+bwd on a real torchao NVFP4Tensor
```

- smoke_01 runs only upstream quack code paths. A failure here is environment,
  not our driver. Also bit-checks our CPU scale packer against quack's.
- smoke_02 is the composition bet from the design doc (section 4.1): blockscaled
  NVFP4 mainloop + gated epilogue, which no stock quack driver wires together.
  Checks postact, stored preact, and the alpha scalar.
- smoke_03 is the MoE-shaped path: expert-sorted packed rows, uneven seqlens
  (incl. an empty and a 1-token expert), dQaccum-padded SFA, per-expert
  per_tensor_scale folded into SFB, a second forward with a different total_m
  (must not recompile), and the non-gated (down-proj) engine.
- smoke_04 is the integration seam (`backend="fp4_cute"`): route -> quantized
  grouped up GEMM + LoRA delta -> gated activation -> quantized grouped down
  GEMM + LoRA delta -> combine, forward AND backward (LoRA A/B + hidden
  grads), against a pure-torch STE oracle plus a dequant-backend info diff.

All comparisons are against fp32 oracles that dequantize the exact operands the
kernel consumes, so tolerances only cover fp32 accumulation order + bf16 output
rounding (rtol/atol 2e-2).

Known-unproven pieces these scripts exist to validate (design doc, open
questions): fp4 fake-tensor shape handling under varlen, `overlap_accum_sf`
with the gated TileStore, dynamic total_m without recompile.
