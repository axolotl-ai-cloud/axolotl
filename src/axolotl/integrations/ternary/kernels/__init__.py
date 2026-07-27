"""Fused kernels for the ternary QAT path.

Everything here is an optional accelerator: each entry point is numerically pinned
to its eager equivalent in `..quant` and must be swappable for it.

- `triton.fake_quant_weight(weight, lambda_, group_size, scale)` — absmean, codes,
  f16-rounded dequant and λ-interpolation in one pass,
- `triton.act_quant(x, lambda_)` — per-token int8 activation fake-quant,
- `triton.pack_codes(codes)` — 4-codes-per-byte snapshot packing,
- `triton.flip_count(packed_prev, packed_now)` — code churn between snapshots,
- `int8_gemm.int8_gemm(...)` — W2A8 tensor-core forward once λ == 1.

Imports stay lazy so a CPU-only or Triton-less environment can still import the
integration.
"""
