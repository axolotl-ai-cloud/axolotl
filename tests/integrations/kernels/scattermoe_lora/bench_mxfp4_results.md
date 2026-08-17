# ScatterMoE LoRA — MXFP4 benchmark

## Routing mode: dense — NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition

- **GPU**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Shape**: E=128, K=2048, N=1024, top_k=8, M=4096, rank=16 (active experts = 128)
- **Iters**: 10 warmup + 50 timed, fwd+bwd per iter
- **HBM peak (datasheet)**: 1792 GB/s

| Config | ms/iter | tokens/s | peak mem (MB) | HBM GB/s | HBM % |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 baseline | 5.25 | 6244998 | 1252.8 | 105.5 | 5.9 |
| Strategy A (selective dequant) | 30.57 | 1071778 | 8557.3 | 18.1 | 1.0 |
| Strategy B (fused MX) | 12.24 | 2677582 | 1425.3 | 13.0 | 0.7 |

## Routing mode: sparse — NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition

- **GPU**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Shape**: E=256, K=2048, N=1024, top_k=8, M=4096, rank=16 (active experts = 10)
- **Iters**: 10 warmup + 50 timed, fwd+bwd per iter
- **HBM peak (datasheet)**: 1792 GB/s

| Config | ms/iter | tokens/s | peak mem (MB) | HBM GB/s | HBM % |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 baseline | 6.55 | 5006027 | 1960.8 | 9.0 | 0.5 |
| Strategy A (selective dequant) | 5.75 | 5695789 | 2059.9 | 10.2 | 0.6 |
| Strategy B (fused MX) | 8.95 | 3661270 | 1997.8 | 3.1 | 0.2 |

## Routing mode: balanced — M sweep — NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition

- **GPU**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Base shape**: E=256, K=2048, N=1024, top_k=8, rank=16
- **M values**: 256, 1024, 4096, 16384
- **Iters**: 10 warmup + 50 timed, fwd+bwd per iter
- **HBM peak (datasheet)**: 1792 GB/s

### Summary (ms/iter, fwd+bwd)

| M | active / E | bf16 ms | Strategy A ms | Strategy B ms | winner (A vs B) |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 256 | 215/256 (0.84) | 2.99 | OOM | 8.24 | B |
| 1024 | 251/256 (0.98) | 3.43 | OOM | 10.74 | B |
| 4096 | 255/256 (1.00) | 6.56 | OOM | 16.50 | B |
| 16384 | 256/256 (1.00) | 24.15 | OOM | 46.56 | B |

### M=256 (active experts = 215 / 256, num_active/E = 0.840)

| Config | ms/iter | tokens/s | peak mem (MB) | HBM GB/s | HBM % |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 baseline | 2.99 | 685596 | 1686.0 | 302.2 | 16.9 |
| Strategy A (selective dequant) | OOM | OOM | OOM | OOM | OOM |
| Strategy B (fused MX) | 8.24 | 248639 | 1954.9 | 29.2 | 1.6 |

### M=1024 (active experts = 251 / 256, num_active/E = 0.980)

| Config | ms/iter | tokens/s | peak mem (MB) | HBM GB/s | HBM % |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 baseline | 3.43 | 2389143 | 1744.2 | 308.3 | 17.2 |
| Strategy A (selective dequant) | OOM | OOM | OOM | OOM | OOM |
| Strategy B (fused MX) | 10.74 | 762567 | 2058.1 | 26.4 | 1.5 |

### M=4096 (active experts = 255 / 256, num_active/E = 0.996)

| Config | ms/iter | tokens/s | peak mem (MB) | HBM GB/s | HBM % |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 baseline | 6.56 | 4994760 | 1960.8 | 165.6 | 9.2 |
| Strategy A (selective dequant) | OOM | OOM | OOM | OOM | OOM |
| Strategy B (fused MX) | 16.50 | 1985884 | 2280.0 | 18.2 | 1.0 |

### M=16384 (active experts = 256 / 256, num_active/E = 1.000)

| Config | ms/iter | tokens/s | peak mem (MB) | HBM GB/s | HBM % |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 baseline | 24.15 | 5427073 | 2827.0 | 47.2 | 2.6 |
| Strategy A (selective dequant) | OOM | OOM | OOM | OOM | OOM |
| Strategy B (fused MX) | 46.56 | 2814943 | 3149.0 | 7.6 | 0.4 |

### Notes

- **Strategy A OOMs at all M** under load-balanced routing at E=256 because
  the torchao MXTensor dequant path materializes several full-shape fp32/int32
  unpack buffers (~12 GiB combined for [256, 1024, 2048] at fp4 → fp32) while
  vLLM colocated on this workstation pins ~88 GB of HBM, leaving only ~14 GB
  free. Extrapolating from the dense E=128 case above (Strategy A peak
  ~8.6 GB at 128 active experts), the E=256 / 256-active dequant peak would
  be ~17 GB — over the available headroom.
- **Active-expert count is essentially E at every sampled M.** Under a
  load-balance-regularized router (per-token N(0,1) noise + N(0,0.5) per-expert
  bias), `E[active] ≈ E · (1 − (1 − top_k/E)^M)`. With E=256 / top_k=8 this
  yields ≥ 215 unique experts even at M=256 and saturates at 256 by M ≈ 16K.
  Balanced routing therefore does **not** generate a low-active regime at
  these token counts — i.e. the A-vs-B crossover does not appear in this
  sweep; B wins by default because A does not fit.
- **B vs bf16:** Strategy B is consistently 1.9–2.9× slower than the bf16
  baseline (similar to the dense E=128 ratio of ~2.3×). HBM utilization for
  both is modest (B 0.4–1.6 %, bf16 2.6–17.2 %), suggesting the kernels are
  compute- or scheduling-bound for these shapes, not bandwidth-bound.
- **Where the A-vs-B crossover lives, by theory:** Strategy A is preferred
  when `num_active / E` is small enough that the dequant cost is offset by
  the cheaper bf16 matmul — the prior `sparse` row (10/256 active, A=5.75 ms
  vs B=8.95 ms) sits in that regime. Strategy B is preferred near
  `num_active / E ≈ 1`, where dequant of all experts dominates. The threshold
  between the two — somewhere in the 10/256 to 215/256 band — is **not
  observable from the balanced-router setting**; eliciting it would need an
  M smaller than 256, a synthetic deliberately-sparse router, or freeing the
  vLLM GPU and rerunning at E=256.
