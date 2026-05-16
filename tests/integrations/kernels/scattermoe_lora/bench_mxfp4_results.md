# ScatterMoE LoRA — MXFP4 benchmark

## Routing mode: dense — NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition

- **GPU**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Shape**: E=128, K=2048, N=1024, top_k=8, M=4096, rank=16 (active experts = 128)
- **Iters**: 10 warmup + 50 timed, fwd+bwd per iter
- **HBM peak (datasheet)**: 1792 GB/s

| Config | ms/iter | tokens/s | peak mem (MB) | HBM GB/s | HBM % |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 baseline | 5.21 | 6292624 | 1740.8 | 106.3 | 5.9 |
| Strategy A (selective dequant) | 30.48 | 1075055 | 9033.3 | 18.2 | 1.0 |
| Strategy B (fused MX) | 12.38 | 2645811 | 1889.3 | 12.9 | 0.7 |

## Routing mode: sparse — NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition

- **GPU**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Shape**: E=256, K=2048, N=1024, top_k=8, M=4096, rank=16 (active experts = 10)
- **Iters**: 10 warmup + 50 timed, fwd+bwd per iter
- **HBM peak (datasheet)**: 1792 GB/s

| Config | ms/iter | tokens/s | peak mem (MB) | HBM GB/s | HBM % |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 baseline | 6.43 | 5095584 | 2936.8 | 9.1 | 0.5 |
| Strategy A (selective dequant) | 5.61 | 5841487 | 3011.9 | 10.5 | 0.6 |
| Strategy B (fused MX) | 8.76 | 3742393 | 2925.8 | 3.2 | 0.2 |

## Routing mode: balanced — M sweep — NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition

- **GPU**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Base shape**: E=256, K=2048, N=1024, top_k=8, rank=16
- **M values**: 256, 1024, 4096, 16384
- **Iters**: 10 warmup + 50 timed, fwd+bwd per iter
- **HBM peak (datasheet)**: 1792 GB/s

### Summary (ms/iter, fwd+bwd)

| M | active / E | bf16 ms | Strategy A ms | Strategy B ms | winner (A vs B) |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 256 | 215/256 (0.84) | 2.84 | OOM | 8.23 | B |
| 1024 | 251/256 (0.98) | 3.33 | OOM | 10.71 | B |
| 4096 | 255/256 (1.00) | 6.38 | OOM | 16.25 | B |
| 16384 | 256/256 (1.00) | 23.93 | OOM | 46.33 | B |

### M=256 (active experts = 215 / 256, num_active/E = 0.840)

| Config | ms/iter | tokens/s | peak mem (MB) | HBM GB/s | HBM % |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 baseline | 2.84 | 721826 | 1638.0 | 318.2 | 17.8 |
| Strategy A (selective dequant) | OOM | OOM | OOM | OOM | OOM |
| Strategy B (fused MX) | 8.23 | 248799 | 1882.9 | 29.2 | 1.6 |

### M=1024 (active experts = 251 / 256, num_active/E = 0.980)

| Config | ms/iter | tokens/s | peak mem (MB) | HBM GB/s | HBM % |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 baseline | 3.33 | 2456874 | 1696.2 | 317.0 | 17.7 |
| Strategy A (selective dequant) | OOM | OOM | OOM | OOM | OOM |
| Strategy B (fused MX) | 10.71 | 764813 | 1986.1 | 26.5 | 1.5 |

### M=4096 (active experts = 255 / 256, num_active/E = 0.996)

| Config | ms/iter | tokens/s | peak mem (MB) | HBM GB/s | HBM % |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 baseline | 6.38 | 5134150 | 1912.8 | 170.2 | 9.5 |
| Strategy A (selective dequant) | OOM | OOM | OOM | OOM | OOM |
| Strategy B (fused MX) | 16.25 | 2016613 | 2208.0 | 18.5 | 1.0 |

### M=16384 (active experts = 256 / 256, num_active/E = 1.000)

| Config | ms/iter | tokens/s | peak mem (MB) | HBM GB/s | HBM % |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 baseline | 23.93 | 5476524 | 2779.0 | 47.7 | 2.7 |
| Strategy A (selective dequant) | OOM | OOM | OOM | OOM | OOM |
| Strategy B (fused MX) | 46.33 | 2828866 | 3077.0 | 7.6 | 0.4 |

### Notes

- **Strategy A OOMs at all M** under load-balanced routing at E=256 because
  the torchao MXTensor dequant path materializes several full-shape fp32/int32
  unpack buffers (~12 GiB combined for [256, 1024, 2048] at fp4 → fp32) while
  vLLM colocated on this workstation pins ~88 GB of HBM, leaving only ~14 GB
  free. Extrapolating from the dense E=128 case above (Strategy A peak
  9033 MB at 128 active experts), the E=256 / 256-active dequant peak would be
  ~18 GB — over the available headroom.
- **Active-expert count is essentially E at every sampled M.** Under a
  load-balance-regularized router (per-token N(0,1) noise + N(0,0.5) per-expert
  bias), `E[active] ≈ E · (1 − (1 − top_k/E)^M)`. With E=256 / top_k=8 this
  yields ≥ 215 unique experts even at M=256 and saturates at 256 by M ≈ 16K.
  Balanced routing therefore does **not** generate a low-active regime at
  these token counts — i.e. the A-vs-B crossover does not appear in this
  sweep; B wins by default because A does not fit.
- **B vs bf16:** Strategy B is consistently 1.9–2.9× slower than the bf16
  baseline (similar to the dense E=128 ratio of ~2.4×). HBM utilization for
  both is modest (B 0.4–1.6 %, bf16 2.7–17.8 %), suggesting the kernels are
  compute- or scheduling-bound for these shapes, not bandwidth-bound.
- **Where the A-vs-B crossover lives, by theory:** Strategy A is preferred
  when `num_active / E` is small enough that the dequant cost is offset by
  the cheaper bf16 matmul — the prior `sparse` row (10/256 active, A=5.61 ms
  vs B=8.76 ms) sits in that regime. Strategy B is preferred near
  `num_active / E ≈ 1`, where dequant of all experts dominates. The threshold
  between the two — somewhere in the 10/256 to 215/256 band — is **not
  observable from the balanced-router setting**; eliciting it would need an
  M smaller than 256, a synthetic deliberately-sparse router, or freeing the
  vLLM GPU and rerunning at E=256.
