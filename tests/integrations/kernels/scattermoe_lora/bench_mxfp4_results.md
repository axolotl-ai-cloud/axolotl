# ScatterMoE LoRA — MXFP4 benchmark

- **GPU**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Shape**: E=128, K=2048, N=1024, top_k=8, M=4096, rank=16 (active experts = 128)
- **Iters**: 10 warmup + 50 timed, fwd+bwd per iter
- **HBM peak (datasheet)**: 1792 GB/s

| Config | ms/iter | tokens/s | peak mem (MB) | HBM GB/s | HBM % |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 baseline | 5.21 | 6292624 | 1740.8 | 106.3 | 5.9 |
| Strategy A (selective dequant) | 30.48 | 1075055 | 9033.3 | 18.2 | 1.0 |
| Strategy B (fused MX) | 12.38 | 2645811 | 1889.3 | 12.9 | 0.7 |
