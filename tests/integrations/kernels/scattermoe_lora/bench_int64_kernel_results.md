# scatter2scatter INT64_INDICES bench

GPU: **NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition**

Median of 10 iters, 3 warmup. `top_k=8`, dtype=bf16, 128 experts.

`auto_int64` is the wrapper's auto-dispatch verdict from `_needs_int64_indices`. At overflow shapes the int32 path is silently incorrect, so the int32 column shows the chunked workaround's wall-clock from PR #3667 as the apples-to-apples baseline.

| Shape | T | L_scattered | out elems | auto_int64 | int32 ms | int64 ms | chunked ms | int64 vs fast (%) |
|---|---|---|---|---|---|---|---|---|
| small | 8192 | 65536 | 1.34e+08 | False | 2.687 | 2.689 | — | +0.0 |
| medium | 128000 | 1024000 | 2.10e+09 | False | 40.220 | 40.581 | — | +0.9 |
| overflow_524k_s16 | 32768 | 262144 | 4.29e+09 | True | — | 79.572 | 79.985 | -0.5 |

Acceptance: ≤5% regression on the int32 fast path at small/medium shapes, ≤25% regression on the int64 path vs the chunked workaround at overflow shapes.
