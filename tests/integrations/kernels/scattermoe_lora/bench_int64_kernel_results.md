# scatter2scatter INT64_INDICES bench

GPU: **NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition**

Median of 10 iters, 3 warmup. `top_k=8`, dtype=bf16, 128 experts.

`auto_int64` is the wrapper's auto-dispatch verdict from `_needs_int64_indices`. At overflow shapes the int32 path is silently incorrect (the multiplication wraps mid-buffer), so only the int64 timing is reported.

| Shape | T | L_scattered | out elems | auto_int64 | int32 ms | int64 ms | int64 vs int32 (%) |
|---|---|---|---|---|---|---|---|
| small | 8192 | 65536 | 1.34e+08 | False | 2.699 | 2.704 | +0.2 |
| medium | 128000 | 1024000 | 2.10e+09 | False | 40.126 | 40.790 | +1.7 |
| overflow_524k_s16 | 32768 | 262144 | 4.29e+09 | True | — | 80.105 | — |

Acceptance: ≤5% regression on the int32 fast path at small/medium shapes (the auto-dispatch picks int32 there, so this row characterises the JIT overhead of having an int64 variant available).
