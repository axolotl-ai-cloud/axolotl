# Provenance of the vendored Marlin W4A16 CUDA sources

These files implement the NVFP4 weight-only (W4A16) Marlin MoE GEMM and the
`gptq_marlin_repack` weight-prep op. They are **extracted from
[vLLM](https://github.com/vllm-project/vllm)** (`csrc/`), which in turn builds on the
**Marlin** kernel by Elias Frantar (`Copyright (C) Marlin.2024 Elias Frantar`, modified by
Neural Magic). Licensing follows the upstream sources: **Apache-2.0** (vLLM) and the Marlin
license headers retained verbatim in the individual files.

Vendored here (rather than importing vLLM) so this optional sm120 backend adds **no vLLM
runtime dependency** — consistent with the cutlass-dsl / DeepGEMM optional-backend pattern.

## What was changed vs upstream

Only the two host-wrapper translation units were edited; the kernels and headers are otherwise
verbatim:

- `libtorch_stable/moe/marlin_moe_wna16/ops_standalone.cu` — the `moe_wna16_marlin_gemm` host
  wrapper, ported from vLLM's `torch::stable::Tensor` ABI to a regular `torch::Tensor` ABI, and
  the `STABLE_TORCH_LIBRARY_IMPL` registration replaced by a `PYBIND11_MODULE`.
- `libtorch_stable/moe/marlin_moe_wna16/repack_standalone.cu` — the `gptq_marlin_repack` host
  wrapper, same `torch::stable` -> `torch::Tensor` port; compiled in the default `marlin`
  namespace (the GEMM TU uses `marlin_moe_wna16`).

Everything else (`marlin_template.h`, `marlin.cuh`, `marlin_mma.h`, `marlin_dtypes.cuh`,
`dequant.h`, `kernel.h`, `kernel_selector.h`, `launcher_body.inc`, `core/scalar_type.hpp`,
`sm80_kernel_bfloat16_fe2m1f_bfloat16.cu`) is copied verbatim from vLLM.

## Equivalence

The ported `moe_wna16_marlin_gemm` and `gptq_marlin_repack` were validated **bit-exact** against
the same vLLM ops (identical inputs -> identical outputs). The Python prep (`../prep.py`) copies
vLLM's pure-torch scale helpers verbatim, so the full NVFP4 -> Marlin weight prep is bit-identical
to vLLM's `prepare_nvfp4_moe_layer_for_marlin`.
