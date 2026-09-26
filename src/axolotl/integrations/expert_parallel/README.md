# Expert Parallelism Integration

Replaces the MoE dispatch/combine path with token-parallel expert dispatch, on one of two backends.

## Backends

| `expert_parallel_backend` | Dispatch | Needs |
|----------------------------|----------|-------|
| `deep_ep`                  | DeepEP's fused NVSHMEM kernels | NVLink (intranode) or RDMA (internode); the `deep_ep` build — see [Installation](#installation) |
| `torch`                     | Plain `all_to_all_single` over the EP process group | Any NCCL or gloo fabric — no extra build, works over PCIe |
| `auto` (default)            | `deep_ep` if importable, else `torch` | — |

Both backends compose with the same local-experts kernel (ScatterMoE, SonicMoE, grouped_mm, eager), the same FSDP mesh, and the same sharding/save-load path — only the token dispatch/combine collective differs. Set explicitly to pin one:

```yaml
expert_parallel_backend: torch  # or deep_ep; omit for auto
```

An explicit `deep_ep` without the `deep_ep` package installed follows `expert_parallel_fallback_on_unsupported` (warn + fall back to no EP, or raise); an explicit `torch` never imports or requires `deep_ep`.

### Checkpointing with the `torch` backend

The `torch` backend's dispatch is training-critical to recompute correctly under activation checkpointing: the router's `topk` must not re-run (a different, nondeterministically-tied result would desync the `all_to_all` shapes each rank sends, which hangs the other ranks) and the token-count split tensors must not be recomputed either. The plugin installs a policy that always saves these under `gradient_checkpointing` (with or without `selective_checkpointing`, including `activation_offloading: hidden_states`) and under FSDP2 `fsdp_config.activation_checkpointing`; reentrant checkpointing and the TRL offloader modes (`activation_offloading: true | legacy | disk`) are rejected at config validation. The dispatched/combined rows themselves are an optional save, on by default (`expert_parallel_save_dispatch: true`) — turn it off to recompute the `all_to_all` instead of holding its output, trading a small amount of network traffic for memory.

## Requirements

Ampere (sm_80, A100) or Hopper (sm_90, H100), all-pairs NVLink — for the `deep_ep` backend only. The `torch` backend runs on any GPU topology with a working NCCL or gloo backend, including PCIe-only setups with no NVLink.

## Installation

The `torch` backend needs nothing beyond axolotl's own dependencies. The rest of this section is the `deep_ep` build.

**Hopper (sm_90, H100), multi-node with NCCL 2.29+ (torch 2.11+) and OFED:**

```bash
git clone --depth 1 https://github.com/deepseek-ai/DeepEP.git
cd DeepEP
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=16 uv pip install --no-build-isolation .
python -c "import deep_ep; print(deep_ep.Buffer)"
```

**Hopper (sm_90, H100), single-node intranode-only (no OFED):**

```bash
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP
git checkout v1.2.1
# Patch 1: setup.py — honor DISABLE_NVSHMEM=1
git apply <<'EOF'
--- a/setup.py
+++ b/setup.py
@@ -19,7 +19,10 @@ if __name__ == '__main__':
     disable_nvshmem = False
     nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
     nvshmem_host_lib = 'libnvshmem_host.so'
-    if nvshmem_dir is None:
+    if int(os.getenv('DISABLE_NVSHMEM', '0')):
+        disable_nvshmem = True
+        nvshmem_dir = None
+    elif nvshmem_dir is None:
         try:
             nvshmem_dir = importlib.util.find_spec("nvidia.nvshmem").submodule_search_locations[0]
             nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)
EOF

# Patch 2: setup.py — drop -rdc=true when NVSHMEM is disabled
git apply <<'EOF'
--- a/setup.py
+++ b/setup.py
@@ -71,7 +71,9 @@ if __name__ == '__main__':
         os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '9.0')

         # CUDA 12 flags
-        nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])
+        nvcc_flags.append('--ptxas-options=--register-usage-level=10')
+        if not disable_nvshmem:
+            nvcc_flags.append('-rdc=true')

     # Disable LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
     if os.environ['TORCH_CUDA_ARCH_LIST'].strip() != '9.0':
EOF

DISABLE_NVSHMEM=1 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=16 \
  uv pip install --no-build-isolation .
python -c "import deep_ep; print(deep_ep.Buffer)"
```

Notes:

- Hopper kernels (FP8, TMA, etc.) are preserved; only intranode dispatch/combine is built — appropriate for single-node H100×{4,8}.
- Patch 1 lets `DISABLE_NVSHMEM=1` skip the NVSHMEM build path, which would otherwise need Mellanox OFED dev headers (`infiniband/mlx5dv.h`).
- Patch 2 drops `-rdc=true` when NVSHMEM is off; otherwise the device-link step has nothing to link against and import fails with `__cudaRegisterLinkedBinary_*` undefined symbol.
- The `v1.2.1` pin is required: it is the last release whose `setup.py` still carries the `disable_nvshmem` path the two patches edit. DeepEP `main` restructured `setup.py` and removed that path ([DeepEP #664](https://github.com/deepseek-ai/DeepEP/pull/664)), so against HEAD the patches fail with `patch failed: setup.py:19`. The pin also sidesteps DeepEP HEAD's `csrc/elastic/` (Engram/EPv2, commit `b306af0`), which needs `ncclGinRequest_t` from NCCL 2.29+.

**Ampere (sm_80, A100, intranode-only)** — needs two small source patches gated on `DISABLE_NVSHMEM=1`:

```bash
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP
git checkout v1.2.1

# Patch 1: setup.py — honor DISABLE_NVSHMEM=1
git apply <<'EOF'
--- a/setup.py
+++ b/setup.py
@@ -19,7 +19,10 @@ if __name__ == '__main__':
     disable_nvshmem = False
     nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
     nvshmem_host_lib = 'libnvshmem_host.so'
-    if nvshmem_dir is None:
+    if int(os.getenv('DISABLE_NVSHMEM', '0')):
+        disable_nvshmem = True
+        nvshmem_dir = None
+    elif nvshmem_dir is None:
         try:
             nvshmem_dir = importlib.util.find_spec("nvidia.nvshmem").submodule_search_locations[0]
             nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)
EOF

# Patch 2: csrc/deep_ep.cpp — gate the three mask_buffer methods
git apply <<'EOF'
--- a/csrc/deep_ep.cpp
+++ b/csrc/deep_ep.cpp
@@ -1823,22 +1823,34 @@ bool is_sm90_compiled() {
 }

 void Buffer::low_latency_update_mask_buffer(int rank_to_mask, bool mask) {
+#ifndef DISABLE_NVSHMEM
     EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
     EP_HOST_ASSERT(rank_to_mask >= 0 and rank_to_mask < num_ranks);
     internode_ll::update_mask_buffer(mask_buffer_ptr, rank_to_mask, mask, at::cuda::getCurrentCUDAStream());
+#else
+    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
+#endif
 }

 void Buffer::low_latency_query_mask_buffer(const torch::Tensor& mask_status) {
+#ifndef DISABLE_NVSHMEM
     EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
     EP_HOST_ASSERT(mask_status.numel() == num_ranks && mask_status.scalar_type() == torch::kInt32);

     internode_ll::query_mask_buffer(
         mask_buffer_ptr, num_ranks, reinterpret_cast<int*>(mask_status.data_ptr()), at::cuda::getCurrentCUDAStream());
+#else
+    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
+#endif
 }

 void Buffer::low_latency_clean_mask_buffer() {
+#ifndef DISABLE_NVSHMEM
     EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
     internode_ll::clean_mask_buffer(mask_buffer_ptr, num_ranks, at::cuda::getCurrentCUDAStream());
+#else
+    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
+#endif
 }

 }  // namespace deep_ep
EOF

DISABLE_NVSHMEM=1 DISABLE_SM90_FEATURES=1 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=16 \
  uv pip install --no-build-isolation .
python -c "import deep_ep; print(deep_ep.Buffer)"
```

## Usage

```yaml
plugins:
  - axolotl.integrations.expert_parallel.ExpertParallelPlugin

expert_parallel_size: 2  # 1 = disabled (default); > 1 = enabled
```

For composition with FSDP at 4+ GPUs, set both `expert_parallel_size` and `dp_shard_size`. The product must equal `world_size`:

```yaml
# 4-GPU example: ep × dp_shard = 2 × 2 = 4
expert_parallel_size: 2
dp_shard_size: 2
fsdp_version: 2
fsdp_config:
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: Qwen3MoeDecoderLayer
  state_dict_type: FULL_STATE_DICT
  reshard_after_forward: true
```

See full example configs at [`examples/expert_parallel/`](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/expert_parallel).

#### Implementation notes

EP composes with the local-experts kernel you've already configured: ScatterMoE, SonicMoE, grouped_mm, or eager — the same four for either backend.

EP composes with FSDP on orthogonal mesh axes: experts are sharded across the `ep` axis, non-expert params across `dp_shard`. The two collectives run on disjoint process groups, so they don't conflict. Layout follows [*Expert Parallelism with FSDP* (tinkerings.dev)](https://tinkerings.dev/posts/expert_parallel.html) — "rows share weights, columns move tokens."

| Your existing config                                | Local kernel under `deep_ep` | Local kernel under `torch` |
|-----------------------------------------------------|-------------------------------|-----------------------------|
| `use_scattermoe: true`                              | ScatterMoE (Triton)           | ScatterMoE (Triton)         |
| `use_sonicmoe: true`                                | SonicMoE (bf16 experts)       | SonicMoE (bf16 experts)     |
| `experts_implementation: grouped_mm` / `batched_mm` | grouped_mm (transformers)     | grouped_mm (transformers)   |
| `experts_implementation: eager`                     | eager Python loop             | eager Python loop           |
| (unset)                                             | grouped_mm (default)          | grouped_mm (default)        |

## Limitations

- Models' modeling code must use `@use_experts_implementation` (canonical 3D `gate_up_proj` / `down_proj`). `ModuleList` as used in Mixtral is not supported.
- `num_experts` must be divisible by `expert_parallel_size`.
- Supported mesh axes: EP, EP × dp_shard, **EP × cp**, EP × cp × dp_shard (experts shard on `ep`,
  the sequence on `cp`, non-expert weights on `dp_shard`). EP × **TP** is not yet supported and
  raises `NotImplementedError`. EP × CP requires the model's attention to be context-parallel-aware
  on the `cp` axis (e.g. GLM-5.2 DSA via the kernels plugin); stock attention uses accelerate CP.
- DeepEP limitation: Low-latency (LL) kernels are inter-node only by design (pure RDMA via IBGDA). Single-node + intranode setups always use the standard kernels and don't benefit from LL.
- FP8 dispatch needs Hopper + DISABLE_SM90_FEATURES=0.

## Troubleshooting

### `CUBLAS_STATUS_INVALID_VALUE` on a basic bf16 GEMM after `import deep_ep`

The system's `libcublas.so.13` is older than what cu130 torch expects. Put the cu13 lib that ships with the torch wheel on `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="$(python -c 'import nvidia.cu13 as m; print(list(m.__path__)[0] + "/lib")'):$LD_LIBRARY_PATH"
```

Unrelated to DeepEP itself, but anyone on cu130 torch hits it on boxes with a system CUDA toolkit older than 13.0.

### `CUDA error 803` (system not yet initialized)

On driver `< 580`, also prepend `/usr/local/cuda-13.0/compat` to `LD_LIBRARY_PATH`. **Do not** add the compat dir on driver `≥ 580` (its `libcuda` is older than the running driver and triggers `CUDA error 803`).
