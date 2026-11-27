# Expert Parallelism Integration

Replaces the MoE dispatch/combine path with token-parallel expert dispatch, on one of three backends.

## Backends

| `expert_parallel_backend` | Dispatch | Needs |
|----------------------------|----------|-------|
| `deep_ep`                  | DeepEP's fused NVSHMEM kernels | NVLink (intranode) or RDMA (internode); the `deep_ep` build — see [Installation](#installation) |
| `torch`                     | Plain `all_to_all_single` over the EP process group | Any NCCL or gloo fabric — no extra build, works over PCIe |
| `optmoe`               | The `optmoe` package over NVLink symmetric memory: `fused` mode runs dispatch, SwiGLU experts and combine in one persistent kernel per direction; `dispatch` mode runs OPTMoE's peer dispatch/combine around the configured local-experts kernel | One NVLink node and the `optmoe` package; `fused` needs Hopper or Blackwell, `dispatch` SM80+ — see [`optmoe` backend](#optmoe-backend) |
| `auto` (default)            | `deep_ep` if importable, else `torch` (never `optmoe`) | — |

`deep_ep`, `torch` and `optmoe` in `dispatch` mode compose with the same local-experts kernel (ScatterMoE, SonicMoE, grouped_mm, eager), the same FSDP mesh, and the same sharding/save-load path — only the token dispatch/combine collective differs. Set explicitly to pin one:

```yaml
expert_parallel_backend: torch  # or deep_ep; omit for auto
```

An explicit `deep_ep` without the `deep_ep` package installed follows `expert_parallel_fallback_on_unsupported` (warn + fall back to no EP, or raise); an explicit `torch` never imports or requires `deep_ep`.

### Checkpointing with the `torch` backend

Recompute must replay the forward's routing: a re-run router `topk` can break ties differently, desyncing the `all_to_all` shapes each rank sends and hanging the other ranks. The plugin installs a policy that always saves the router `topk` and the token-count splits:

- It applies under `activation_checkpointing` with either implementation: `hf` (with or without `selective`, including `offload.hidden_states` or `offload.all`) and FSDP2's `implementation: fsdp`.
- The `topk` save is scoped to the MoE block that owns each Experts module; other `topk`s in the layer (e.g. GLM-DSA's attention indexer) recompute as usual.
- Config validation rejects reentrant checkpointing.

The dispatched/combined rows are recomputed by default. `expert_parallel_save_dispatch: true` saves them instead, removing the backward `all_to_all`s at the cost of memory (24 GiB on Qwen3-30B-A3B at 32k packed on 2 GPUs). Recompute cost 1.6 s/step on a PCIe-only pair and broke even on an NVLink pair, so enable the save only on bandwidth-starved links with memory to spare. With the default, EP=2 peaks lower than FSDP2 alone (40.7 vs 48.5 GiB live on a Qwen3-30B-A3B LoRA step at 32k, 2xH100), because each rank only materialises its own experts' rows.

### Chunked dispatch with the `torch` backend

`expert_parallel_dispatch_chunks: N` (default `1`, `torch` backend only) splits each MoE layer's tokens into `N` chunks and pipelines them: chunk `i+1`'s dispatch `all_to_all` is in flight while chunk `i`'s expert GEMMs run, and chunk `i`'s combine is consumed only after chunk `i+1`'s GEMMs are enqueued, so the collectives can overlap compute. All chunks' split counts come from one count exchange and one device-to-host copy per layer. Only the forward overlaps: autograd runs each chunk's backward chain back to back. An explicit `expert_parallel_backend: deep_ep` with `N > 1` is rejected at config validation.

Overlap is not a speedup guarantee: each chunk adds its own collectives and kernel launches, and that per-chunk overhead can exceed what the overlap hides. On 2x PCIe GPUs (16k tokens, hidden 4096), `N: 2` was 12-18% faster when expert compute dominated and ~5% slower when communication-bound; `N: 4` helped only in the most compute-heavy case, and `N: 8` was never faster and up to 3x slower. Benchmark your model's forward before raising `N`, and start from `2`.

### `optmoe` backend

`optmoe` (the [OPTMoE](https://github.com/axolotl-ai-cloud/optmoe) package) moves tokens over NVLink symmetric memory in one of two modes, chosen by `expert_parallel_optmoe_mode`:

| `expert_parallel_optmoe_mode` | Dispatch / combine | Local experts kernel |
|--------------------------------|--------------------|----------------------|
| `fused`   | one persistent kernel per direction fusing dispatch, SwiGLU experts and combine (full fine-tune, or a LoRA adapter on the experts through the `*_lora` precisions) | not used: `use_sonicmoe`, `use_scattermoe` and `experts_implementation` are ignored (config validation warns) and apply only if the backend falls back to `torch` |
| `dispatch` | OPTMoE's peer dispatch and combine (`optmoe::peer_dispatch` / `peer_combine`): every rank pulls the rows routed to its experts straight out of the peers' buffers and pushes the expert outputs back into per-token slots | whichever the config names, exactly as under `deep_ep` / `torch`, so DoRA and other adapters the fused LoRA precisions do not cover, quantized experts, biases, non-SiLU gates, clamped SwiGLU and shared experts all work |
| `auto` (default) | `fused` when the config and model pass the fused checks below, otherwise `dispatch` when the hardware supports it, otherwise the `expert_parallel_fallback_on_unsupported` behaviour | |

```yaml
plugins:
  - axolotl.integrations.expert_parallel.ExpertParallelPlugin
expert_parallel_size: 8
expert_parallel_backend: optmoe
expert_parallel_optmoe_mode: auto       # fused | dispatch
expert_parallel_optmoe_precision: auto  # fused mode: bf16 | mxfp8 | fp8 | fp8_bwd (| *_lora)
expert_parallel_optmoe_workspace_bound: auto  # fused mode, SM100: auto | max
expert_parallel_optmoe_stored_context: auto  # fused mode, SM100: auto | recompute | always | off
expert_parallel_optmoe_nvfp4_merged_cache: layer  # nvfp4_lora: layer | all
expert_parallel_optmoe_nvfp4_headroom_log2: 0  # nvfp4_lora: 0 to 4, merged-weight headroom as a power of two
expert_parallel_optmoe_fused_lora: true  # LoRA on BF16 experts in the fused mode; false: dispatch
expert_parallel_optmoe_lora_merged_cache: layer  # bf16_lora / mxfp8_lora / fp8_lora: layer | all
fsdp_version: 2
fsdp_config:
  state_dict_type: FULL_STATE_DICT
```

Install the package with `pip install axolotl[optmoe]` (or `pip install optmoe`), then its compiled kernel extension with `python -m optmoe.install_kernels`, which fetches the wheel for this Python, torch and CUDA version or builds it from source. The `dispatch` mode needs only the package itself (Triton kernels and torch symmetric memory, no kernel extension).

#### Fused mode

| Precision | SM90 (Hopper) | SM100 (Blackwell) |
|-----------|---------------|-------------------|
| `auto`    | `bf16`        | `mxfp8`           |
| `bf16`    | yes           | yes               |
| `mxfp8`   | no            | yes               |
| `fp8` (blockwise E4M3 forward and backward) | yes | no |
| `fp8_bwd` (BF16 forward, blockwise FP8 backward) | not implemented yet | no |
| `nvfp4_lora` (frozen packed NVFP4 experts, PEFT expert LoRA) | no | yes (`auto` picks it with `adapter: lora` on a modelopt NVFP4 base) |
| `bf16_lora` (frozen BF16 experts, PEFT expert LoRA, `bf16` kernels) | yes (`bf16` with an adapter) | yes (`bf16` with an adapter) |
| `mxfp8_lora` (the same on the `mxfp8` kernels) | no | yes (`auto` / `mxfp8` with an adapter) |
| `fp8_lora` (the same on the blockwise `fp8` kernels) | yes (`fp8` with an adapter) | no |

Master weights keep their own dtype (BF16 or FP32) and weight gradients come back in it. SM120 (RTX PRO 6000, RTX 50-series) and older GPUs are unsupported and fall back.

`nvfp4_lora` trains an adapter on packed NVFP4 experts. The kernels plugin loads the modelopt NVFP4 experts and its `ParamWrapper` fastpath hands the `lora_target_parameters` adapter to the experts module, so the config needs `use_sonicmoe: true`, `nvfp4_merge_aware: false` (merge-aware training stays the SonicMoE path), `lora_dropout: 0` and both `experts.gate_up_proj` and `experts.down_proj` in `lora_target_parameters` ([examples/expert_parallel/qwen3_30ba3b_nvfp4_lora_optmoe.yaml](../../../../examples/expert_parallel/qwen3_30ba3b_nvfp4_lora_optmoe.yaml)). The packed experts stay frozen plain tensors outside FSDP on every rank (`fully_shard_experts` passes them as `ignored_params`; dp_shard peers hold the same EP slice); a `optmoe.nvfp4_lora.Nvfp4LoraExperts` is built over each layer's bytes in place at train begin (so a `torch_compile` forward creates none; a layer not on the GPU yet is built at its first forward), and a global optimizer post-step hook invalidates the requant cache so the merged weights are rebuilt from the updated adapter (gradient accumulation reuses them; FSDP2-sharded adapters are read unsharded inside the forward and the backward). `expert_parallel_optmoe_nvfp4_merged_cache: layer` (default) requantizes each layer's forms right before its forward and its backward and keeps the last two layers resident; `all` holds every layer's four forms across the step. The transposed backward forms are quantized from the merged tile (`OPTMOE_NVFP4_REQUANT=cuda_fused`), so no transposed copy of the base is stored. The checkpoint's `input_scale` tensors (read once from its safetensors) set the static activation scales. Adapter checkpoints take the usual EP / FSDP2 gather paths; `merge-lora` is the SonicMoE merge-aware path's job.

`bf16_lora`, `mxfp8_lora` and `fp8_lora` train the same PEFT expert adapter on a BF16 base with the full fine-tune kernels of `bf16`, `mxfp8` and `fp8`: an expert LoRA config with `auto`, `bf16`, `mxfp8` or `fp8` resolves to them (`auto` is `mxfp8_lora` on SM100, `bf16_lora` on SM90). The config needs `use_sonicmoe: true` (its ParamWrapper fast path hands the factors over), `lora_dropout: 0`, no DoRA and both expert projections in `lora_target_parameters`; `expert_parallel_optmoe_fused_lora: false` keeps such adapters on the dispatch mode. The frozen base stays in the experts module and is FSDP2-sharded with the layer; `fused_lora.py` keeps one `optmoe.lora.LoraExperts` per layer and the shared `MergedWeightCache`, which a global optimizer post-step hook invalidates, so each layer's merged weights `bf16(W + scaling * B @ A)` are rebuilt after every step from the unsharded base FSDP2 gathered for that forward or backward (`expert_parallel_optmoe_lora_merged_cache: layer` merges per use and keeps the last two layers, `all` merges once per step). The kernels run on the merged weights; the adapter gradients come from rank-r grouped GEMMs on the backward's pools (blockwise FP8: projected from its FP8 dW) and the base gets none. The fused forward is the `optmoe::lora_forward` custom op, which selective checkpointing saves like `persistent_forward`.

**Fallback matrix.** The fused checks below decide between the modes; the dispatch mode has its own, much shorter list. Under `expert_parallel_fallback_on_unsupported: true` (the default) a `optmoe` that cannot run in the requested mode logs the reason and degrades; with `false` it raises with the reason.

| Condition | Checked | `fused` (forced) | `auto` | `dispatch` (forced) |
|-----------|---------|------------------|--------|---------------------|
| `optmoe` package not installed | config load | `torch` | `torch` | `torch` |
| an adapter is configured and the config does not qualify for a fused LoRA precision (see above; `expert_parallel_optmoe_fused_lora: false` for BF16 bases), or a `*_lora` precision without `adapter: lora` | config load | `torch` | `dispatch` | `dispatch` |
| an adapter is configured and the experts do not match the route: not packed `NVFP4Tensor`s on SM100 for `nvfp4_lora`, packed NVFP4 for the BF16 LoRA precisions | after model build | `torch` | `dispatch` | `dispatch` |
| quantized loading (`load_in_4bit`, `load_in_8bit`, `gptq`, `model_quantization_config`) | config load | `torch` | `dispatch` | `dispatch` |
| compute capability other than SM90/SM100 (SM80, SM120, ...) | after model build | `torch` | `dispatch` | `dispatch` |
| `hidden_size % 256 != 0` or expert intermediate size `% 128 != 0` | after model build | `torch` | `dispatch` | `dispatch` |
| expert biases, a non-SiLU gate, a custom `_apply_gate` (e.g. `alpha`), or a clamped SwiGLU (`limit`/`swiglu_limit`; the fused backward has no clamp yet) | after model build | `torch` | `dispatch` | `dispatch` |
| experts not in HF `gate_up_proj [E, 2I, H]` / `down_proj [E, H, I]` layout (non-gated, transposed, interleaved, post-expert norm), or MoE layers with different shapes | after model build | `torch` | `dispatch` | `dispatch` |
| the precision does not match the architecture (table above) | after model build | `torch` | `dispatch` | `dispatch` |
| no CUDA, or compute capability below SM80 | after model build | `torch` | `torch` | `torch` |
| the EP group spans more than one host (symmetric memory is node-local) | runtime build | error | error | error |

"`torch`" means `expert_parallel_backend: torch` with whichever local kernel the config names (so a forced `fused` with an adapter the fused LoRA precisions do not cover and `use_sonicmoe: true` lands on `torch` + SonicMoE; the default `auto` lands on `dispatch` + SonicMoE). There is no `model_type` gate: any HF MoE whose experts match the layout above is accepted by the fused mode, and any HF MoE the local kernel accepts by the dispatch mode.

#### Dispatch mode

Every rank publishes its tokens, routing and weights into its own symmetric buffers; after a barrier each rank pulls exactly the rows routed to its experts (`optmoe::pack_peer_rows` over NVLink) into a fixed layout of `expert_parallel_size * capacity` rows (row `s * capacity + t` is token `t` of rank `s`, or a zero row with all `-1` ids), runs the configured local kernel on them through the same `_run_local` path as `deep_ep` / `torch`, and pushes the expert outputs back into per-token slots of the source ranks, which sum them in FP32. Backward is the same two transfers with the payloads swapped, so routing-weight gradients flow like the torch backend's. Rows travel as BF16; the local kernel sets the compute precision (`expert_parallel_optmoe_precision` does not apply).

The layout is a pure function of the routing: there is no host-side count, no sort and no `.item()`, so the layer stays `torch.compile`-clean; the price is that the local kernel sees the fixed `expert_parallel_size * capacity` rows (sentinel rows are zero-compute in grouped_mm, SonicMoE and ScatterMoE). The capacity is sized like the fused runtime (`sequence_len * max(micro_batch_size, eval_batch_size) / context_parallel_size`). Selective checkpointing replays the transfers on recompute by default; `expert_parallel_save_dispatch: true` keeps the dispatch/combine outputs instead, as it keeps the torch all-to-all outputs. `expert_parallel_optmoe_compact: true` hands the local kernel only the routed rows at the cost of one device-to-host count per layer (`optmoe::peer_count_to_host`, always saved under checkpointing like the torch backend's split counts; it breaks `fullgraph` compilation the same way). Adding `expert_parallel_optmoe_bounded: true` removes that stall: each layer receives into a bound from its high-water routed count (filled to SonicMoE's own row bucket) and the count is read only after the local kernel was issued; outputs and gradients stay bitwise those of the compact rows, and a layer whose count exceeds its bound re-runs exactly (8xH100 checkpointed layer 19.2 -> 18.7 ms, 8xB200 12.5 -> 11.9 ms; `packages/optmoe/docs/compose-design.md` 5.7). `expert_parallel_dispatch_chunks` does not apply. `NCCL_MAX_CTAS` is exported (and checked) only while the fused mode is still possible for the config; a forced `dispatch` mode, an adapter or quantized loading skip it.

**Runtime.** The plugin builds one runtime per process in `post_model_load` on the resolved EP group, sized for `sequence_len * max(micro_batch_size, eval_batch_size) / context_parallel_size` tokens per rank (rounded up to the kernel's token alignment), with one symmetric buffer shared by every MoE layer. At train begin, after the model is wrapped, a callback runs `OptMoERuntime.prepare` (one forward and backward on synthetic inputs with the configured tokens per rank, weight dtype and LoRA rank), so every rank compiles and loads the fused kernels before any rank starts one; a kernel left waiting in the cross-rank barrier for a peer that is still compiling traps after 60 s. It is destroyed in `post_train_unload`. Under FSDP2 the expert weights are read with `.to_local()` and keep the usual expert grad scaling. The forward and backward are `torch.library` custom ops (`optmoe::persistent_forward` / `persistent_backward`) with fake implementations and no host syncs in the seam, so the EP layer compiles under `torch_compile: true` without graph breaks (the four-GPU compile parity test, dp_shard 2 x EP 2 in BF16, passes on B200 against a compiled torch reference); nothing from the forward is retained (backward restages the inputs and recomputes). Selective checkpointing always saves the fused op's output and the router `topk`, and reentrant checkpointing is rejected as for `torch`.

**SM reservation.** The persistent kernels spin on all but 16 SMs. FSDP's NCCL collectives must fit in the reserved SMs or they deadlock against the spinning grid, so the plugin exports `NCCL_MAX_CTAS = 16 // 2 = 8` (`16 // 3 = 5` under HSDP, `dp_replicate_size > 1`) through the plugin `get_distributed_env` hook, before any process group exists. A user-set `NCCL_MAX_CTAS` wins but must not exceed that value; `pre_model_load` raises otherwise. Launchers that create the process group before Axolotl loads its config must export the variable themselves.

`expert_parallel_dispatch_chunks` and `expert_parallel_save_dispatch` do not apply (communication overlaps inside the kernels). The native fused backward currently needs a single host, `intermediate % 256 == 0` and BF16 (or blockwise FP8 on SM90); otherwise the runtime uses its NCCL reference backward behind the same op.

## Requirements

Ampere (sm_80, A100), Hopper (sm_90, H100) or Blackwell (sm_100, B200), all-pairs NVLink — for the `deep_ep` backend only. The `torch` backend runs on any GPU topology with a working NCCL or gloo backend, including PCIe-only setups with no NVLink.

## Installation

The `torch` backend needs nothing beyond axolotl's own dependencies. The rest of this section is the `deep_ep` build.

**Hopper (sm_90, H100), multi-node with NCCL 2.29+ (torch 2.11+) and OFED:**

```bash
git clone --depth 1 https://github.com/deepseek-ai/DeepEP.git
cd DeepEP
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=16 uv pip install --no-build-isolation .
python -c "import deep_ep; print(deep_ep.Buffer)"
```

**Hopper (sm_90, H100) or Blackwell (sm_100, B200), node-local EP on CUDA 13.0:**

DeepEP `main` requires CUDA 13.1; `v1.2.1` builds against the CUDA 13.0 toolkit
that the Axolotl images ship once CUDA 13's relocated `cuda/std/*` headers (`include/cccl`) are on the
include path. It links the `nvidia-nvshmem-cu13` wheel that torch already depends on, so it needs no
source patches, system NVSHMEM or Mellanox OFED; beyond CUDA, the only system headers it compiles
against are rdma-core's `infiniband/mlx5dv.h` (`libibverbs-dev`), which the Axolotl images include.
No GPU is needed for the build; the wheel can be built once and installed on every node.
Set `TORCH_CUDA_ARCH_LIST=9.0` for H100 or `10.0` for B200:

```bash
git clone --depth 1 --branch v1.2.1 https://github.com/deepseek-ai/DeepEP.git
cd DeepEP
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=16 \
  CFLAGS=-I/usr/local/cuda/include/cccl CXXFLAGS=-I/usr/local/cuda/include/cccl \
  NVCC_PREPEND_FLAGS=-I/usr/local/cuda/include/cccl
python setup.py bdist_wheel -d dist     # uv pip install . fails in the build backend on this tag
uv pip install --no-deps dist/deep_ep-*.whl
python -c "import deep_ep; print(deep_ep.Buffer)"
```

Notes:

- `TORCH_CUDA_ARCH_LIST=10.0` makes `setup.py` set `DISABLE_AGGRESSIVE_PTX_INSTRS=1` on its own; the sm_90-only PTX tricks are not built. Hopper features (FP8 dispatch, TMA) stay on.
- Axolotl's mesh keeps `ep` as the innermost axis, so EP groups are node-local and DeepEP only uses its NVLink intranode path; the NVSHMEM/RDMA path is linked but idle (`num_rdma_bytes` stays 0).
- The `axolotlai/axolotl-cloud` `-blackwell` image tags ship the `10.0` wheel together with the rest of the Blackwell kernel stack (see the [Docker docs](https://docs.axolotl.ai/docs/docker.html)).

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

Notes:

- Patch 1 is required because `DISABLE_SM90_FEATURES=1` asserts a build without NVSHMEM, and `setup.py` otherwise picks up the `nvidia-nvshmem` wheel that torch depends on.
- The `v1.2.1` pin is required: it is the last release whose `setup.py` still carries the `disable_nvshmem` path that patch 1 edits. DeepEP `main` restructured `setup.py` and removed that path ([DeepEP #664](https://github.com/deepseek-ai/DeepEP/pull/664)), so against HEAD the patch fails with `patch failed: setup.py:19`. The pin also sidesteps DeepEP HEAD's `csrc/elastic/` (Engram/EPv2, commit `b306af0`), which needs `ncclGinRequest_t` from NCCL 2.29+.

## Usage

EP requires FSDP2 (`fsdp_version: 2` with an `fsdp_config` block) and `fsdp_config.state_dict_type: FULL_STATE_DICT`; config validation rejects EP under DDP, DeepSpeed, or a sharded state dict. The dense weights FSDP-shard, the experts are wrapped on the non-`ep` axes, and the full state dict gathers every EP group's experts before rank 0 writes. `SHARDED_STATE_DICT` (accelerate's FSDP2 default when unset) would keep only EP group 0's experts, because every group's expert shard has the same name, shape and offset and DCP deduplicates them.

```yaml
plugins:
  - axolotl.integrations.expert_parallel.ExpertParallelPlugin

expert_parallel_size: 2  # 1 = disabled (default); > 1 = enabled
fsdp_version: 2
fsdp_config:
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: Qwen3MoeDecoderLayer
  state_dict_type: FULL_STATE_DICT
```

`dp_shard_size` is the dense FSDP degree (transformers' `fsdp_size`); `dp_replicate_size × dp_shard_size × context_parallel_size` must equal `world_size`, and an unset `dp_shard_size` takes the ranks the other degrees leave. EP regroups the `dp_shard_size` ranks rather than adding a factor, so `expert_parallel_size` must divide `dp_shard_size`: each group of `expert_parallel_size` consecutive ranks holds one full set of experts, split across its ranks, and each expert slice is FSDP-split across the `dp_shard_size / expert_parallel_size` groups (transformers' `efsdp_size`):

```yaml
# 4 GPUs: dense weights FSDP-sharded 4 ways, experts EP-sharded 2 ways and FSDP-split 2 ways
expert_parallel_size: 2
dp_shard_size: 4  # or leave unset
fsdp_version: 2
fsdp_config:
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: Qwen3MoeDecoderLayer
  state_dict_type: FULL_STATE_DICT
  reshard_after_forward: true
```

Earlier releases read `dp_shard_size` under EP as the split inside an EP group, with `expert_parallel_size` as a separate factor of the world size (`ep × dp_shard = world`, so `dp_shard_size: 2` in the example above). That spelling still loads, resolves to the same layout and logs a deprecation warning with the rewritten value (the old `dp_shard_size` times `expert_parallel_size`); it will be removed in a future release. A `dp_shard_size` that fits neither spelling fails at config load with a suggested value instead of at `Accelerator` creation after the model has loaded.

Full-parameter training works with either layout. At 2 GPUs (`expert_parallel_size` equal to the world size, so experts are not FSDP-split) the dense parameters FSDP-shard across the whole world while the trainable experts are wrapped on a per-rank mesh, so every parameter is a DTensor and fused/foreach optimizers and gradient clipping see one kind of tensor. The expert combine sums gradients over the ranks that sent tokens where FSDP averages, so the experts' reduce-scatter is divided by the dense `dp_shard` degree (`ep × dp_shard` in the old spelling); the full state dict gathers the experts across the EP axis before rank 0 writes, so the saved model carries every expert.

See full example configs at [`examples/expert_parallel/`](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/expert_parallel).

#### Implementation notes

EP runs as one experts implementation, `expert_parallel`, that wraps the local one you've already configured: the plugin records it, sets `experts_implementation: expert_parallel`, and calls the local implementation on the rows each rank receives. The local implementation is, in order of precedence:

1. `use_scattermoe: true` -> ScatterMoE (Triton)
2. `use_sonicmoe: true` -> SonicMoE (bf16 gated experts)
3. `experts_implementation: <name>` -> any implementation registered in transformers' `ALL_EXPERTS_FUNCTIONS` (`grouped_mm`, `batched_mm`, your own) or `eager` (a Python loop, the numerics reference). transformers' `deepgemm` has no backward pass and is rejected.
4. unset -> `grouped_mm` (with the kernels plugin loaded, unset means `eager`: its validator pins that default)

The same list applies to either backend. Slots routed to another rank reach the local implementation as expert id `num_experts` with weight 0, the sentinel transformers' kernels already drop, so a newly registered implementation works under EP with no EP-side code as long as it ignores ids `>= num_experts`.

The pre-`expert_parallel` names (`deep_ep`, `deep_ep_grouped_mm`, `deep_ep_scattermoe`, `deep_ep_sonicmoe`, `torch_ep_eager`, `torch_ep_grouped_mm`, `torch_ep_scattermoe`, `torch_ep_sonicmoe`) are deprecated: they still load, with a warning, as the matching local implementation (`*_scattermoe` / `*_sonicmoe` set `use_scattermoe` / `use_sonicmoe`). The backend prefix is ignored, as before; `expert_parallel_backend` picks the dispatch.

EP composes with FSDP on orthogonal mesh axes: experts are sharded across the `ep` axis, non-expert params across `dp_shard`. The two collectives run on disjoint process groups, so they don't conflict. The mesh is ordered `(dp_replicate, dp_shard, cp, ep, tp)`, `ep` innermost, so each EP all-to-all group is a run of consecutive ranks that stays inside a node whenever `ep × tp` fits on one; the startup log reports which axes cross nodes. Layout follows [*Expert Parallelism with FSDP* (tinkerings.dev)](https://tinkerings.dev/posts/expert_parallel.html) — "rows share weights, columns move tokens."

## Training checkpoints and resume

FSDP2 `FULL_STATE_DICT` checkpoints preserve every EP group's expert weights,
expert LoRA, and optimizer state. The shared ownership-aware checkpoint path serves
both Trainer and Accelerate. See [Full checkpoints and resume](https://docs.axolotl.ai/docs/multi-gpu.html#full-checkpoints-and-resume)
for quantization, memory, and mesh compatibility limits.

### Checkpoint files

- `pytorch_model_fsdp.bin` is the native model checkpoint used by
  `resume_from_checkpoint` and best-model loading. Trainer LoRA/QLoRA checkpoints
  include PEFT export state, including frozen adapter factors, saved modules, and
  required embeddings; the frozen base model is loaded separately. In model-only
  checkpoints of a single-adapter PEFT model, and in every checkpoint written with
  `adapter_checkpoint_format: sharded`, this file only references the
  `adapter_model.safetensors` beside it, and restore reads the adapter from there.
  Full-parameter and multi-adapter checkpoints, and checkpoints from earlier
  versions, hold the complete state in this file.
- `optimizer.bin` preserves every expert owner's optimizer state. It is absent
  when `save_only_model: true`.
- `adapter_model.safetensors` is the ordinary PEFT adapter export used by
  `lora_model_dir`. Model-only intermediate checkpoints also include the native
  resume file; final exports do not. Keep it with the native resume file when
  copying model-only checkpoints, since the resume file may reference it.

Native checkpoints and final exports need not have the same layout. For example,
Mixtral full-model exports unpack fused expert weights, while native checkpoints
retain fused tensors. Packed bitsandbytes checkpoints also carry owner-specific
quantization metadata and require unchanged EP ownership ranges.

### Checkpoints from earlier versions

Older EP checkpoints may contain only EP group 0's expert weights, expert LoRA,
and optimizer moments. The full-checkpoint loader rejects incompatible shapes on
every rank. Model tensor shapes are checked before any model parameters are copied,
so a truncated model checkpoint does not partially overwrite the model.

To recover from an incomplete legacy checkpoint:

- **Full-parameter training:** start a new run from a complete final model export.
- **LoRA:** if the legacy checkpoint has a complete `adapter_model.safetensors`,
  load it with `lora_model_dir` in a new run.

These recovery paths do not recover optimizer state or scheduler progress. Missing
experts cannot be reconstructed from the truncated native checkpoint.

## Limitations

- Models' modeling code must use `@use_experts_implementation` (canonical 3D `gate_up_proj` / `down_proj`, or `up_proj` / `down_proj` for non-gated experts such as Nemotron-H's relu² MoE). `ModuleList` as used in Mixtral is not supported.
- `num_experts` must be divisible by `expert_parallel_size`.
- EP requires FSDP2 with `fsdp_config.state_dict_type: FULL_STATE_DICT` (see [Usage](#usage));
  DDP, DeepSpeed and `SHARDED_STATE_DICT` are rejected at config validation.
- Supported mesh axes (`dp_replicate × ep × cp × dp_shard == world_size`):

  | Layout | `dp_replicate` | `ep` | `cp` | `dp_shard` | Status |
  |---|:---:|:---:|:---:|:---:|---|
  | EP | 1 | =world | 1 | 1 | ✅ experts whole per rank, dense weights sharded over the world |
  | EP × dp_shard | 1 | >1 | 1 | >1 | ✅ |
  | EP × cp | 1 | >1 | >1 | 1 | ✅ expert slices shard over `cp` |
  | EP × cp × dp_shard | 1 | >1 | >1 | >1 | ✅ expert slices shard over `dp_shard × cp` |
  | HSDP × EP × dp_shard | >1 | >1 | 1 | >1 | ✅ `dp_replicate` outermost |
  | HSDP × EP × cp | >1 | >1 | >1 | 1 | ✅ `dp_replicate` outermost |
  | HSDP × EP × cp × dp_shard | >1 | >1 | >1 | >1 | ✅ |
  | HSDP × EP (no `dp_shard`, no `cp`) | >1 | >1 | 1 | 1 | ✅ `dp_replicate` outermost; each replica keeps its expert slices whole (size-1 shard mesh) and reduces them across replicas |
  | EP × TP | * | >1 | * | * | ❌ rejected at config validation (`ValueError`) |

  Experts shard on `ep`, the sequence on `cp`, dense weights on every non-`ep` axis, expert
  slices on `dp_shard × cp`. Stock attention runs CP through ringmaster (Ulysses preferred);
  GLM-5.2 DSA via the kernels plugin brings its own `cp`-aware attention.
- transformers' own expert/tensor parallelism (`distributed_config` with `enable_expert_parallel`,
  `tp_plan`, `tp_size` via `model_kwargs`) is rejected: the plugin shards the experts itself and
  routes tokens with its own backends, not transformers' RouterParallel plan.
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
