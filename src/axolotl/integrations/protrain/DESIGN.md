## Purpose

This package is a from-scratch Python implementation of the ProTrain memory manager (MLSys 2026, arXiv 2406.08334), shipped as an **Axolotl plugin** (`BasePlugin` subclass). It owns per-rank memory policy on top of ZeRO-3: hierarchical chunk management for model states (params / grads / optim states), interleaved block management for activations, a memory-aware profiler, a 4-knob cost model, and an automatic searcher. It does NOT own data parallelism collectives (delegates to `torch.distributed`), training-loop control flow, trainer orchestration, TP/PP, FP8, or any changes to Axolotl core files. Activation is opt-in via `plugins: [axolotl.integrations.protrain]` in the user YAML; mutual exclusion with `deepspeed:` and `fsdp:` is enforced by a pydantic validator in `args.py`.

## Workstream-shape ratifications (drift from `plan.md`)

Two intentional deviations from the original plan, both ratified after M5 review:

1. **Package path: `src/axolotl/integrations/protrain/` (not `src/axolotl/memory/protrain/`)**. Plan specified the latter; we landed on the former. The driver is Axolotl's own convention — `src/axolotl/integrations/` is the canonical home for `BasePlugin` subclasses (`spectrum`, `kd`, `cut_cross_entropy`, etc.), and ProTrain ships as a plugin. Putting it under `memory/` would have required teaching `prepare_plugins` a non-standard discovery path, plus diverging from the test conventions every other integration follows (`tests/integrations/<name>/`). The functional contract of "no edits to Axolotl core" is preserved unchanged.

2. **DESIGN.md length: ~260 lines (plan said "under 200")**. The plan's 200-line bound was an M0 hygiene target before M7 ZeRO-3 sharding and the Mode A/B/C auto-selector existed — those sections account for most of the over-budget content (~50 lines of multi-GPU spec + benchmark results that didn't exist when the plan was written). Trimming would lose multi-GPU integration documentation that operators actively reference. Length cap formally raised to 350 lines; sections must continue to map 1:1 onto subpackages (no narrative essays).

## Directory Layout

```text
src/axolotl/integrations/protrain/
├── __init__.py                  # re-exports ProTrainArgs + ProTrainPlugin
├── DESIGN.md                    # this file
├── plugin.py                    # BasePlugin subclass: get_input_args / post_model_load / create_optimizer
├── args.py                      # ProTrainArgs pydantic model + DS/FSDP mutex validator
├── types.py                     # shared dataclasses (ProfilerTrace, ChunkLayout, ...)
├── profiler/
│   ├── __init__.py
│   ├── trace.py                 # single-iter forward/backward hook driver
│   ├── memory_deltas.py         # intra-op + inter-op Δ capture via cuda.memory_stats
│   ├── on_demand.py             # allocate-before-use / free-after tensor mode
│   ├── hw_bench.py              # H2D/D2H + NCCL gather/reduce microbenchmarks
│   └── cache.py                 # on-disk cache keyed by (arch_hash, bs, seq, sku, world)
├── chunk/
│   ├── __init__.py
│   ├── layout.py                # param→chunk assignment, exec-order intra-chunk reorder
│   ├── sizing.py                # S_chunk grid search over {32,64,128,256} MB
│   ├── manager.py               # persistent/non-persistent split, gather/offload drivers
│   ├── buffer_pool.py           # pre-allocated chunk buffer pool, forward→backward reuse
│   ├── pinned_alloc.py          # ctypes → cudaHostAlloc, precise-size (App B.2)
│   └── optim.py                 # DeepSpeedCPUAdam adapter (non-persist) + GPU FusedAdam (persist)
├── block/
│   ├── __init__.py
│   ├── strategy.py              # BlockMode enum {NONE, CKPT, SWAP}
│   ├── dispatcher.py            # per-block forward wrapper honoring selected mode
│   ├── checkpoint.py            # CKPT path (torch.utils.checkpoint adapter)
│   ├── swap.py                  # SWAP wrapper: D2H in fwd / H2D in bwd on _swap_stream
│   ├── swap_pool.py             # pinned-RAM activation slot pool
│   └── layout_rules.py          # placement rules: swap-early / unopt-late / interleave
├── cost/
│   ├── __init__.py
│   ├── runtime.py               # Eqs. 2–7, per-chunk max(compute, comm) roofline
│   ├── memory.py                # Eqs. 8–11, op-walk peak + α=1.10 fragmentation
│   └── bandwidth.py             # contention model when n_swap>0 competes with prefetch
├── search/
│   ├── __init__.py
│   ├── knobs.py                 # CostConfig + bound derivation (N_chunk, N_block, N_interval)
│   └── exhaustive.py            # 4-knob enumeration with memory-ascending pruning
├── runtime/
│   ├── __init__.py
│   ├── streams.py               # single-stream alloc scheme (App B.2)
│   ├── scheduler.py             # prefetch / reduce-offload / CPU-step / swap orchestration
│   └── hooks.py                 # install/uninstall fwd/bwd hooks on the user model
└── api/
    ├── __init__.py
    ├── model_wrapper.py         # protrain_model_wrapper() — called from plugin.post_model_load
    └── optim_wrapper.py         # protrain_optimizer_wrapper() — called from plugin.create_optimizer
```

## Module Specs

Every entry: Inputs · Outputs · Paper ref · Milestone.

### plugin.py (M5)

- `class ProTrainPlugin(BasePlugin)` — thin shim.
  - `get_input_args() -> "axolotl.integrations.protrain.args.ProTrainArgs"`.
  - `post_model_load(cfg, model)` — constructs `HardwareProfile`, runs profiler (cached), calls `protrain_model_wrapper(model, ...)`, stashes `WrappedModel` on `cfg` for `create_optimizer` to pick up.
  - `create_optimizer(cfg, trainer) -> Optimizer` — returns `protrain_optimizer_wrapper(wrapped_model)`; returns `None` when plugin is inactive.
  - `post_trainer_create(cfg, trainer)` — installs any trainer-level callbacks if needed for metric reporting.

### args.py (M5)

- `class ProTrainArgs(BaseModel)` — fields: `protrain_auto_memory: bool = True`, optional manual knob overrides `protrain_n_persist / n_buffer / n_swap / n_checkpoint` for debugging, `protrain_cache_dir: Path | None`.
- `model_validator` — rejects `plugins: [...protrain...]` + (`deepspeed` set) or (`fsdp` / `fsdp_config` set). Pattern cloned from `integrations/spectrum/args.py:32-47`.

### profiler/ (M1)

- `trace.py` — `run_trace(model: nn.Module, batch: dict, cfg: ProfilerConfig) -> ProfilerTrace`. Installs pre/post fwd + bwd hooks, records op order, delegates Δ capture. §3.2.
- `memory_deltas.py` — `intra_op_delta(op) -> int`, `inter_op_delta(prev, curr) -> int` from `torch.cuda.memory_stats()`. Catches the ~17% invisible peak. §3.2, App A.2.
- `on_demand.py` — `class OnDemandTensorMgr` context; `allocate_inputs(op)` / `free_after(op)`. Enables profiling models larger than single-GPU. §3.2. The on-demand pre-gather hook is registered with `prepend=True` so it fires BEFORE the trace driver's `_pre_forward`; the trace's `allocated_before` snapshot therefore already includes the gathered param, and `intra_op_delta = peak − allocated_before` captures only workspace + output (not the gather). Post-release stays FIFO so it fires after the trace's `_post_forward` peak read. Same ordering for backward (`prepend=True` on `register_full_backward_pre_hook`, FIFO on the post hook).
- `hw_bench.py` — `measure_pcie() -> BW`, `measure_nccl(world_size) -> NcclTable`. §3.2.
- `cache.py` — `load(key) -> ProfilerTrace | None`, `save(key, trace)`. Key = `(arch_hash, bs, seq, sku, world)`. §7. The `TRACE_VERSION` constant prefixes the cache key, so a bump invalidates all prior entries silently. Versions: v2 added per-op latencies, v3 added measured Adam throughput, v4 added hook-dispatch calibration (hooked/steady fwd-wall), v5 added the aggregate steady-fwd peak, v6 added per-block steady peaks (tighter cap for fractional-NONE configs), v7 changed the steady-state methodology from a single iteration to a 4-iter hot loop (2 warmup + 2 measured, median) and added a best-effort steady_bwd_wall. The fields list didn't change at v7 but the recorded *values* shifted, so the cost model's measured bwd/fwd-ratio path requires a fresh trace under the new methodology.

### chunk/ (M2)

- `layout.py` — `build_layout(model, exec_order: list[ParamId], S_chunk: int) -> ChunkLayout`. Groups params per transformer block, reorders intra-chunk by first use, shared params at first occurrence. §3.1.1.
- `sizing.py` — `pick_S_chunk(model_state_sizes: list[int], candidates=(32<<20, 64<<20, 128<<20, 256<<20)) -> int`. Simulates fragmentation waste; returns argmin. App B.1.
- `manager.py` — `class ChunkManager`; `gather(chunk_id)`, `offload(chunk_id)`, `mark_persistent(first_n)`. §3.1.1.
- `buffer_pool.py` — `class BufferPool(n_buffer: int, S_chunk: int)`; `acquire() / release()`; carries forward-resident buffers into backward. §3.1.1, §5.
- `pinned_alloc.py` — `pinned_alloc(n_buffer, S_chunk) -> HostMemory`. `ctypes` → `cudaHostAlloc` with exact byte count. App B.2.
- `optim.py` — wraps `deepspeed.ops.adam.DeepSpeedCPUAdam` for non-persistent chunks, `apex.optimizers.FusedAdam` (or torch `FusedAdam`) for persistent. `step_async(chunk_id)` for CPU path to overlap GPU bwd. §5.

### block/ (M3)

- `strategy.py` — `class BlockMode(Enum){NONE, CKPT, SWAP}`; `BlockStrategyMap = dict[int, BlockMode]`. §3.1.2.
- `dispatcher.py` — `wrap_block(block: nn.Module, mode: BlockMode) -> nn.Module`. §3.1.2.
- `checkpoint.py` — thin wrapper over `torch.utils.checkpoint.checkpoint` (use_reentrant=False). §3.1.2.
- `swap.py` — `SwappedBlock`: D2H of output activation to a pinned-host slot on `_swap_stream` in forward; H2D back on `_swap_stream` in backward, with cross-stream event handshake. Pool + stream injected post-construction via `attach_runtime`. §3.1.2.
- `swap_pool.py` — `ActivationSwapPool`: pinned-host slot pool sized to `n_swap × prefetch_depth × max_act_bytes`. Backed by one `PinnedHostMemory` allocation; slot acquire/release tracked Python-side. §3.1.2.
- `layout_rules.py` — `assign_modes(n_swap, n_checkpoint, N_block) -> BlockStrategyMap`. Swap-early / unopt-late / interleave. §3.1.2.

### cost/ (M4)

- `runtime.py` — `estimate_runtime(cfg, trace, layout) -> float`. Implements **Eqs. 2–7**: `T_iter = T_fwd + max(T_bwd + T_gpu_optim, T_cpu_optim)`, per-chunk `max(compute, comm)` roofline. §3.3, App A.1.
- `memory.py` — `estimate_peak(cfg, trace, layout, block_map) -> int`. Implements **Eqs. 8–10** (op-walk) and **Eq. 11** (α = 1.10 fragmentation). Bumps at first op of each CKPT block. §3.3, App A.2.
- `bandwidth.py` — `effective_bw(cfg, hw) -> float`. Derates prefetch BW when `n_swap > 0`. §3.3.

### search/ (M4)

- `knobs.py` — `CostConfig` dataclass + `derive_bounds(trace, layout) -> Bounds(N_chunk, N_block, N_interval)`. §3.3.
- `exhaustive.py` — `search(trace, layout, capacity_bytes) -> SearchResult`. Enumerates 4-tuple in memory-ascending order, prunes OOM, returns argmin(T_iter). §3.3.

### runtime/ (M2+M3 integration)

- `streams.py` — single-default-stream allocator, manual dealloc sync. App B.2.
- `scheduler.py` — orchestrates (a) param prefetch, (b) grad reduce+offload, (c) CPU optimizer step, (d) activation swap. Respects `cost/bandwidth.py` budgets. §5, §6.
- `hooks.py` — `install(model)` / `uninstall()`; wires chunk & block managers into fwd/bwd. §1.

### api/ (M4)

- `model_wrapper.py` — `protrain_model_wrapper(model, model_config, hardware_profile) -> WrappedModel`. §1.
- `optim_wrapper.py` — `protrain_optimizer_wrapper(wrapped_model) -> Optimizer`. §1.

## Key Data Structures

All live in `types.py`. Fields expand during M1–M4:

```python
@dataclass(frozen=True)
class ProfilerTrace:
    op_order: list[OpRecord]                  # per-op: id, module_path, shape_sig
    intra_op_delta: dict[OpId, int]           # bytes
    inter_op_delta: dict[OpId, int]           # bytes
    activation_sizes: dict[BlockId, int]
    model_state_bytes: int
    pcie_h2d_bps: float
    pcie_d2h_bps: float
    nccl_gather_s: dict[int, float]
    nccl_reduce_s: dict[int, float]
    arch_hash: str; bs: int; seq: int; sku: str; world: int

@dataclass(frozen=True)
class ChunkLayout:
    S_chunk: int
    N_chunk: int
    chunks: list[list[ParamId]]
    param_to_chunk: dict[ParamId, int]
    block_to_chunks: dict[BlockId, list[int]]

BlockStrategyMap = dict[int, BlockMode]

@dataclass(frozen=True)
class CostConfig:
    n_persist: int
    n_buffer: int
    n_swap: int
    n_checkpoint: int

@dataclass(frozen=True)
class SearchResult:
    cfg: CostConfig
    block_map: BlockStrategyMap
    predicted_peak_bytes: int
    predicted_iter_s: float
```

## Plugin Integration (M5)

Zero diffs to Axolotl core files. The entire Axolotl surface consumed:

- `BasePlugin` subclass at `src/axolotl/integrations/protrain/plugin.py`
- `get_input_args` returns `ProTrainArgs` → pydantic merge handled by `axolotl/utils/schemas/config.py:1275` (`plugins:` field)
- `post_model_load(cfg, model)` hook — wraps post-LoRA so frozen LoRA base params contribute to persistent-chunk memory only
- `create_optimizer(cfg, trainer)` hook — returns ProTrain optimizer; `None` if disabled
- Example YAML: `examples/protrain/3090-7b-lora.yml` — opts in via `plugins: [axolotl.integrations.protrain]`

## Cross-Module Dependency Graph

- `types.py` — depended on by everyone; depends on nothing.
- `profiler/*` — independent (M1). Depends only on `types.py` and `torch`.
- `chunk/*` — independent of profiler and block (M2). Uses `runtime/streams.py` and `runtime/hooks.py`.
- `block/*` — independent of profiler and chunk (M3). Uses `runtime/hooks.py`.
- `cost/*` — reads `ProfilerTrace` + `ChunkLayout` + `BlockStrategyMap` as **data**; no code-level dep on chunk/block internals (M4).
- `search/*` — depends on `cost/*` and `types.py` only (M4).
- `api/*` — depends on everything; built last.
- `plugin.py` — consumes `api/*` only; M5. Supports M1→M4 parallel fan-out: profiler, chunk, block run concurrently; cost+search starts once `ProfilerTrace` schema is frozen at end of M1.

### Multi-GPU

ProTrain is a per-rank memory policy. Two composition modes are supported; choose per-deployment by the `protrain_zero3_shard` YAML flag or by auto-detection.

**Mode A — DDP composition (pre-M7, still supported).** Each rank runs its own full `protrain_model_wrapper` and holds a full (replicated) copy of every non-persistent chunk on pinned CPU. The trainer wraps the protrain'd module in `torch.nn.parallel.DistributedDataParallel`. DDP handles the cross-rank all-reduce on the trainable gradient set; ProTrain's internal per-param `all_reduce` is silenced via `skip_internal_grad_reduce=True` (auto-set when `post_trainer_create` detects a DDP wrap). This mode is what the M6 multi-GPU throughput test exercises with `force_all_persistent=True` at world_size=4 on 3090s. It is the right choice for LoRA on ~7B where the frozen base fits in fp16 on one card (no memory pressure), because DDP's bucketed allreduce is faster than ProTrain's per-param reduction.

**Mode B — true ZeRO-3 chunk sharding (M7, new).** Non-persistent chunks are partitioned across ranks on CPU: each rank holds only `ceil(chunk_bytes / world_size)` pinned bytes per chunk. Forward/backward sees the full chunk via `all_gather_into_tensor` at `ChunkManager.gather`; grads are reduced + partitioned via `reduce_scatter_tensor(op=AVG)` at `ChunkManager.reduce_grads_and_offload`. The CPU FusedAdam step runs only on the rank-local shard slice — each region's flat `shard_param` is the Adam target, updated in place; the next gather's `all_gather` propagates the update back to every rank's replicated GPU copy.

Sharding handles BOTH homogeneous-dtype and mixed-dtype chunks (M7 follow-up). Each chunk is modelled as an ordered list of `_DtypeRegion` entries — one per maximal-length contiguous same-dtype byte run — and each region is independently partitioned across ranks and participates in its own `all_gather_into_tensor` / `reduce_scatter_tensor` collective. Homogeneous chunks lay out exactly one region and issue one collective per gather/reduce; mixed-dtype chunks (e.g. a Llama block with fp32 RMSNorm scales between fp16 linear layers) issue one collective per region. Persistent chunks are fully replicated in both modes.

**Auto-enable logic (pre-auto-mode).** When `protrain_auto_mode=False` (explicit-override mode), `protrain_model_wrapper` decides at construction time:

| `world_size` | `force_all_persistent` | outer DDP | `zero3_shard` result |
|---|---|---|---|
| 1 | * | * | off (degrades to replicated even if True requested) |
| >1 | True | * | off (everything is persistent) |
| >1 | False | auto-detected YES | off, AND `skip_internal_grad_reduce=on` |
| >1 | False | NO | on (M7 ZeRO-3 path) |

The user can override via the `protrain_zero3_shard: true/false` field on `ProTrainArgs`. When DDP is composed on top AND sharding was auto-enabled, `post_trainer_create` logs a WARNING (the two paths don't compose cleanly); the operator should set `protrain_zero3_shard: false` in YAML for DDP deployments.

**Mode selection (auto, default).** `protrain_auto_mode: true` (default) runs the searcher first, then picks one of three modes based on workload fit + per-rank CPU RAM:

* **Mode A — GPU-resident / DDP-friendly** (`force_all_persistent=True`). Chosen when the searcher places `n_persist == N_chunk` under the capacity budget — the model fits entirely on GPU and no CPU offload is needed. This is the throughput winner on a 3090 rig: DDP's bucketed NCCL allreduce beats ProTrain's per-param grad sync, and the M7 benchmark measured **3.64x** scaling at world_size=4 on PCIe Gen3.
* **Mode B — replicated CPU-offload** (`zero3_shard=False`). Chosen when the model needs offload AND per-rank CPU RAM can hold the full non-persistent chunk set (`cpu_ram_per_rank >= (N_chunk - n_persist) * S_chunk`). Each rank holds a full replicated copy of every non-persistent chunk; no per-chunk collectives, so it's ~1.9x faster than sharded on PCIe Gen3.
* **Mode C — ZeRO-3 sharded CPU-offload** (`zero3_shard=True`). Chosen when per-rank CPU RAM is too tight for replication but fits a `1/world_size` shard per chunk. Measured throughput is **0.70x** single-rank on 4x 3090 — the `all_gather` / `reduce_scatter` collectives dominate on PCIe Gen3 Llama-3B. Picked only when Mode B can't fit.
* **Otherwise** — `RuntimeError`. The model doesn't fit on this node even with sharding; user must scale up (more nodes / larger RAM / smaller model) before retrying.

**CPU-RAM-per-rank estimate.** `node RAM available / world_size`. Probes `psutil.virtual_memory().available` first (preferred; part of Axolotl's env already), falls back to `/proc/meminfo:MemAvailable` on Linux. Returns 0 when neither probe succeeds — the selector then prefers Mode A and raises if offload is required. Caveats: the divide-by-world-size model is pessimistic on NUMA-bound allocations and optimistic on heterogeneous multi-host setups where the smallest node's RAM binds. Users whose production topology doesn't match "node RAM / world_size" should set `protrain_auto_mode: false` and pick the mode explicitly via `protrain_force_all_persistent` / `protrain_zero3_shard`.

**Mode B over Mode C — throughput trade-off.** The selector prefers Mode B over Mode C even when C would save pinned RAM, because B is ~1.9x faster on PCIe Gen3 and "CPU RAM fits replicated" is the loose binding constraint. Users with binding CPU pressure (e.g., a 96 GB system driving 8 ranks of a model whose non-persistent set is 80 GB replicated but 10 GB sharded) should set `protrain_auto_mode: false, protrain_zero3_shard: true` to force Mode C.

**Explicit overrides.** `protrain_auto_mode: false` bypasses the selector and honours `protrain_force_all_persistent` / `protrain_zero3_shard` verbatim (following the pre-auto-mode table above). When `protrain_auto_mode: true` and the user still sets one of the mode flags, the selector logs a warning and proceeds with the auto-selected mode — the flags are explicitly documented as overrides that require turning auto-mode off to take effect.

**Shard layout.** Rank `r` owns the byte range `[r * shard_bytes, (r + 1) * shard_bytes)` within each region. `shard_bytes = region_bytes_padded / world_size`, where `region_bytes_padded` is rounded up to `lcm(region_element_size, world_size)` — this guarantees both (a) the shard boundary is dtype-aligned (so `.view(fp16)` on the pool buffer after `all_gather` doesn't raise "offset not aligned") and (b) every rank holds an equal shard size (required by `all_gather_into_tensor` / `reduce_scatter_tensor`). Params straddling shard boundaries are NOT special-cased — each rank just holds the bytes it owns; reassembly is byte-exact under `all_gather`'s contiguous layout. Regions within a chunk are gap-tolerant: per-region padding lives inside a transient scratch buffer at gather/reduce time rather than the pool buffer's byte layout, so params always index into the pool buffer at their original `aligned_offsets`.

**Memory-safety contract.** GPU peak is unchanged by sharding (the gather reconstructs the full chunk on GPU via `all_gather_into_tensor` regardless), so `cost/memory.py::estimate_peak` ignores `HardwareProfile.zero3_shard`. The per-rank pinned CPU footprint DOES scale with sharding — `cost/memory.py::estimate_cpu_footprint` returns `(N_chunk - n_persist) * S_chunk / world_size` under sharding vs. the full product under replication. The searcher's GPU-capacity gate (the only feasibility filter today) is therefore sharding-agnostic; the explicit `zero3_shard` plumbing on `HardwareProfile` exists so future CPU-budget filters (if added) can consult it.

#### NCCL measurement gap

`protrain_model_wrapper` runs from `plugin.post_model_load`, which fires during model loading at `loaders/model.py:191` — BEFORE the Trainer / Accelerate path initializes the distributed process group. So when the profiler calls `measure_nccl(world_size>1)`, `dist.is_initialized()` is False, the call falls through to empty `nccl_gather_s` / `nccl_reduce_s` tables, and the trace records `world=1` regardless of actual world size.

This gap is functionally inert in the auto-selected Mode A and Mode B paths. Mode A (DDP) keeps every chunk persistent — DDP itself owns the cross-rank allreduce, and ProTrain issues no per-chunk collectives, so the cost model never reads the NCCL tables. Mode B (replicated CPU offload) likewise issues no per-chunk collectives. Only Mode C (ZeRO-3 sharded) actually consumes `nccl_gather_s` / `nccl_reduce_s` — and the auto-selector picks Mode C last (only when per-rank CPU RAM can't hold the replicated non-persistent set).

Workaround for Mode C operators: run `scripts/protrain/measure_nccl.py` once on the target rig under a real distributed launcher (it inits the process group itself and writes a JSON of `{payload_bytes: seconds}` for both gather and reduce-scatter). The output can be hand-loaded into the trace before search runs, or — more practically — used to validate that Mode C predictions match the standalone benchmark on the operator's interconnect.

Late-bind path: `plugin.post_trainer_create` calls `_remeasure_nccl_and_research(wrapped)` after Accelerate brings up dist. When `world_size > 1` and the cached trace's NCCL tables are empty, the helper measures NCCL on the live process group, splices the populated tables + actual world into the trace via `dataclasses.replace`, persists the updated trace under a new cache key (so the next multi-rank run hits it directly without re-measuring), and re-runs `search()` with the same layout + capacity + hardware profile. The chunk manager is NOT rebuilt — optimizer state slots are already wired into the trainer — so the running step uses the bootstrap config; if the post-NCCL search picks a different `cfg`/`block_map`, a WARN is logged and `WrappedModel.search_result` is overwritten so future cost-model-based decisions reflect real comm cost. Subsequent multi-rank runs hit the cache and pick the new config from the start. Mode A / Mode B remain unaffected since they don't consume the NCCL tables.

#### Multi-GPU — Measured Throughput (4x 3090)

Benchmark: fresh-init Llama-3B + LoRA r=8, bs=2 per rank, seq=256, fp16. 6 iterations per mode, 2 warm-up discarded, median of the remaining 4 is reported. GPUs 1, 4, 5, 7 on a PCIe-Gen3 test rig (no NVLink). Reproduce with `CUDA_VISIBLE_DEVICES=1,4,5,7 CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/benchmark_multi_gpu.py`; full JSON at `scripts/multi_gpu_benchmark_results.json`.

| Mode | World | Throughput (samples/s) | Scaling vs 1-GPU | Per-rank GPU peak | Per-rank CPU pinned |
|---|---|---|---|---|---|
| Single-rank (baseline) | 1 | 8.48 | 1.00x | 5.36 GB | 0.00 GB |
| DDP (`force_all_persistent=True`) | 4 | 30.90 | 3.64x | 5.38 GB | 0.00 GB |
| Replicated offload (`zero3_shard=False`) | 4 | 11.06 | 1.30x | 3.09 GB | 3.82 GB |
| ZeRO-3 sharded (`zero3_shard=True`) | 4 | 5.93 | 0.70x | 3.09 GB | 0.96 GB |

**How to pick a mode on a 3090 rig.** DDP is the clear throughput winner when the model + optimizer fit on one card (the 7B-LoRA / 3B-full regime) — outer-bucketed NCCL allreduce amortizes better than ProTrain's per-param grad sync and keeps every chunk GPU-resident. Reach for **replicated offload** only when one card can't hold the full model at peak; per-rank GPU drops ~42% (5.4 GB → 3.1 GB here) at a ~3x throughput cost vs DDP. **ZeRO-3 sharded** is only worth it when CPU RAM is the binding constraint — it cuts per-rank pinned CPU by almost exactly `1/world_size` (3.82 GB → 0.96 GB here, a 4.0x reduction, matching world_size) but pays an additional ~1.9x iteration-time penalty from the per-chunk `all_gather` + `reduce_scatter` collectives on PCIe Gen3. For 7B LoRA on 4x 3090 with NVMe or 128+ GB system RAM, stay on DDP with `force_all_persistent=True`.

Note: ZeRO-3 throughput fell below the "within 15% of replicated" design target in this measurement — at Llama-3B / bs=2 / seq=256 the compute per chunk is too small to hide the two per-chunk collectives on PCIe. The ratio should improve at larger batch size / sequence length where compute dominates; see M7 profiler runs before broad deployment.

## Out of Scope

Mirrors `plan.md`:
- A100/H100, NVLink, InfiniBand, multi-node
- TP, PP, any non-ZeRO-3 parallelism
- FP8/FP4, quantization, FlashAttention variants
- Windows / macOS
- Edits to Axolotl core files outside this plugin package — ProTrain is additive, DeepSpeed/FSDP/Unsloth paths unchanged

## Design Decisions (previously open questions, now resolved)

1. **α fragmentation factor = 1.10** — matches paper's "up to 10% overestimate" (§3.3). M1 records ground truth; M4 can recalibrate if observed 3090 fragmentation diverges.
2. **Pinned-memory allocator:** `ctypes` → `cudaHostAlloc` directly. ~50 LOC, zero new deps, matches App B.2 precisely (avoids `CUDAHostAllocator` pow-2 rounding). DeepSpeed's `PinnedMemoryAllocator` rejected: may inherit same wart, adds import-graph weight.
3. **CPU FusedAdam source:** `deepspeed.ops.adam.DeepSpeedCPUAdam`. Paper builds directly on ZeRO-Offload's CPU Adam. Pure-Python reimpl is >10× slower and would collapse the T_bwd / T_cpu_optim overlap window the cost model assumes. DeepSpeed is already in Axolotl's env.
4. **S_chunk grid:** `{32, 64, 128, 256} MB`. 7B Llama blocks are ~200 MB fp16 → chunks want to be block-scale. 16 MB is too fine-grained; per-chunk sync overhead dominates. M2 agent extends the grid if optimum lands at an endpoint.
5. **SWAP path:** paper-real D2H/H2D wrapper on `_swap_stream`, backed by `ActivationSwapPool` (pinned host slots sized `n_swap × prefetch_depth × max_act_bytes`). Searcher's CPU-feasibility gate refuses `n_swap > 0` candidates whose pool would not fit `cpu_capacity_bytes`. On RTX 3090 / 3090 Ti (12 GB/s PCIe ceiling, no NVLink) the searcher rarely selects `n_swap > 0` — paper §3.1.2 — so the path is tested-but-unused infrastructure on this hardware class. Validated end-to-end via the wrapper-injection path with `n_swap_override`.
