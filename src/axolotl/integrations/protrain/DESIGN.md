## Purpose

This package is a from-scratch Python implementation of the ProTrain memory manager (MLSys 2026, arXiv 2406.08334), shipped as an **Axolotl plugin** (`BasePlugin` subclass). It owns per-rank memory policy on top of ZeRO-3: hierarchical chunk management for model states (params / grads / optim states), interleaved block management for activations, a memory-aware profiler, a 5-axis cost model (`n_persist`, `n_buffer`, `n_swap`, `n_checkpoint`, `n_offload` — the OFFLOAD axis was added by Option B / `BLOCK_MODE_OFFLOAD_DESIGN.md`), and an automatic searcher. It does NOT own data parallelism collectives (delegates to `torch.distributed`), training-loop control flow, trainer orchestration, TP/PP, FP8, or any changes to Axolotl core files. Activation requires BOTH `plugins: [axolotl.integrations.protrain.ProTrainPlugin]` in the user YAML (the only spelling the pydantic validator in `args.py` accepts — see `_PROTRAIN_PLUGIN_KEYS`) AND `protrain_auto_memory: true` (default `false` in `ProTrainArgs`); listing the plugin alone registers the args schema but leaves the runtime hooks dormant, so a config that omits `protrain_auto_memory: true` is a no-op. Mutual exclusion with `deepspeed:` and `fsdp:` is enforced by the same validator.

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
│   ├── strategy.py              # BlockMode enum {NONE, CKPT, SWAP, OFFLOAD}
│   ├── dispatcher.py            # per-block forward wrapper honoring selected mode
│   ├── checkpoint.py            # CKPT path (torch.utils.checkpoint adapter)
│   ├── swap.py                  # SWAP wrapper: D2H in fwd / H2D in bwd on _swap_stream
│   ├── swap_pool.py             # pinned-RAM activation slot pool
│   ├── offload.py               # OFFLOAD path (Option B): non-persist chunk re-gather in bwd, no recompute
│   └── layout_rules.py          # placement rules: swap-early / unopt-late / interleave (incl. n_offload)
├── cost/
│   ├── __init__.py
│   ├── runtime.py               # Eqs. 2–7, per-chunk max(compute, comm) roofline
│   ├── memory.py                # Eqs. 8–11, op-walk peak + α=1.10 fragmentation
│   └── bandwidth.py             # contention model when n_swap>0 competes with prefetch
├── search/
│   ├── __init__.py
│   ├── knobs.py                 # CostConfig + bound derivation (N_chunk, N_block, N_interval)
│   └── exhaustive.py            # 5-axis enumeration (incl. n_offload) with memory-ascending pruning
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
  - `post_model_load(cfg, model)` — constructs `HardwareProfile`, runs profiler (cached), calls `protrain_model_wrapper(model, ...)`, stashes `WrappedModel` on `cfg` for the post-trainer-create hook to pick up.
  - `create_optimizer(cfg, trainer) -> Optimizer` — returns `protrain_optimizer_wrapper(wrapped_model)` for plugin loaders that dispatch to it. Axolotl's `OptimizerMixin.create_optimizer` does NOT call into `PluginManager.create_optimizer` (unlike `SchedulerMixin.create_scheduler`), so the actual install path is `post_trainer_create`, below; this method exists for plugin hosts that DO route through it (and to keep the contract documented).
  - `post_trainer_create(cfg, trainer)` — canonical optimizer install path under Axolotl/HF Trainer: builds `protrain_optimizer_wrapper(wrapped)`, assigns it to `trainer.optimizer`, and registers the optimizer-state checkpoint/load callbacks.

### args.py (M5)

- `class ProTrainArgs(BaseModel)` — fields: `protrain_auto_memory: bool = False` (master enable; the plugin's `post_model_load` / `post_trainer_create` hooks early-return when this is `False`, so listing the plugin alone is a no-op until the user explicitly opts in), optional manual knob overrides `protrain_n_persist / n_buffer / n_swap / n_checkpoint` for debugging, `protrain_cache_dir: Path | None`.
- `model_validator` — enforces the dual gate (the plugin must be listed AND `protrain_auto_memory: true` must be set for runtime activation) and rejects `plugins: [...protrain...]` + `protrain_auto_memory: true` + (`deepspeed` set) or (`fsdp` / `fsdp_config` set). Pattern cloned from `integrations/spectrum/args.py:32-47`.

### profiler/ (M1)

- `trace.py` — `run_trace(model: nn.Module, batch: dict, cfg: ProfilerConfig) -> ProfilerTrace`. Installs pre/post fwd + bwd hooks, records op order, delegates Δ capture. §3.2.
- `memory_deltas.py` — `intra_op_delta(op) -> int`, `inter_op_delta(prev, curr) -> int` from `torch.cuda.memory_stats()`. Catches the ~17% invisible peak. §3.2, App A.2.
- `on_demand.py` — `class OnDemandTensorMgr` context; `allocate_inputs(op)` / `free_after(op)`. Enables profiling models larger than single-GPU. §3.2. Hook registration order:
  - Pre-gather hook registered with `prepend=True` → fires BEFORE the trace driver's `_pre_forward`
  - Trace's `allocated_before` snapshot includes the gathered param
  - `intra_op_delta = peak − allocated_before` captures only workspace + output (not the gather)
  - Post-release uses FIFO ordering → fires after the trace's `_post_forward` peak read
  - Same ordering pattern for backward (`prepend=True` on `register_full_backward_pre_hook`, FIFO on the post hook)
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

- `strategy.py` — `class BlockMode(Enum){NONE, CKPT, SWAP, OFFLOAD}`; `BlockStrategyMap = dict[int, BlockMode]`. §3.1.2.
- `dispatcher.py` — `wrap_block(block: nn.Module, mode: BlockMode) -> nn.Module`. §3.1.2.
- `checkpoint.py` — thin wrapper over `torch.utils.checkpoint.checkpoint` (use_reentrant=False). §3.1.2.
- `swap.py` — `SwappedBlock`: wraps the block's forward in a `torch.autograd.graph.saved_tensors_hooks` context so **every autograd-saved tensor** (not just the block output) is D2H-copied to a pinned-host slot on `_swap_stream` in forward and H2D-copied back on `_swap_stream` in backward, with cross-stream event handshake against the default compute stream. Pool + stream are injected post-construction via `attach_runtime`; wrapper lifetime spans one fwd+bwd pair, and memory accounting must charge the sum of saved-tensor bytes (activations, RNG state, intermediate tensors), not just the block output. §3.1.2.
- `swap_pool.py` — `ActivationSwapPool`: pinned-host slot pool sized to `n_swap × prefetch_depth × max_act_bytes`. Backed by one `PinnedHostMemory` allocation; slot acquire/release tracked Python-side. §3.1.2.
- `offload.py` — Option B path: keeps a non-persistent chunk's owning block under `BlockMode.NONE` (no recompute) by re-gathering the chunk for backward and offloading after fwd. See `BLOCK_MODE_OFFLOAD_DESIGN.md` §3 / §6 for the storage-ptr book-keeping and runtime hook contract.
- `layout_rules.py` — `assign_modes(n_swap, n_checkpoint, n_offload, N_block) -> BlockStrategyMap`. Swap-early / unopt-late / interleave; `n_offload` honors the unopt-late rule (`BLOCK_MODE_OFFLOAD_DESIGN.md` §5.1). §3.1.2.

### cost/ (M4)

- `runtime.py` — `estimate_runtime(cfg, trace, layout) -> float`. Implements **Eqs. 2–7**: `T_iter = T_fwd + max(T_bwd + T_gpu_optim, T_cpu_optim)`, per-chunk `max(compute, comm)` roofline. §3.3, App A.1.
- `memory.py` — `estimate_peak(cfg, trace, layout, block_map) -> int`. Implements **Eqs. 8–10** (op-walk) and **Eq. 11** (α = 1.10 fragmentation). Bumps at first op of each CKPT block. §3.3, App A.2.
- `bandwidth.py` — `effective_bw(cfg, hw) -> float`. Derates prefetch BW when `n_swap > 0`. §3.3.

### search/ (M4)

- `knobs.py` — `CostConfig` dataclass + `derive_bounds(trace, layout) -> Bounds(N_chunk, N_block, N_interval)`. §3.3.
- `exhaustive.py` — `search(trace, layout, capacity_bytes) -> SearchResult`. Enumerates the 5-axis tuple `(n_persist, n_buffer, n_swap, n_checkpoint, n_offload)` in memory-ascending order, prunes OOM, returns argmin(T_iter). The `n_offload` axis (Option B) is the outermost loop; see `BLOCK_MODE_OFFLOAD_DESIGN.md` §5 for the enumeration order. §3.3.

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
    n_persist: int       # chunks pinned on GPU
    n_buffer: int        # pre-allocated chunk buffers
    n_swap: int          # blocks using activation swap
    n_checkpoint: int    # blocks using gradient checkpointing
    n_offload: int = 0   # blocks using BlockMode.OFFLOAD (Option B; see BLOCK_MODE_OFFLOAD_DESIGN.md)

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
- Example YAML: `examples/protrain/3090-8b-lora.yml` — opts in via `plugins: [axolotl.integrations.protrain.ProTrainPlugin]`

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

ProTrain is a per-rank memory policy. Two composition variants are supported (described first below as the explicit-override "pre-auto-mode" pair, then re-indexed in the auto-mode section as A/B/C); choose per-deployment by the `protrain_zero3_shard` YAML flag or by auto-detection. Note: the labels in the explicit-override pair below ("Composition-1 / Composition-2") are local to this subsection and do NOT correspond to the "Mode A / Mode B / Mode C" labels used by the auto-selector further down — see the cross-reference at the end of each entry.

**Composition-1 — DDP composition (pre-M7, still supported).** Each rank runs its own full `protrain_model_wrapper` and holds a full (replicated) copy of every non-persistent chunk on pinned CPU. The trainer wraps the protrain'd module in `torch.nn.parallel.DistributedDataParallel`. DDP handles the cross-rank all-reduce on the trainable gradient set; ProTrain's internal per-param `all_reduce` is silenced via `skip_internal_grad_reduce=True` (auto-set when `post_trainer_create` detects a DDP wrap). This mode is what the M6 multi-GPU throughput test exercises with `force_all_persistent=True` at world_size=4 on 3090s. It is the right choice for LoRA on ~7B where the frozen base fits in fp16 on one card (no memory pressure), because DDP's bucketed allreduce is faster than ProTrain's per-param reduction. (In auto-mode terms below, this overlaps Mode A — `force_all_persistent=True`, `protrain_zero3_shard=false`.)

**Composition-2 — true ZeRO-3 chunk sharding (M7, new; `protrain_zero3_shard=true`).** Non-persistent chunks are partitioned across ranks on CPU: each rank holds only `ceil(chunk_bytes / world_size)` pinned bytes per chunk. Forward/backward sees the full chunk via `all_gather_into_tensor` at `ChunkManager.gather`; grads are reduced + partitioned via `reduce_scatter_tensor(op=AVG)` at `ChunkManager.reduce_grads_and_offload`. The CPU FusedAdam step runs only on the rank-local shard slice — each region's flat `shard_param` is the Adam target, updated in place; the next gather's `all_gather` propagates the update back to every rank's replicated GPU copy. (In auto-mode terms below, this is Mode C — sharded CPU-offload.)

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

Late-bind path: `plugin.post_trainer_create` calls `_remeasure_nccl_and_research(wrapped)` after Accelerate brings up dist. When `world_size > 1` and the cached trace's NCCL tables are empty, the helper measures NCCL on the live process group, splices the populated tables + actual world into the trace via `dataclasses.replace`, persists the updated trace under a new cache key (so the next multi-rank run hits it directly without re-measuring), and re-runs `search()` with the same layout + capacity + hardware profile. The chunk manager is NOT rebuilt — optimizer state slots are already wired into the trainer — so the running step uses the bootstrap config; if the post-NCCL search picks a different `cfg`/`block_map`, a WARN is logged and `WrappedModel.search_result` is overwritten so future cost-model-based decisions reflect real comm cost. Subsequent multi-rank runs hit the cache and pick the new config from the start. Mode A and Mode B remain unaffected since they don't consume the NCCL tables.

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

### Memory Allocation Strategy (App B.2 — WIRED)

**Status: both App B.2 components are wired across the chunk manager, buffer pool, and SWAP unpack path.**

App B.2 of the paper has **two distinct components**, each addressing a different allocator pathology:

1. **Single-stream GPU allocation routing** — the *GPU heap* concern. PyTorch's caching allocator keeps a per-stream free list, so cross-stream allocator reuse needs `record_stream` hand-holding or the allocator fragments. Wired via `SingleStreamAllocator`.
2. **Custom precise-size pinned-host allocator** — the *pinned host* concern. PyTorch's `torch.empty(pin_memory=True)` routes through `CUDAHostAllocator`, which rounds up to the next power of two. The paper explicitly calls this out: *"the default pinned memory allocator (CUDAHostAllocator) often over-allocates by rounding up to the nearest power of two, leading to significant memory waste."* Wired via `PinnedHostMemory`, which calls `cudaHostAlloc` directly via ctypes for an exact byte count.

#### Component 1: Single-Stream GPU Allocation

- **Paper's design.** App B.2 specifies that ProTrain routes *every* GPU allocation (chunk gather, prefetch buffers, optimizer state) through the *default* CUDA stream. PyTorch's caching allocator keeps a separate free list per stream, so a tensor freed on stream A cannot be reused for an allocation on stream B without `record_stream` hand-holding. Single-streaming the allocations gives the allocator one heap to amortize across all phases and avoids cross-stream dealloc-sync bookkeeping.

- **Heap routing vs. kernel scheduling.** App B.2 governs *which heap an allocation comes from*, not which stream a kernel runs on. The wire-up keeps the dedicated `_prefetch_stream` and `_swap_stream` for PCIe-vs-compute overlap (those streams are about *kernel launch ordering*) but routes the *allocations* underneath them through the default-stream heap via `SingleStreamAllocator`. Cross-stream tensor consumption stays correct because every wrapped allocation that hands a buffer to a non-default stream calls `tensor.record_stream(non_default_stream)` immediately after exiting the allocator context, defering allocator reuse until the consuming stream has retired the work.

- **Wired call sites.**
  - `chunk/buffer_pool.py::BufferPool.__init__` — pre-allocates every pool slot (n_buffer × S_chunk bytes) on the default-stream heap. **Highest-leverage single change** — pool slots are the dominant sustained GPU allocation in ProTrain. No `record_stream` needed: pool slots' lifetimes are owned by the pool and only return to the allocator at teardown.
  - `chunk/manager.py::_ensure_persistent_buffer` — long-lived persistent-chunk GPU buffers. No `record_stream` (long-lived).
  - `chunk/manager.py::_empty_placeholder` — cached zero-element `param.data` sentinel. No `record_stream` (process-lived, not a kernel consumer).
  - `chunk/manager.py::_gather_sharded` — per-region `my_shard_gpu` and `gather_scratch` scratch tensors. **Critical wrap** — this method is called from `Scheduler._gather_on_prefetch_stream` inside `with torch.cuda.stream(self._prefetch_stream):`. Without the wrap, scratch tensors would land on the prefetch-stream heap and fragment the allocator. `record_stream(current_stream)` discipline applied: the scratch buffers are tied to whichever stream is actually consuming them (the prefetch stream in steady-state, the default stream in synchronous fallback).
  - `chunk/manager.py::_reduce_scatter_and_offload_shard` — per-region `region_grad` and `my_shard_grad_gpu` scratch tensors. Defensively wrapped with `record_stream`-when-needed: today this method runs on the default stream (called from `Scheduler.post_block_backward` which does not establish a stream context), so the wrap is a no-op there. Wrap kept so a future caller from a non-default stream stays correct without changes.
  - `chunk/manager.py::restore_to_gpu` — every per-slot, per-region, and persistent-chunk teardown allocation routes through a per-call `_alloc_empty(shape, dtype)` helper. Teardown runs on the default stream, so no `record_stream` needed.
  - `block/swap.py::unpack_from_pool` — `gpu_buf` activation swap-in buffer wrapped in `SingleStreamAllocator()` (commit `55e47da5`). The existing `gpu_buf.record_stream(handle.swap_stream)` inside the swap-stream H2D context provides the required cross-stream lifetime tie. The wrap lands AFTER the SWAP gate's `mem_get_info` headroom check + `RuntimeError` raise (`3f74f80c`), so the gate's failure path is unaffected.

- **`record_stream` discipline (contract for future contributors).** Any time you allocate a buffer inside `SingleStreamAllocator()` and then hand it to a non-default stream (prefetch stream, swap stream, etc.), you MUST call `buf.record_stream(non_default_stream)` immediately after exiting the allocator context. Skipping this is silent-data-corruption-class: the caching allocator will reuse the storage as soon as the default stream's pending work retires, even while the non-default stream is still reading or writing the bytes. Long-lived buffers (pool slots, persistent chunks, process-lived sentinels) are exempt — their lifetime is bounded by the manager, not by stream completion.

#### Component 2: Custom Pinned-Host Allocator

- **Paper's design.** PyTorch's `torch.empty(pin_memory=True)` routes through `CUDAHostAllocator`, which rounds the requested byte count up to the next power of two. For a 24 MB chunk that's a 32 MB allocation; for the trailing chunk of a 7B-param model the round-up can waste tens of MB across the offload set. ProTrain implements its own pinned allocator (`chunk/pinned_alloc.py::PinnedHostMemory`) that calls `cudaHostAlloc` directly via `ctypes` with the exact byte count, avoiding the rounding waste entirely.

- **PinnedHostMemory contract.** `PinnedHostMemory(n_buffer, S_chunk)` allocates `n_buffer × S_chunk` bytes pinned-host. `buffer(i)` returns a zero-copy `torch.Tensor` view over slot `i`; `release_buffer(i)` decrements the borrow refcount. `close()` raises if any borrow is still outstanding (use-after-free guard). The `__del__` path leaks rather than free under outstanding borrows, on the basis that a destructor-time leak is preferable to a dangling-pointer free. If `libcudart` cannot be loaded via `ctypes`, the allocator falls back to `torch.empty(size, pin_memory=True)` and exposes `is_precise_size = False` so tests can detect the regression.

- **Wired call sites (pinned host).**
  - `chunk/buffer_pool.py::BufferPool.__init__` — backing pinned-host region for the GPU buffer pool's H2D staging slots (`n_buffer × S_chunk`). One `PinnedHostMemory` per pool.
  - `chunk/manager.py::materialize_offload` — TWO unified `PinnedHostMemory` regions per manager: one for every non-persistent chunk's param shadow (replicated) or per-rank shard bytes (sharded), one for trainable-param grad shadows. Sized to the precise sum of per-chunk aligned bytes plus a 16-byte inter-chunk alignment pad. Per-chunk views into the pools are `narrow()` slices; the BUG 2 intra-chunk dtype-region alignment is preserved per-chunk under the unified layout. Closed via `_close_cpu_pools` from `restore_to_gpu` (deterministic teardown) or `__del__` (GC safety net). See `tests/protrain/test_chunk_manager_offload.py::test_materialize_offload_uses_precise_pinned_pool` for the precise-sizing assertion.
  - `block/swap_pool.py::ActivationSwapPool` — backing pinned-host region for activation swap slots (`n_swap × prefetch_depth × max_act_bytes`). One `PinnedHostMemory` per pool.

- **Allocation sites still on `torch.empty(pin_memory=True)` (unintentional).** *None* in the wired ProTrain runtime as of this commit. If a follow-up adds a new pinned-host allocation site it should default to `PinnedHostMemory` for paper fidelity.

#### Measurement status

Peak-memory delta from the wire-up has not been measured on RTX 3090 reference hardware in this commit (the `α = 1.10` fragmentation factor — item 1 above — was already absorbing the un-wired fragmentation cost in the cost model). To-be-measured in a follow-up: re-run the M1 profiler ground-truth before and after the wire-up; if peak drops by more than ~5% on a 1.5B-param target shape, recalibrate `α` downward. The single-stream wire-up's correctness — the `record_stream` discipline at every cross-stream site — has been validated by the new `tests/protrain/test_single_stream_allocator.py` test (heap-affinity assertion via free-then-reallocate fragmentation probe + nested-stream context-manager composition test). The pinned-host wire-up's correctness — total pool bytes equals the sum of per-chunk aligned bytes — is asserted by `tests/protrain/test_chunk_manager_offload.py::test_materialize_offload_uses_precise_pinned_pool`.
