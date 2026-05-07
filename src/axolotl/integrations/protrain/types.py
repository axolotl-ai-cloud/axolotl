"""Shared data types for the ProTrain memory manager.

Pure data shapes only — no runtime logic, no torch tensors allocated at import
time. Every downstream subpackage (profiler, chunk, block, cost, search,
runtime, api) depends on this module. Keeping it allocation-light lets the
subpackages develop in parallel against a stable contract.

Paper references: MLSys 2026, arXiv 2406.08334 (§3.1–3.3, Appendix A–B).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, NewType

if TYPE_CHECKING:
    from torch import nn


# ---------------------------------------------------------------------------
# Identifier aliases
# ---------------------------------------------------------------------------

# Dotted path from `model.named_parameters()`, e.g. "layers.0.attn.q_proj.weight".
# Stable across pickling, debuggable, and what all profiler/chunk modules key on.
ParamId = NewType("ParamId", str)

# Monotonic op index during the profiler's single-iteration trace.
OpId = NewType("OpId", int)

# Transformer block index, 0 .. N_block-1.
BlockId = NewType("BlockId", int)

# Chunk index, 0 .. N_chunk-1.
ChunkId = NewType("ChunkId", int)


# ---------------------------------------------------------------------------
# Block modes (§3.1.2)
# ---------------------------------------------------------------------------


class BlockMode(str, Enum):
    """Activation strategy selected per transformer block."""

    NONE = "none"  # keep activations on GPU, no checkpoint, no swap
    CKPT = "ckpt"  # drop + recompute in backward
    SWAP = "swap"  # offload to CPU in forward, prefetch in backward (feature-flagged)
    OFFLOAD = "offload"  # param-offload-aware NONE-equivalent for non-persistent chunks


# Per-block mode selection, output of `block.layout_rules.assign_modes`.
BlockStrategyMap = dict[BlockId, BlockMode]


# ---------------------------------------------------------------------------
# Profiler inputs + outputs (§3.2, App A.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpRecord:
    """One op captured during the profiler trace."""

    op_id: OpId
    module_path: str  # dotted nn.Module path owning this op
    qualified_name: str  # e.g. "aten::addmm", "prim::Constant"
    shape_signature: tuple[tuple[int, ...], ...]  # input tensor shapes
    block_id: BlockId | None  # transformer block, if inside one
    is_forward: bool  # True for fwd, False for bwd


@dataclass(frozen=True)
class ProfilerConfig:
    """Arguments to `profiler.trace.run_trace`."""

    batch_size: int
    seq_len: int
    device: str  # e.g. "cuda:2"
    include_backward: bool = True
    on_demand: bool = True  # OnDemandTensorMgr for models > single-GPU
    # Distributed world size. ``None`` (default) means "auto-detect" — the
    # tracer probes ``torch.distributed.get_world_size()`` if a process
    # group is initialized and falls back to 1 otherwise. Pass an explicit
    # int to force a specific size (sanity-checked against the live group
    # by ``measure_nccl``).
    world_size: int | None = None


@dataclass(frozen=True)
class ProfilerTrace:
    """Serializable single-iteration trace. Cache key: (arch_hash, bs, seq, sku, world).

    Re-profile triggers: any change to model arch, batch_size * seq_len, GPU SKU or
    count, PCIe/NVLink topology (§7).
    """

    # Operator trace
    op_order: tuple[OpRecord, ...]
    intra_op_delta: dict[OpId, int]  # bytes; peak_during_op - allocated_before_op
    inter_op_delta: dict[OpId, int]  # bytes; peak_between_hooks - allocated_prev_end

    # Per-block summaries
    activation_sizes: dict[BlockId, int]  # retained-activation bytes per block

    # Model-state constants (constant across the run given the model + dtype config)
    model_state_bytes: int  # fp16 params + grads + fp32 master + momentums

    # Hardware microbenchmarks (§3.2 hardware profiling)
    pcie_h2d_bps: float
    pcie_d2h_bps: float
    nccl_gather_s: dict[int, float]  # keyed by payload size in bytes
    nccl_reduce_s: dict[int, float]

    # Cache key components
    arch_hash: str  # deterministic hash of model architecture
    bs: int
    seq: int
    sku: str  # torch.cuda.get_device_name() result
    world: int  # world_size at profile time

    # Per-op wall-clock latencies (seconds), measured via torch.cuda.Event during
    # the same single-iteration trace. Keys match ``op_order[i].op_id``. Populated
    # for forward ops and for the synthetic ``<backward>`` op that stands in for
    # the aggregate backward pass. Consumed by ``cost/runtime.py`` to replace the
    # activation-bytes compute-rate proxy with measured per-block compute time.
    # Optional: traces predating this field deserialize with an empty dict, in
    # which case ``cost/runtime.py`` falls back to the roofline proxy and logs a
    # warning. New in TRACE_VERSION=2 (see profiler/cache.py).
    op_latencies: dict[OpId, float] = field(default_factory=dict)

    # Measured CPU / GPU Adam throughput (bytes/sec) from the hw_bench
    # microbenchmarks. Replaces the hardcoded ``_CPU_ADAM_BYTES_PER_SEC``
    # / ``_GPU_ADAM_BYTES_PER_SEC`` priors in ``cost/runtime.py``. 0.0
    # means "unavailable" — the cost model falls back to a hardcoded
    # prior and logs a warning. New in TRACE_VERSION=3.
    cpu_adam_bytes_per_sec: float = 0.0
    gpu_adam_bytes_per_sec: float = 0.0

    # Hook-dispatch calibration fields — new in TRACE_VERSION=4.
    #
    # The profiler installs pre/post forward hooks on every ``nn.Module`` to
    # record per-op memory deltas + latencies. On transformer-sized models
    # (~1000 leaf modules) the hook dispatch alone inflates measured forward
    # wall time ~2.5x over a steady-state (hook-less) forward. The cost
    # model consumes this ratio to scale the hooked per-op latencies down
    # to a realistic prior:
    #
    #   scale = steady_fwd_wall_s / hooked_fwd_wall_s
    #   t_fwd_calibrated = sum(per_block_latencies) * scale
    #
    # ``hooked_fwd_wall_s`` is the total wall-clock of the hooked forward
    # (measured via a ``torch.cuda.Event`` pair around the full forward
    # pass, NOT summed from per-op latencies — that sum misses inter-op
    # Python overhead).
    #
    # ``steady_fwd_wall_s`` is the same forward measured BEFORE hooks are
    # installed, on the same warm model + batch, with a pair of un-hooked
    # warmup passes first so allocator state is representative.
    #
    # ``steady_bwd_wall_s`` is the hook-less backward wall-clock, captured
    # on a separately-timed un-hooked backward (optional; 0.0 means
    # "unavailable" — the cost model falls back to ``bwd_fwd_ratio`` of
    # the scaled forward).
    #
    # Traces loaded from cache that predate v4 have 0.0 defaults here; the
    # cost model detects the 0.0 and falls back to the unscaled per-op
    # sum (identity scale factor), preserving backward compatibility until
    # the cache is refreshed.
    hooked_fwd_wall_s: float = 0.0
    steady_fwd_wall_s: float = 0.0
    steady_bwd_wall_s: float = 0.0
    # ``steady_fwd_peak_bytes`` is ``torch.cuda.max_memory_allocated()``
    # captured across the hook-less steady forward pass. Used by the
    # memory cost model as a ground-truth floor on the forward
    # contribution — eliminates the search's "retained-NONE-activations"
    # over-estimate when a hot-iter measurement is available. 0 means
    # unavailable (pre-v5 cached traces, or CUDA unavailable at profile
    # time).
    steady_fwd_peak_bytes: int = 0

    # Per-block peak bytes captured during the hook-less steady forward.
    # Lightweight forward pre/post hooks installed ONLY at block level (tens
    # of blocks, not the ~1000 leaves the main profiling path targets) call
    # ``torch.cuda.reset_peak_memory_stats`` before each block and read
    # ``torch.cuda.max_memory_allocated`` after. Keys are global transformer-
    # block indices discovered via ``flatten_block_trees(discover_blocks(...))``
    # — encoder blocks own ids ``[0, n_enc)``, decoder blocks own ids
    # ``[n_enc, n_enc + n_dec)`` on encoder-decoder models; values are
    # per-block peak bytes observed during that block's forward.
    #
    # The memory cost model consumes ``max(steady_fwd_block_peak_bytes.values())``
    # as a ground-truth upper bound on the FORWARD peak for any NONE/CKPT/SWAP
    # mix — unlike ``steady_fwd_peak_bytes`` (which is an aggregate only valid
    # for all-NONE configs), the per-block max bounds any fractional-NONE
    # config too: CKPT/SWAP blocks free their activations before the next
    # block runs, so the forward peak across a mixed configuration cannot
    # exceed the max per-block peak observed during the all-NONE profile.
    # Backward CKPT recomputation bumps are added on top because they occur
    # during backward and weren't measured here.
    #
    # Empty dict means unavailable (pre-v6 cached traces, or CUDA unavailable
    # at profile time). New in TRACE_VERSION=6.
    steady_fwd_block_peak_bytes: dict[BlockId, int] = field(default_factory=dict)

    # ----- Backward-aware peak measurements (TRACE_VERSION 19) -----
    #
    # The paper's profiler is explicitly backward-aware (§3.2 / App A.2):
    # backward dominates peak memory because retained activations,
    # gradient buffers, and CKPT recompute windows all overlap during
    # the backward pass. Pre-v19 ``run_trace`` ran the profiler with
    # ``include_backward=False`` to avoid OOM on 7B-class single-3090
    # callers; the cost model's bwd peak then degraded to an analytical
    # estimate derived from forward measurements only.
    #
    # ``steady_bwd_peak_bytes`` is the cumulative
    # ``torch.cuda.max_memory_allocated`` observed across the hook-less
    # steady backward pass — captured in the same 4-iter hot loop that
    # produces ``steady_bwd_wall_s``. It bounds the BACKWARD peak from
    # below (cannot be lower than what the un-CKPT-ed bootstrap actually
    # used) and the cost model uses it as a sanity check against
    # estimated peaks: a candidate config whose modeled peak is below
    # this measured floor is over-optimistic.
    #
    # ``steady_bwd_block_peak_bytes`` mirrors ``steady_fwd_block_peak_bytes``
    # for backward: per-block peaks captured via lightweight
    # ``register_full_backward_hook`` pairs around each transformer
    # block. Future cost-model calibration can derive a per-block
    # backward bump from these values; today they are recorded for
    # telemetry / future use.
    #
    # Both fields are 0 / empty when:
    #   - the trace ran with ``include_backward=False``,
    #   - on-demand mode engaged (steady-state is skipped entirely),
    #   - CUDA was unavailable, or
    #   - the steady-state backward iter raised (e.g. analytical fallback
    #     fired), in which case ``steady_bwd_block_peak_bytes`` is also
    #     cleared so the recorded set is internally consistent.
    #
    # Pre-v19 traces deserialize with the empty / zero defaults; the
    # cost model preserves its existing v18 behaviour in that case.
    steady_bwd_peak_bytes: int = 0
    steady_bwd_block_peak_bytes: dict[BlockId, int] = field(default_factory=dict)

    # Sustained fp16 compute throughput (TFLOPS) on the trace SKU, measured
    # by ``profiler.hw_bench.measure_compute_rate``. Consumed by
    # ``cost/runtime.py`` to scale per-op latencies when the live training
    # device's SKU differs from the cached trace's SKU — e.g. trace captured
    # on 3090 Ti, replayed on plain 3090. Same-SKU traces see ``scale ≈ 1.0``
    # and the calibration is a no-op. ``0.0`` means unavailable (pre-v8
    # caches, CUDA unavailable, or measurement failed); the cost model
    # then falls back to ``hw_bench.DEFAULT_COMPUTE_RATE_TFLOPS``. New in
    # TRACE_VERSION=8.
    compute_rate_tflops: float = 0.0

    # Fraction of model parameters with ``requires_grad=True`` at trace time
    # (range [0.0, 1.0]). LoRA / adapter training has very low trainable
    # fractions (~0.1% on 7B-LoRA-r8) — backward compute is then ~1× forward
    # rather than the canonical 2× full-finetune ratio, because autograd
    # skips frozen subgraphs. The cost model's ``_bwd_compute_time_from_trace``
    # consults this fraction to pick a tighter fallback ratio when the
    # measured ``steady_bwd_wall_s`` is unavailable (7B-class profiler runs
    # OOM the backward without chunk offload engaged). 0.0 means unmeasured
    # (pre-v8) — falls back to the canonical 2× ratio. New in TRACE_VERSION=8.
    trainable_param_fraction: float = 0.0

    # ----- Phase-2 chunked-runtime measurements (TRACE_VERSION 10) -----
    #
    # The phase-2 profiler runs a short chunked steady-state fwd+bwd+step
    # loop INSIDE ``protrain_model_wrapper`` (after the initial trace +
    # initial search but before returning the wrapped model). It measures
    # backward time with the chunk manager engaged — closing the gap that
    # forced ``include_backward=False`` on 7B+ profiles where the
    # unwrapped backward OOMs.
    #
    # ``steady_bwd_chunked_wall_s`` is the median measured backward
    # wall-clock under the bootstrap config, in seconds. Includes
    # gradient checkpoint recompute for ``phase2_n_checkpoint`` blocks
    # plus any chunk-gather / reduce-offload overhead inherent to the
    # chunked path. The cost model translates this into a config-
    # independent base via:
    #
    #     base_bwd = steady_bwd_chunked_wall_s
    #              - phase2_n_checkpoint * phase2_per_block_recompute_s
    #     predicted_bwd(cfg) = base_bwd + k_ckpt(cfg) * per_block_compute(cfg)
    #
    # where ``k_ckpt(cfg)`` is the count of CKPT blocks in the candidate's
    # block_map. The translation handles the case where the post-research
    # search picks a different ``n_checkpoint`` than the bootstrap's
    # measurement (the common case — phase-2 reveals real backward cost
    # and the search may switch some blocks from CKPT to NONE).
    #
    # ``steady_step_overlap_s`` is the wall-clock window where backward
    # compute and the optimizer step overlap, captured via
    # ``torch.cuda.Event`` pairs around the bwd→step transition. The
    # cost model does not consume this directly today (the paper's
    # T_iter = T_FWD + max{T_BWD + T_GPU_OPT, T_CPU_OPT} accounts for
    # overlap implicitly), but it's recorded for future cost-model
    # tuning + telemetry validation.
    #
    # ``steady_phase2_peak_bytes`` records the CUDA high-water mark
    # during the same chunked measurement. When the final post-phase-2
    # config matches ``phase2_n_persist`` / ``phase2_n_buffer`` /
    # ``phase2_n_checkpoint``, the wrapper can use this as a measured
    # peak calibration instead of the analytical CKPT op-walk bound.
    #
    # These fields default to 0.0 / 0; the cost model treats 0.0 in
    # ``steady_bwd_chunked_wall_s`` as "no phase-2 measurement available"
    # and falls back to the v8 path (``steady_bwd_wall_s`` ratio →
    # trainable-fraction heuristic → 2× canonical).
    steady_bwd_chunked_wall_s: float = 0.0
    steady_step_overlap_s: float = 0.0
    steady_phase2_peak_bytes: int = 0
    phase2_n_persist: int = 0
    phase2_n_buffer: int = 0
    phase2_n_checkpoint: int = 0
    phase2_per_block_recompute_s: float = 0.0

    # ----- Phase-2 chunked-runtime forward measurement (TRACE_VERSION 11) -----
    #
    # ``steady_fwd_chunked_wall_s`` is the median measured forward
    # wall-clock under the bootstrap config, captured by the same
    # phase-2 measurement loop that produces ``steady_bwd_chunked_wall_s``.
    # Forward time under the chunk manager includes any
    # chunk-prefetch / gather overhead that's inherent to the chunked
    # runtime AND the actual fused-kernel forward compute — closing the
    # forward over-prediction gap left over after phase-2 backward
    # calibration.
    #
    # Unlike the backward, the forward cost is approximately
    # config-independent at the cost-model level: forward never
    # recomputes (recompute happens in backward for CKPT blocks), so
    # there's no per-cfg adjustment to apply on top of the measurement.
    # The cost model simply uses ``steady_fwd_chunked_wall_s`` directly
    # as the forward-compute total when populated:
    #
    #     t_fwd_compute_total = steady_fwd_chunked_wall_s   (overrides
    #         the per-op-latency sum + hook-scale + roofline cap path)
    #
    # Per-block compute distribution is preserved from the per-op path
    # without rescaling. The aggregate chunked wall replaces the forward
    # total directly, while the per-block shape remains the recompute
    # basis for CKPT accounting.
    #
    # ``0.0`` (default) means "no phase-2 forward measurement
    # available" and the cost model falls back to the v10 path
    # (per-op-latency sum with hook scale + roofline cap).
    steady_fwd_chunked_wall_s: float = 0.0

    # ----- Phase-2 analytical-baseline calibration (TRACE_VERSION 20) -----
    #
    # When phase-2 measures the bootstrap config's wall and peak it gives
    # us a measurement-anchored *absolute* time/size scale at one cfg
    # point. To translate that to a different production cfg whose
    # ``cfg.n_swap > 0`` (or whose chunked-wall override is otherwise
    # gated off), the cost model needs to know what its own analytical
    # path WOULD have predicted at the bootstrap cfg — that is the
    # baseline against which the measurement-vs-analytical RATIO is
    # well-defined and can be carried across cfgs.
    #
    # ``phase2_iter_s`` is the measured fwd+bwd+step iter wall at the
    # bootstrap cfg (sum of medians from ``measure_chunked_steady``).
    # ``phase2_analytical_iter_s`` is the analytical (non-phase-2)
    # ``estimate_runtime`` prediction at the SAME bootstrap cfg,
    # captured pre-splice so the chunked-wall override does not
    # short-circuit the analytical path.
    # The cost model derives a multiplicative scale
    # ``α = phase2_iter_s / phase2_analytical_iter_s`` and applies it to
    # any analytical-path prediction. When the analytical path is not
    # taken (e.g. ``cfg.n_swap == 0`` and chunked walls populated) α is
    # not consulted — the chunked-wall override is already absolute.
    #
    # ``phase2_analytical_peak_bytes`` plays the analogous role for peak
    # memory: ``estimate_peak`` evaluated at the bootstrap cfg pre-splice.
    # Combined with the measured ``steady_phase2_peak_bytes`` it lets the
    # post-search peak calibration apply a CFG-DELTA correction
    # (``floor = phase2_peak + max(0, peak_analytical(prod_cfg)
    # - phase2_analytical_peak_bytes)``) so the measurement-anchored
    # floor stays in force even when the searcher picks a production
    # cfg different from the bootstrap.
    #
    # All three fields default to 0 / 0.0 — that is the "no phase-2
    # baseline available" sentinel that collapses both calibrations to
    # their pre-refactor behaviour (no α scaling on the runtime side;
    # only the same-cfg measurement window on the peak side).
    phase2_iter_s: float = 0.0
    phase2_analytical_iter_s: float = 0.0
    phase2_analytical_peak_bytes: int = 0

    # ----- Block -> tree-index registry (TRACE_VERSION 16) -----
    #
    # Maps each global ``BlockId`` to its forward-order tree index
    # (encoder=0, decoder=1; single-tree causal-LM models use 0
    # exclusively). Captured at trace-construction time by walking the
    # ``BlockTree`` list returned by
    # :func:`axolotl.integrations.protrain.block.layout_rules.discover_blocks`
    # and emitting ``block_id -> tree.forward_order`` for every block
    # in flatten order. Persisting this map removes the cost model's
    # need to parse ``OpRecord.module_path`` prefixes (``encoder.``,
    # ``decoder.``) — that string-prefix path is brittle for any future
    # enc-dec family with non-``encoder``/``decoder`` naming.
    #
    # Empty dict (default) means "unavailable" — the cost model falls
    # back to the legacy module_path prefix parse for traces predating
    # this field (degenerate test inputs that construct a
    # ``ProfilerTrace`` directly without populating it). Cached traces
    # written by an older code path are invalidated by the
    # TRACE_VERSION bump.
    block_tree_index: dict[BlockId, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chunk layout (§3.1.1, App B.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkLayout:
    """Per-rank chunk assignment plus intra-chunk ordering. Output of M2 layout pass.

    ``mandatory_persistent`` records ChunkIds that the *runtime* must keep
    GPU-resident regardless of the user-chosen prefix ``cfg.n_persist``.
    These are typically chunks that contain at least one non-block param
    (e.g. ``model.norm.weight`` in a Llama tail or an untied lm_head):
    such params live outside any block's ``layout.block_to_chunks`` entry,
    so the block-granularity scheduler never gathers them on a hook —
    if offloaded they would be zero-sized when post-block forward consumes
    them. Pinning them is a runtime correctness requirement, NOT part of
    the search's prefix budget.

    The paper's persistent-set is a prefix ``[0, n_persist)`` (line 188
    of the source paper); ``mandatory_persistent`` is the local
    integration's correctness extension. Cost model + search keep
    ``cfg.n_persist`` strictly meaning "prefix length the search chose";
    the runtime resident set is ``{0..n_persist-1} ∪ mandatory_persistent``.

    The default is an empty frozenset so legacy ``ChunkLayout(...)``
    constructions stay drop-in compatible.
    """

    S_chunk: int  # bytes per chunk
    N_chunk: int  # total chunks
    chunks: tuple[tuple[ParamId, ...], ...]  # exec-order within each chunk
    param_to_chunk: dict[ParamId, ChunkId]
    block_to_chunks: dict[BlockId, tuple[ChunkId, ...]]
    # Chunks that MUST stay GPU-resident regardless of n_persist; see
    # docstring for the runtime-correctness motivation. ``frozenset`` so
    # the layout stays hashable + immutable.
    mandatory_persistent: frozenset[ChunkId] = field(default_factory=frozenset)

    def effective_persistent_ids(self, n_persist: int) -> frozenset[ChunkId]:
        """Return ``{0..n_persist-1} ∪ mandatory_persistent`` as a frozenset.

        Single source of truth for "which chunks are GPU-resident under
        ``n_persist``" so the searcher, cost model, and runtime construction
        cannot disagree. Clamps ``n_persist`` defensively into
        ``[0, N_chunk]``.
        """
        n = max(0, min(int(n_persist), int(self.N_chunk)))
        prefix = {ChunkId(i) for i in range(n)}
        return frozenset(prefix | set(self.mandatory_persistent))


# ---------------------------------------------------------------------------
# Cost / search (§3.3, App A)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostConfig:
    """The five tunable knobs (§3.3 table + Option B §3.6).

    ``n_offload`` is the new Option B axis (see
    ``BLOCK_MODE_OFFLOAD_DESIGN.md`` §3.6 / §4.3). It defaults to 0
    so legacy 4-knob callers continue to construct identical
    configurations. The searcher's outer loop enumerates non-zero
    values; pre-Option-B producers (tests, model wrapper synth-cfg
    builders) keep working unchanged.
    """

    n_persist: int  # chunks pinned on GPU
    n_buffer: int  # pre-allocated chunk buffers
    n_swap: int  # blocks using activation swap
    n_checkpoint: int  # blocks using gradient checkpointing
    n_offload: int = 0  # blocks using BlockMode.OFFLOAD (Option B §3.6)


@dataclass(frozen=True)
class Bounds:
    """Upper bounds on the four knobs, derived from trace + layout."""

    N_chunk: int
    N_block: int
    N_interval: int  # swap-interval bound in compute units


@dataclass(frozen=True)
class SearchResult:
    """Output of `search.exhaustive.search`."""

    cfg: CostConfig
    block_map: BlockStrategyMap
    predicted_peak_bytes: int
    predicted_iter_s: float


# ---------------------------------------------------------------------------
# Hardware profile (§3.2, §7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareProfile:
    """Static hardware description consumed by the searcher.

    ProTrain is RTX 3090 / 3090 Ti scoped for this workstream — treat the two
    SKUs as equivalent when picking the target pool.

    The ``zero3_shard`` flag is plumbed from ``protrain_model_wrapper`` (which
    decides sharding on/off via the same auto-detect logic documented in
    ``DESIGN.md §Multi-GPU``) through to ``cost/memory.estimate_cpu_footprint``
    so per-rank CPU-pressure accounting reflects ZeRO-3 partitioning. It does
    NOT change the GPU peak estimate — the gather materializes the full chunk
    on GPU regardless of sharding — so ``estimate_peak`` ignores this field.
    """

    gpu_sku: str
    gpu_memory_bytes: int
    gpu_count: int  # world size for this run
    pcie_h2d_bps: float
    pcie_d2h_bps: float
    has_nvlink: bool  # informational; we never use NVLink paths
    zero3_shard: bool = False  # True when M7 chunk-sharding is active
    # Measured Adam throughput (bytes/sec). 0.0 means "unavailable" —
    # ``cost/runtime.estimate_runtime`` falls back to a hardcoded prior in
    # that case. Populated by
    # :func:`axolotl.integrations.protrain.profiler.hw_bench.measure_cpu_adam`
    # and ``measure_gpu_adam`` after :func:`run_trace` completes, then
    # plumbed into the HardwareProfile the searcher consumes. New in
    # TRACE_VERSION=3 (see profiler/cache.py).
    cpu_adam_bytes_per_sec: float = 0.0
    gpu_adam_bytes_per_sec: float = 0.0
    # Live compute rate (fp16 TFLOPS) on the training device, used to scale
    # cached traces captured on a different SKU. ``0.0`` means "unmeasured";
    # ``cost/runtime.py`` then assumes same-SKU and applies an identity
    # scale. Populated by ``profiler.hw_bench.measure_compute_rate`` from
    # the model_wrapper just before the searcher runs.
    gpu_compute_tflops: float = 0.0


# ---------------------------------------------------------------------------


@dataclass
class WrappedModel:
    """Opaque handle returned by `protrain_model_wrapper`.

    Owns: ChunkManager, BlockStrategyMap (via search_result), installed hooks, the
    chosen SearchResult, and the Scheduler. Mutable because it holds runtime state
    (hook handles, buffer pool). Concrete internal types are `object` here to keep
    this module pure data — see `chunk.manager`, `runtime.scheduler`, etc.
    """

    module: "nn.Module"  # the original model, with hooks installed
    search_result: SearchResult
    chunk_manager: object = None
    scheduler: object = None
    _hook_handles: list[object] = field(default_factory=list, repr=False)


__all__ = [
    "ParamId",
    "OpId",
    "BlockId",
    "ChunkId",
    "BlockMode",
    "BlockStrategyMap",
    "OpRecord",
    "ProfilerConfig",
    "ProfilerTrace",
    "ChunkLayout",
    "CostConfig",
    "Bounds",
    "SearchResult",
    "HardwareProfile",
    "WrappedModel",
]
