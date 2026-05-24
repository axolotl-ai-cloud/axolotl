"""Shared data types for the ProTrain memory manager."""

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
    # Suppress on-demand gate; honour caller's Mode-A intent.
    force_all_persistent: bool = False
    # None = auto-detect from torch.distributed; explicit int sanity-checked.
    world_size: int | None = None


@dataclass(frozen=True)
class ProfilerTrace:
    """Serializable single-iteration trace. Cache key: (arch_hash, bs, seq, sku, world)."""

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

    # Per-op latencies (s) keyed by op_id; 0.0/missing → roofline fallback in cost/runtime.
    op_latencies: dict[OpId, float] = field(default_factory=dict)

    # Measured Adam throughput (bytes/sec); 0.0 → hardcoded prior fallback.
    cpu_adam_bytes_per_sec: float = 0.0
    gpu_adam_bytes_per_sec: float = 0.0

    # Hook-dispatch calibration: steady_fwd_wall_s / hooked_fwd_wall_s scales per-op latencies.
    hooked_fwd_wall_s: float = 0.0
    steady_fwd_wall_s: float = 0.0
    steady_bwd_wall_s: float = 0.0
    # Hook-less steady-forward max_memory_allocated; ground-truth fwd peak floor.
    steady_fwd_peak_bytes: int = 0

    # Per-block hook-less forward peaks; max bounds any NONE/CKPT/SWAP mix.
    steady_fwd_block_peak_bytes: dict[BlockId, int] = field(default_factory=dict)

    # Backward-aware peaks; 0/empty when trace ran without backward.
    steady_bwd_peak_bytes: int = 0
    steady_bwd_block_peak_bytes: dict[BlockId, int] = field(default_factory=dict)

    # fp16 TFLOPS on trace SKU for cross-SKU per-op scaling; 0.0 → DEFAULT_COMPUTE_RATE_TFLOPS.
    compute_rate_tflops: float = 0.0

    # requires_grad fraction; low fractions (LoRA ~0.1%) get tighter bwd/fwd ratio fallback.
    trainable_param_fraction: float = 0.0

    # Phase-2 chunked-runtime measurement; 0.0 → v8 fallback path.
    steady_bwd_chunked_wall_s: float = 0.0
    steady_step_overlap_s: float = 0.0
    steady_phase2_peak_bytes: int = 0
    phase2_n_persist: int = 0
    phase2_n_buffer: int = 0
    phase2_n_checkpoint: int = 0
    phase2_per_block_recompute_s: float = 0.0

    # Phase-2 chunked forward wall; overrides per-op-latency path at boot cfg.
    steady_fwd_chunked_wall_s: float = 0.0

    # Phase-2 analytical baseline for cfg-delta calibration.
    phase2_iter_s: float = 0.0
    phase2_analytical_iter_s: float = 0.0
    phase2_analytical_peak_bytes: int = 0

    # Per-component analytical baselines (alphafwd / alphabwd / alphaopt calibration).
    phase2_fwd_s: float = 0.0
    phase2_bwd_s: float = 0.0
    phase2_step_s: float = 0.0
    phase2_analytical_fwd_s: float = 0.0
    phase2_analytical_bwd_s: float = 0.0
    phase2_analytical_step_s: float = 0.0

    # Residual whole-iter overhead anchor; alpha_residual = phase2_iter_s / per_comp_pred.
    phase2_per_comp_pred_iter_s: float = 0.0

    # block_id → tree.forward_order; empty falls back to module_path prefix parse.
    block_tree_index: dict[BlockId, int] = field(default_factory=dict)

    # Architecture metadata for the per-block CKPT internal-residual proxy
    # (cost/memory._block_internal_saved_bytes). 0 = unknown (legacy traces);
    # the residual term degrades to the pre-fix block-output-only chain.
    hidden_size: int = 0
    num_attention_heads: int = 0
    intermediate_size: int = 0


# ---------------------------------------------------------------------------
# Chunk layout (§3.1.1, App B.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkLayout:
    """Per-rank chunk assignment plus intra-chunk ordering."""

    S_chunk: int  # bytes per chunk
    N_chunk: int  # total chunks
    chunks: tuple[tuple[ParamId, ...], ...]  # exec-order within each chunk
    param_to_chunk: dict[ParamId, ChunkId]
    block_to_chunks: dict[BlockId, tuple[ChunkId, ...]]
    # Chunks runtime must keep GPU-resident regardless of n_persist (non-block params).
    mandatory_persistent: frozenset[ChunkId] = field(default_factory=frozenset)

    def effective_persistent_ids(self, n_persist: int) -> frozenset[ChunkId]:
        """Return ``{0..n_persist-1} | mandatory_persistent`` as a frozenset."""
        n = max(0, min(int(n_persist), int(self.N_chunk)))
        # Lazy per-instance memo; layout is conceptually immutable, only the
        # block_to_chunks dict breaks the @dataclass(frozen=True) hash. The
        # cache is keyed on clamped n only so it's safe across callers.
        cache: dict[int, frozenset[ChunkId]] | None = self.__dict__.get(
            "_effective_persistent_cache"
        )
        if cache is None:
            cache = {}
            object.__setattr__(self, "_effective_persistent_cache", cache)
        hit = cache.get(n)
        if hit is not None:
            return hit
        prefix = {ChunkId(i) for i in range(n)}
        result = frozenset(prefix | set(self.mandatory_persistent))
        cache[n] = result
        return result


# ---------------------------------------------------------------------------
# Cost / search (§3.3, App A)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostConfig:
    """The five tunable knobs (n_persist, n_buffer, n_swap, n_checkpoint, n_offload)."""

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
    predicted_init_transient_peak_bytes: int = 0


# ---------------------------------------------------------------------------
# Hardware profile (§3.2, §7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareProfile:
    """Static hardware description consumed by the searcher."""

    gpu_sku: str
    gpu_memory_bytes: int
    gpu_count: int  # world size for this run
    pcie_h2d_bps: float
    pcie_d2h_bps: float
    has_nvlink: bool  # informational; we never use NVLink paths
    zero3_shard: bool = False  # True when M7 chunk-sharding is active
    # Measured Adam throughput (bytes/sec); 0.0 → hardcoded prior fallback.
    cpu_adam_bytes_per_sec: float = 0.0
    gpu_adam_bytes_per_sec: float = 0.0
    # Live fp16 TFLOPS for cross-SKU scaling; 0.0 → identity scale.
    gpu_compute_tflops: float = 0.0
    # Per-dtype alpha lookup; bnb-4-bit Params4bit maps to 0.5 (packed).
    dominant_param_bytes_per_element: float = 2.0


# ---------------------------------------------------------------------------


@dataclass
class WrappedModel:
    """Opaque handle returned by ``protrain_model_wrapper`` (chunk_manager + scheduler + hooks)."""

    module: "nn.Module"  # the original model, with hooks installed
    search_result: SearchResult
    chunk_manager: object = None
    scheduler: object = None
    _hook_handles: list[object] = field(default_factory=list, repr=False)
    _closed: bool = field(default=False, repr=False)

    def close(self) -> None:
        """Tear down every wrapper-owned resource (hooks → scheduler → chunk_manager). Idempotent."""
        if self._closed:
            return
        self._closed = True

        # Local import to avoid pulling logging deps at module import time.
        from axolotl.utils.logging import get_logger

        log = get_logger(__name__)

        for handle in self._hook_handles:
            try:
                handle.remove()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 — best-effort
                log.debug("WrappedModel.close: hook handle.remove failed: %s", exc)
        self._hook_handles = []

        scheduler = self.scheduler
        if scheduler is not None:
            try:
                scheduler.close()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 — best-effort
                log.debug("WrappedModel.close: scheduler.close failed: %s", exc)
            self.scheduler = None

        chunk_manager = self.chunk_manager
        if chunk_manager is not None:
            try:
                chunk_manager.close()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 — best-effort
                log.debug("WrappedModel.close: chunk_manager.close failed: %s", exc)
            self.chunk_manager = None


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
