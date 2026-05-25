"""Public model-wrapper entry point for the ProTrain runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from torch import nn

from axolotl.integrations.protrain.block import (
    assign_modes,
    discover_blocks,
    flatten_block_trees,
    unwrap_block,
    wrap_block,
)
from axolotl.integrations.protrain.chunk import (
    BufferPool,
    ChunkManager,
    CpuFusedAdamAdapter,
    GpuFusedAdamAdapter,
    PinnedHostMemory,
    build_layout,
    pick_S_chunk,
)
from axolotl.integrations.protrain.cost.bandwidth import effective_bw
from axolotl.integrations.protrain.profiler import (
    load_cached_trace,
    run_trace,
    save_cached_trace,
)
from axolotl.integrations.protrain.profiler.cache import ProfilerCacheKey
from axolotl.integrations.protrain.profiler.hw_bench import measure_compute_rate
from axolotl.integrations.protrain.profiler.trace import (
    _arch_hash,
    synth_trace_from_overrides,
)
from axolotl.integrations.protrain.runtime.hooks import install_hooks
from axolotl.integrations.protrain.runtime.scheduler import Scheduler
from axolotl.integrations.protrain.search import search
from axolotl.integrations.protrain.search.exhaustive import (
    block_map_runtime_admissible,
    min_n_buffer_for,
)
from axolotl.integrations.protrain.types import (
    BlockId,
    ChunkId,
    CostConfig,
    HardwareProfile,
    ParamId,
    ProfilerConfig,
    SearchResult,
    WrappedModel,
)
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

LOG = get_logger(__name__)


# Reserve 2 GiB for CUDA context + allocator overhead.
_DEFAULT_HEADROOM_BYTES = 2 * (1 << 30)

# Slack for allocator frag, framework working set, dataloader workers.
_DEFAULT_CPU_HEADROOM_BYTES = 2 * (1 << 30)


def _sku(device: "torch.device | str") -> str:
    import torch

    try:
        return torch.cuda.get_device_name(device)
    except Exception:  # pragma: no cover — defensive, CPU-only lanes
        return "cpu"


def _detect_dominant_param_bytes_per_element(model: nn.Module) -> float:
    """Return the modal logical bytes-per-element across the model's params."""
    # bitsandbytes is optional.
    _Params4bit: type | None = None
    try:
        import bitsandbytes.nn as _bnb_nn  # type: ignore[import-untyped]
    except Exception as _bnb_exc:  # noqa: BLE001 — defensive; bnb is optional
        LOG.debug(
            "bitsandbytes.nn import failed (%s); 4-bit dtype detection "
            "skipped — params classify by storage element_size().",
            _bnb_exc,
        )
    else:
        _Params4bit = getattr(_bnb_nn, "Params4bit", None)

    # Logical element = one weight value as autograd sees it; Params4bit packs 2 per byte.
    by_bpe: dict[float, int] = {}
    for _, param in model.named_parameters():
        try:
            storage_numel = int(param.numel())
        except Exception as _exc:  # noqa: BLE001 — defensive, missing/meta params
            LOG.debug(
                "param.numel() failed during dtype detection (%s); skipping param.",
                _exc,
            )
            continue
        if storage_numel <= 0:
            continue
        if _Params4bit is not None and isinstance(param, _Params4bit):
            logical_numel = storage_numel * 2
            bpe = 0.5
        else:
            try:
                bpe = float(int(param.element_size()))
            except Exception as _exc:  # noqa: BLE001 — defensive
                LOG.debug(
                    "param.element_size() failed during dtype detection "
                    "(%s); skipping param.",
                    _exc,
                )
                continue
            logical_numel = storage_numel
        by_bpe[bpe] = by_bpe.get(bpe, 0) + logical_numel

    if not by_bpe:
        return 2.0

    # Ties favour smaller bpe so the searcher picks the tighter-budget regime on mixed models.
    dominant_bpe = min(
        by_bpe.keys(),
        key=lambda b: (-by_bpe[b], b),
    )
    return float(dominant_bpe)


def _dummy_batch(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    device: "torch.device | str",
) -> dict:
    """Build a sample batch appropriate for ``model``'s task type."""
    from axolotl.integrations.protrain.profiler.batch_factory import build_batch

    return build_batch(model, batch_size, seq_len, device)


def _infer_vocab_size(model: nn.Module) -> int:
    """Best-effort vocab size from common HF config shapes."""
    from axolotl.integrations.protrain.profiler.batch_factory import (
        _infer_vocab_size as _impl,
    )

    return _impl(model)


def _build_block_spans(
    model: nn.Module,
) -> tuple[list[nn.Module], dict[BlockId, list[ParamId]]]:
    """Return (blocks_list, block_id -> list[ParamId]) for the model."""
    blocks = flatten_block_trees(discover_blocks(model))
    named = list(model.named_parameters())

    block_prefixes: list[str] = []
    for block in blocks:
        prefix = _module_path_in(model, block)
        if prefix is None:
            prefix = ""
        block_prefixes.append(prefix)

    spans: dict[BlockId, list[ParamId]] = {BlockId(i): [] for i in range(len(blocks))}
    for param_name, _ in named:
        for idx, prefix in enumerate(block_prefixes):
            # Trailing "." guards against ``h.1`` matching ``h.10``.
            if prefix and (param_name == prefix or param_name.startswith(prefix + ".")):
                spans[BlockId(idx)].append(cast(ParamId, param_name))
                break
    return blocks, spans


def _module_path_in(root: nn.Module, target: nn.Module) -> str | None:
    """Return the dotted path of ``target`` inside ``root``, or None."""
    for name, candidate in root.named_modules():
        if candidate is target:
            return name or None
    return None


def _param_exec_order(
    model: nn.Module,
    block_spans: dict[BlockId, list[ParamId]],
    trace,
) -> list[ParamId]:
    """Param-level execution order derived from ``trace.op_order``."""
    del block_spans  # block grouping happens in build_layout

    # Direct params per module (children visited via their own ops).
    module_to_param_names: dict[str, list[str]] = {}
    for mod_path, module in model.named_modules():
        names = [
            f"{mod_path}.{p_name}" if mod_path else p_name
            for p_name, _ in module.named_parameters(recurse=False)
        ]
        if names:
            module_to_param_names[mod_path] = names

    # Identity-based dedup so weight-tied params collapse to first-use slot.
    seen_names: set[str] = set()
    seen_ids: set[int] = set()
    name_to_param = dict(model.named_parameters())
    order: list[ParamId] = []

    for rec in trace.op_order:
        if not rec.is_forward:
            continue
        param_names = module_to_param_names.get(rec.module_path)
        if not param_names:
            continue
        for name in param_names:
            if name in seen_names:
                continue
            param = name_to_param.get(name)
            if param is None:
                continue
            pid = id(param)
            if pid in seen_ids:
                seen_names.add(name)
                continue
            seen_ids.add(pid)
            seen_names.add(name)
            order.append(cast(ParamId, name))

    # Catch-all for params the trace never touched.
    for name, param in name_to_param.items():
        if name in seen_names:
            continue
        if id(param) in seen_ids:
            continue
        seen_ids.add(id(param))
        seen_names.add(name)
        order.append(cast(ParamId, name))

    return order


def _chunk_bytes(layout, chunk_manager) -> dict[int, int]:
    """Return ``{chunk_id -> actual bytes of its params}`` for ``layout``."""
    params_by_id = {str(name): p for name, p in chunk_manager.model.named_parameters()}
    out: dict[int, int] = {}
    for cid, pids in enumerate(layout.chunks):
        total = 0
        for pid in pids:
            p = params_by_id.get(str(pid))
            if p is None:
                continue
            total += int(p.numel()) * int(p.element_size())
        out[cid] = total
    return out


def predict_init_transient_peak_bytes(
    layout,
    hw: HardwareProfile,
    chunk_manager=None,
) -> int:
    """Predict the GPU high-water mark during the init transient window."""
    # Local import avoids cost.memory import cycle.
    from axolotl.integrations.protrain.cost.memory import ALPHA_FRAGMENTATION

    n_chunk = int(getattr(layout, "N_chunk", 0))
    s_chunk = int(getattr(layout, "S_chunk", 0))
    if n_chunk <= 0 or s_chunk <= 0:
        return 0

    if chunk_manager is not None:
        try:
            cb = _chunk_bytes(layout, chunk_manager)
        except Exception as exc:  # noqa: BLE001 — defensive, broken stub
            LOG.debug(
                "predict_init_transient_peak_bytes: _chunk_bytes failed "
                "(%s); falling back to N_chunk * S_chunk upper bound.",
                exc,
            )
            sum_chunk_bytes = n_chunk * s_chunk
        else:
            sum_chunk_bytes = sum(int(v) for v in cb.values())
            # Stub models with empty named_parameters collapse to 0; fall back to upper bound.
            if sum_chunk_bytes <= 0:
                sum_chunk_bytes = n_chunk * s_chunk
    else:
        sum_chunk_bytes = n_chunk * s_chunk

    # Reserved for future per-dtype iter-1 alpha refinement.
    _ = hw.dominant_param_bytes_per_element

    return int(sum_chunk_bytes * ALPHA_FRAGMENTATION)


def _calibrate_peak_with_actual_chunk_bytes(
    original_peak: int,
    layout,
    chunk_manager,
    cfg,
    trace=None,
    block_map=None,
    hw=None,
) -> int:
    """Recompute ``predicted_peak_bytes`` using actual chunk bytes + CKPT correction."""
    from axolotl.integrations.protrain.cost.memory import (
        ALPHA_FRAGMENTATION,
        _compute_ckpt_chain_bytes,
        _saved_tensor_bytes_per_block,
        alpha_fragmentation_for_cfg,
    )
    from axolotl.integrations.protrain.types import BlockMode

    S = layout.S_chunk
    cb = _chunk_bytes(layout, chunk_manager)

    fp16_total_bytes = layout.N_chunk * layout.S_chunk
    model_state_total = int(getattr(trace, "model_state_bytes", 0) or 0) if trace else 0
    if fp16_total_bytes > 0 and model_state_total > 0:
        persistent_factor = max(1.0, model_state_total / fp16_total_bytes)
    else:
        persistent_factor = 1.0
    buffer_factor = 2.0  # fp16 params (gathered) + fp16 grads (accumulated)
    if hw is not None:
        alpha = alpha_fragmentation_for_cfg(hw.dominant_param_bytes_per_element, cfg)
    else:
        alpha = ALPHA_FRAGMENTATION

    # Shared between production and cfg-delta floor boot-cfg paths.
    if trace is not None:
        saved_bytes_proxy = _saved_tensor_bytes_per_block(trace)
        act_sizes_full = dict(trace.activation_sizes)
        max_op_delta_global = 0
        for op in trace.op_order:
            if not op.is_forward:
                continue
            if op.block_id is None:
                continue
            contrib = trace.intra_op_delta.get(op.op_id, 0) + trace.inter_op_delta.get(
                op.op_id, 0
            )
            if contrib > max_op_delta_global:
                max_op_delta_global = contrib
    else:
        saved_bytes_proxy = {}
        act_sizes_full = {}
        max_op_delta_global = 0

    def _reconstruct_f_bm(bmap) -> tuple[int, int]:
        """Trace-derived F_bm reconstruction; returns (f_bm, n_ckpt).

        CKPT activation contribution uses chain semantics (sum across CKPT
        blocks) to mirror ``estimate_peak``'s ``ckpt_chain_bytes`` term. The
        prior ``max`` of a single block structurally under-bounded F_bm
        whenever multiple blocks were checkpointed.
        """
        if bmap is None or not act_sizes_full:
            return 0, 0
        live_none_bytes = 0
        for bid_, mode_ in bmap.items():
            if mode_ is BlockMode.NONE or mode_ is BlockMode.OFFLOAD:
                live_none_bytes += int(
                    saved_bytes_proxy.get(bid_, act_sizes_full.get(bid_, 0)) or 0
                )
        n_ckpt_ = sum(1 for m in bmap.values() if m is BlockMode.CKPT)
        if trace is not None:
            ckpt_chain = _compute_ckpt_chain_bytes(trace, bmap)
        else:
            ckpt_chain = 0
        return live_none_bytes + ckpt_chain + max_op_delta_global, n_ckpt_

    def _structural_calibrated(
        n_persist_arg: int,
        n_buffer_arg: int,
        original_peak_arg: int,
        bmap_arg,
    ) -> tuple[int, int, int, int]:
        """Structural-calibrated peak: returns (calibrated, persistent, buffer, f_bm) bytes."""
        persistent_ids_local = layout.effective_persistent_ids(n_persist_arg)
        n_persist_eff_local = len(persistent_ids_local)
        n_buffer_local = max(
            0, min(int(n_buffer_arg), layout.N_chunk - n_persist_eff_local)
        )
        actual_persistent_local = sum(
            cb.get(int(cid), 0) for cid in persistent_ids_local
        )
        original_model_state_local = int(
            n_persist_eff_local * S * persistent_factor
            + n_buffer_local * S * buffer_factor
        )
        f_bm_local = max(0, int(original_peak_arg / alpha) - original_model_state_local)
        reconstructed_local, n_ckpt_local = _reconstruct_f_bm(bmap_arg)
        if bmap_arg is not None:
            if n_ckpt_local >= max(1, len(bmap_arg) - 2):
                if f_bm_local > 0:
                    f_bm_local = min(f_bm_local, reconstructed_local)
                else:
                    f_bm_local = reconstructed_local
            else:
                f_bm_local = max(f_bm_local, reconstructed_local)
        buffer_bytes_local = int(n_buffer_local * S * buffer_factor)
        persistent_bytes_local = int(actual_persistent_local * persistent_factor)
        calibration_alpha_local = min(alpha, 1.05)
        calibrated_local = int(
            calibration_alpha_local
            * (persistent_bytes_local + buffer_bytes_local + f_bm_local)
        )
        return (
            calibrated_local,
            persistent_bytes_local,
            buffer_bytes_local,
            f_bm_local,
        )

    persistent_ids = set(int(c) for c in chunk_manager._persistent_ids)

    # Actual persistent param bytes; scaled by persistent_factor to recover full state.
    actual_persistent = sum(cb.get(cid, 0) for cid in persistent_ids)

    # Mirror cost.memory.model_state_present_bytes so reverse-out uses what the cost model added.
    n_persist_eff = len(persistent_ids)
    n_buffer = max(0, min(int(cfg.n_buffer), layout.N_chunk - n_persist_eff))

    original_model_state = int(
        n_persist_eff * S * persistent_factor + n_buffer * S * buffer_factor
    )
    f_bm = max(0, int(original_peak / alpha) - original_model_state)

    # CKPT-dominant: cap (min); else floor (max) so activations survive both failure modes.
    reconstructed_f_bm, n_ckpt = _reconstruct_f_bm(block_map)
    if block_map is not None:
        if n_ckpt >= max(1, len(block_map) - 2):
            if f_bm > 0:
                f_bm = min(f_bm, reconstructed_f_bm)
            else:
                f_bm = reconstructed_f_bm
        else:
            f_bm = max(f_bm, reconstructed_f_bm)

    # Two independent alphas (NOT stacked): paper 1.10 for feasibility, 1.05 post-hoc reporting.
    calibration_alpha = min(alpha, 1.05)
    # BufferPool pre-allocates all n_buffer slots up front; footprint is constant.
    buffer_bytes_eff = int(n_buffer * S * buffer_factor)
    calibrated_persistent = int(actual_persistent * persistent_factor)
    calibrated_raw = calibrated_persistent + buffer_bytes_eff + f_bm
    calibrated = int(calibration_alpha * calibrated_raw)
    LOG.debug(
        "ProTrain calibrate body: cfg=(np=%d nb=%d ns=%d nck=%d nof=%d) "
        "S_chunk=%.3fGiB N_chunk=%d n_persist_eff=%d n_buffer=%d "
        "actual_persistent=%.3fGiB persistent_factor=%.3f buffer_factor=%.2f "
        "f_bm=%.3fGiB calibrated_persistent=%.3fGiB buffer_bytes_eff=%.3fGiB "
        "calibrated_raw=%.3fGiB calibration_alpha=%.3f -> calibrated=%.3fGiB "
        "(original_peak=%.3fGiB original_model_state=%.3fGiB)",
        cfg.n_persist,
        cfg.n_buffer,
        cfg.n_swap,
        cfg.n_checkpoint,
        cfg.n_offload,
        S / (1 << 30),
        layout.N_chunk,
        n_persist_eff,
        n_buffer,
        actual_persistent / (1 << 30),
        persistent_factor,
        buffer_factor,
        f_bm / (1 << 30),
        calibrated_persistent / (1 << 30),
        buffer_bytes_eff / (1 << 30),
        calibrated_raw / (1 << 30),
        calibration_alpha,
        calibrated / (1 << 30),
        original_peak / (1 << 30),
        original_model_state / (1 << 30),
    )
    if trace is not None and block_map is not None:
        phase2_peak = int(getattr(trace, "steady_phase2_peak_bytes", 0) or 0)
        if phase2_peak > 0:
            n_ckpt = sum(1 for m in block_map.values() if m is BlockMode.CKPT)
            # Compare against cfg.n_persist (prefix), matching how phase2_n_persist was recorded.
            phase2_matches_cfg = (
                int(cfg.n_persist) == int(getattr(trace, "phase2_n_persist", -1))
                and int(cfg.n_buffer) == int(getattr(trace, "phase2_n_buffer", -1))
                and n_ckpt == int(getattr(trace, "phase2_n_checkpoint", -1))
            )

            _PHASE2_SAFETY_MARGIN = 0.05
            phase2_analytical_peak = int(
                getattr(trace, "phase2_analytical_peak_bytes", 0) or 0
            )
            if phase2_matches_cfg:
                phase2_floor = int((1.0 + _PHASE2_SAFETY_MARGIN) * phase2_peak)
                if phase2_peak > calibrated:
                    calibrated = phase2_floor
                else:
                    # Symmetric 5% margin on the over-predict side too.
                    calibrated = min(calibrated, phase2_floor)
            elif phase2_analytical_peak > 0 and hw is not None:
                # Cfg-delta path: floor anchors phase2_peak, delta is structural prod - boot.
                from axolotl.integrations.protrain.cost.memory import (
                    estimate_peak as _estimate_peak,
                )

                prod_analytical_peak = int(
                    _estimate_peak(cfg, trace, layout, block_map, hw)
                )
                prod_calibrated, _, _, _ = _structural_calibrated(
                    int(cfg.n_persist),
                    int(cfg.n_buffer),
                    prod_analytical_peak,
                    block_map,
                )
                # Validate the trace's boot shape is the canonical (n_persist=0 + all-CKPT).
                boot_n_persist = int(getattr(trace, "phase2_n_persist", -1))
                boot_n_buffer = int(getattr(trace, "phase2_n_buffer", -1))
                boot_n_ckpt = int(getattr(trace, "phase2_n_checkpoint", -1))
                if (
                    boot_n_persist == 0
                    and boot_n_ckpt == len(block_map)
                    and boot_n_buffer >= 0
                ):
                    boot_block_map = {bid_: BlockMode.CKPT for bid_ in block_map.keys()}
                    boot_calibrated, _, _, _ = _structural_calibrated(
                        boot_n_persist,
                        boot_n_buffer,
                        phase2_analytical_peak,
                        boot_block_map,
                    )
                    delta_raw = max(0, prod_calibrated - boot_calibrated)
                    calibrated_floor = max(
                        int(phase2_peak + delta_raw),
                        phase2_peak,
                    )
                    LOG.info(
                        "ProTrain peak cfg-delta (calibrated-delta): "
                        "phase2_peak=%.2f GB phase2_anal=%.2f GB "
                        "prod_anal=%.2f GB boot_calibrated=%.2f GB "
                        "prod_calibrated=%.2f GB delta_raw=%.2f GB "
                        "floor=%.2f GB calibrated=%.2f GB",
                        phase2_peak / (1 << 30),
                        phase2_analytical_peak / (1 << 30),
                        prod_analytical_peak / (1 << 30),
                        boot_calibrated / (1 << 30),
                        prod_calibrated / (1 << 30),
                        delta_raw / (1 << 30),
                        calibrated_floor / (1 << 30),
                        calibrated / (1 << 30),
                    )
                elif phase2_analytical_peak > 0:
                    # Legacy alpha-stripped fallback for non-canonical boot shapes.
                    delta_raw_legacy = (
                        prod_analytical_peak - phase2_analytical_peak
                    ) / float(ALPHA_FRAGMENTATION)
                    delta_raw_legacy = max(0.0, delta_raw_legacy)
                    calibrated_floor = max(
                        int(phase2_peak + delta_raw_legacy),
                        phase2_peak,
                    )
                    LOG.info(
                        "ProTrain peak cfg-delta (legacy alpha-strip): "
                        "phase2_peak=%.2f GB phase2_anal=%.2f GB "
                        "prod_anal=%.2f GB delta_raw=%.2f GB "
                        "floor=%.2f GB calibrated=%.2f GB",
                        phase2_peak / (1 << 30),
                        phase2_analytical_peak / (1 << 30),
                        prod_analytical_peak / (1 << 30),
                        delta_raw_legacy / (1 << 30),
                        calibrated_floor / (1 << 30),
                        calibrated / (1 << 30),
                    )
                else:
                    delta = max(0, prod_analytical_peak - phase2_analytical_peak)
                    calibrated_floor = int(phase2_peak + delta)
                # Anchor as LOWER bound only — structural body's cfg-specific terms survive above.
                calibrated = max(calibrated, calibrated_floor)
    return calibrated


def _cpu_ram_per_rank_bytes(world_size: int) -> int:
    """Best-effort estimate of per-rank available CPU RAM in bytes."""
    ws = max(1, int(world_size))
    try:
        import psutil

        return max(0, int(psutil.virtual_memory().available) // ws)
    except ImportError:
        pass

    # Fallback: /proc/meminfo on Linux.
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return max(0, (kb * 1024) // ws)
    except (FileNotFoundError, OSError, ValueError):
        pass

    # No probe: 0 lets the auto-selector pick the safest fit-on-GPU path.
    return 0


def _default_cpu_capacity_for_search(gpu_count: int) -> int | None:
    """Derive the per-rank CPU capacity used as a search-time hard filter."""
    gc = max(1, int(gpu_count))
    try:
        import psutil
    except ImportError:
        LOG.warning(
            "psutil not installed; ProTrain search-time CPU feasibility "
            "filter is disabled. Install psutil to enable host-RAM "
            "filtering of search candidates."
        )
        return None
    try:
        available = int(psutil.virtual_memory().available)
    except Exception as exc:  # noqa: BLE001 — defensive on exotic platforms
        LOG.warning(
            "psutil.virtual_memory() raised %s; ProTrain search-time CPU "
            "feasibility filter is disabled for this run.",
            exc,
        )
        return None
    per_rank = available // gc - _DEFAULT_CPU_HEADROOM_BYTES
    return max(0, int(per_rank))


def _select_mode(
    search_result: SearchResult,
    layout,
    hw: HardwareProfile,
    world_size: int,
    cpu_ram_per_rank_bytes: int,
    *,
    auto_mode: bool,
    user_force_all_persistent: bool,
    user_zero3_shard: bool | None,
) -> tuple[bool, bool]:
    """Resolve ``(force_all_persistent, zero3_shard)`` for the wrapper."""
    if not auto_mode:
        return (
            bool(user_force_all_persistent),
            bool(user_zero3_shard) if user_zero3_shard is not None else False,
        )

    # Single-rank: honour searcher's persistent-vs-offload decision.
    if world_size <= 1:
        return (
            int(search_result.cfg.n_persist) >= int(layout.N_chunk),
            False,
        )

    # Mode A: searcher says everything fits on GPU.
    if int(search_result.cfg.n_persist) >= int(layout.N_chunk):
        return (True, False)

    from dataclasses import replace as _replace

    from axolotl.integrations.protrain.cost.memory import (
        estimate_cpu_footprint,
    )

    hw_replicated = _replace(hw, zero3_shard=False)
    replicated_footprint = int(
        estimate_cpu_footprint(search_result.cfg, layout, hw_replicated)
    )
    hw_sharded = _replace(hw, zero3_shard=True)
    sharded_footprint = int(
        estimate_cpu_footprint(search_result.cfg, layout, hw_sharded)
    )

    if cpu_ram_per_rank_bytes >= replicated_footprint:
        return (False, False)
    if cpu_ram_per_rank_bytes >= sharded_footprint:
        return (False, True)

    raise RuntimeError(
        "ProTrain auto-mode: model does not fit on this node. Searcher "
        f"picked n_persist={search_result.cfg.n_persist}/"
        f"{layout.N_chunk} (needs CPU offload), but per-rank CPU RAM "
        f"({cpu_ram_per_rank_bytes / 1e9:.1f} GB) is smaller than the "
        f"sharded footprint ({sharded_footprint / 1e9:.1f} GB). Scale "
        "up: more nodes, more system RAM, smaller model, or a larger "
        "per-rank capacity budget."
    )


def _construct_runtime(
    *,
    model: nn.Module,
    blocks: list[nn.Module],
    layout,
    result: SearchResult,
    hardware_profile: HardwareProfile,
    capacity_bytes: int,
    trace,
    zero3_shard,
    device,
) -> tuple["ChunkManager", "Scheduler", list[Any], SearchResult]:
    """Build chunk_manager + scheduler + hooks under a given ``result``."""
    import sys as _sys2

    import torch

    n_persist = result.cfg.n_persist
    # Runtime floor: scheduler lookahead needs current + next block's non-persistent chunks to fit.
    required_n_buffer = min_n_buffer_for(layout, n_persist)
    if result.cfg.n_buffer < required_n_buffer:
        LOG.warning(
            "ProTrain: searcher returned n_buffer=%d but runtime requires "
            ">= %d for the scheduler's lookahead prefetch (n_persist=%d, "
            "N_chunk=%d). Bumping n_buffer; cost-model prediction may be "
            "slightly off.",
            int(result.cfg.n_buffer),
            int(required_n_buffer),
            int(n_persist),
            int(layout.N_chunk),
        )
        n_buffer = int(required_n_buffer)
    else:
        n_buffer = int(result.cfg.n_buffer)

    # All-persistent layout: skip pool construction to avoid burning S_chunk pinned + GPU.
    pinned_host: "PinnedHostMemory | None"
    buffer_pool: "BufferPool | None"
    if n_buffer == 0:
        pinned_host = None
        buffer_pool = None
    else:
        pinned_host = PinnedHostMemory(n_buffer=n_buffer, S_chunk=layout.S_chunk)
        buffer_pool = BufferPool(
            n_buffer=n_buffer,
            S_chunk=layout.S_chunk,
            pinned_host=pinned_host,
            device=device,
        )

    # Effective persistent set = prefix | layout.mandatory_persistent (chunks with non-block params).
    effective_persistent_ids: frozenset[ChunkId] = layout.effective_persistent_ids(
        n_persist
    )
    if layout.mandatory_persistent:
        LOG.info(
            "ProTrain: %d chunks %s pinned by layout.mandatory_persistent "
            "(non-block params the block-granularity scheduler cannot "
            "gather on its own); residency = prefix[0..%d) | mandatory",
            len(layout.mandatory_persistent),
            sorted(layout.mandatory_persistent),
            n_persist,
        )

    # Partition params: persistent → GPU optim; rest → per-chunk CPU FusedAdam.
    params_by_name: dict[str, nn.Parameter] = dict(model.named_parameters())
    persistent_params: list[nn.Parameter] = []
    cpu_params_per_chunk: dict = {}

    for cid, chunk_param_ids in enumerate(layout.chunks):
        chunk_params = [
            params_by_name[str(pid)]
            for pid in chunk_param_ids
            if str(pid) in params_by_name
        ]
        if cid in effective_persistent_ids:
            persistent_params.extend(chunk_params)
        else:
            cpu_params_per_chunk[cid] = chunk_params

    # CpuFusedAdamAdapter construction is deferred to AFTER materialize_offload below.
    gpu_optim: GpuFusedAdamAdapter | None = None
    if persistent_params:
        gpu_optim = GpuFusedAdamAdapter(params=persistent_params, lr=1e-4)

    # ChunkManager silently degrades zero3_shard to False on ws==1.
    _ws = 1
    _rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        _ws = int(torch.distributed.get_world_size())
        _rank = int(torch.distributed.get_rank())
    _zero3 = bool(hardware_profile.zero3_shard) and (_ws > 1)
    LOG.info(
        "ProTrain: distributed context world_size=%d rank=%d zero3_shard=%s "
        "(requested=%s)",
        _ws,
        _rank,
        _zero3,
        zero3_shard,
    )

    # Always-on: zero-element placeholders break custom autograd functions (lora_mlp_kernel)
    # that capture param shape via save_for_backward; per-param scratch cost is one element.
    _shape_preserving = True
    chunk_manager = ChunkManager(
        model=model,
        layout=layout,
        n_persist=n_persist,
        buffer_pool=buffer_pool,
        cpu_optim=None,  # wired in after materialize_offload
        gpu_optim=gpu_optim,
        device=device,
        world_size=_ws,
        rank=_rank,
        zero3_shard=_zero3,
        shape_preserving_placeholders=_shape_preserving,
    )

    # Sanity check: drift between residency sets would misroute chunks between optimisers.
    if chunk_manager._persistent_ids != set(effective_persistent_ids):
        raise RuntimeError(
            "ProTrain invariant violated: "
            "ChunkManager residency drift: expected "
            f"{sorted(effective_persistent_ids)}, got "
            f"{sorted(chunk_manager._persistent_ids)}"
        )

    calibrated_peak = _calibrate_peak_with_actual_chunk_bytes(
        original_peak=result.predicted_peak_bytes,
        layout=layout,
        chunk_manager=chunk_manager,
        cfg=result.cfg,
        trace=trace,
        block_map=result.block_map,
        hw=hardware_profile,
    )
    init_transient_peak = predict_init_transient_peak_bytes(
        layout, hardware_profile, chunk_manager
    )
    if calibrated_peak != result.predicted_peak_bytes or init_transient_peak > 0:
        if calibrated_peak != result.predicted_peak_bytes:
            LOG.info(
                "ProTrain: peak prediction calibrated %.2f -> %.2f GB "
                "using actual per-chunk byte footprint",
                result.predicted_peak_bytes / (1 << 30),
                calibrated_peak / (1 << 30),
            )
        # cfg.n_persist preserves the search's prefix length.
        result = SearchResult(
            cfg=CostConfig(
                n_persist=result.cfg.n_persist,
                n_buffer=result.cfg.n_buffer,
                n_swap=result.cfg.n_swap,
                n_checkpoint=result.cfg.n_checkpoint,
                n_offload=result.cfg.n_offload,
            ),
            block_map=result.block_map,
            predicted_peak_bytes=calibrated_peak,
            predicted_iter_s=result.predicted_iter_s,
            predicted_init_transient_peak_bytes=init_transient_peak,
        )
    LOG.info(
        "ProTrain: predicted peaks: steady=%.2f GiB iter1_transient=%.2f GiB "
        "(ratio=%.2fx; > 2x suggests Mode-C offload regime)",
        result.predicted_peak_bytes / (1 << 30),
        init_transient_peak / (1 << 30),
        (
            init_transient_peak / max(result.predicted_peak_bytes, 1)
            if init_transient_peak > 0
            else 0.0
        ),
    )

    # Move non-persistent chunk data to pinned CPU + install per-param grad hooks.
    alloc_before = (
        torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    )
    freed = chunk_manager.materialize_offload()
    alloc_after = (
        torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    )
    LOG.info(
        "ProTrain: materialize_offload freed %.2f GB (reported), "
        "alloc %.2f -> %.2f GB (torch measured)",
        freed / (1 << 30),
        alloc_before / (1 << 30),
        alloc_after / (1 << 30),
    )
    _sys2.stderr.write(
        f"[protrain] materialize_offload: freed {freed / 1e9:.2f}GB "
        f"(alloc {alloc_before / 1e9:.2f}->{alloc_after / 1e9:.2f}GB)\n"
    )
    _sys2.stderr.flush()

    # Diagnostic: post-materialize param accounting for v53 OOM investigation.
    try:
        import torch as _diag_torch

        _diag_persistent_ids = set(getattr(chunk_manager, "_persistent_ids", set()))
        _diag_non_persistent_ids = set(
            getattr(chunk_manager, "_non_persistent_ids", set())
        )
        _diag_total_numel = 0
        _diag_total_storage_bytes = 0
        _diag_param_count = 0
        for _name, _p in model.named_parameters():
            _diag_param_count += 1
            try:
                _diag_total_numel += int(_p.numel())
            except Exception:  # noqa: BLE001
                pass
            try:
                _diag_total_storage_bytes += int(_p.untyped_storage().nbytes())
            except Exception:  # noqa: BLE001
                pass
        _diag_alloc_gib = (
            _diag_torch.cuda.memory_allocated(device) / (1 << 30)
            if _diag_torch.cuda.is_available()
            else 0.0
        )
        LOG.warning(
            "[protrain-diag] post-materialize: alloc=%.2f GiB "
            "persistent_chunks=%d non_persistent_chunks=%d "
            "total_params=%d sum_numel=%d sum_storage_bytes=%d "
            "(sum_numel * 2B = %.2f GiB ideal-bf16-weights, "
            "sum_storage = %.2f GiB actual-resident)",
            _diag_alloc_gib,
            len(_diag_persistent_ids),
            len(_diag_non_persistent_ids),
            _diag_param_count,
            _diag_total_numel,
            _diag_total_storage_bytes,
            (_diag_total_numel * 2) / (1 << 30),
            _diag_total_storage_bytes / (1 << 30),
        )
    except Exception as _diag_exc:  # noqa: BLE001
        LOG.warning("[protrain-diag] post-materialize accounting failed: %s", _diag_exc)

    # DDP _sync_module_states broadcast trips shared-storage hazard on expand placeholders.
    # ProTrain owns parallelism contract for chunk-managed params (shard at init, gather, reduce_scatter).
    if _shape_preserving:
        # Monkey-patch DDP.__init__ to inject init_sync=False when our marker is present.
        try:
            import torch.nn.parallel as _tnp

            _ddp_cls = _tnp.DistributedDataParallel
            if not getattr(_ddp_cls, "_protrain_init_sync_patched", False):
                _orig_init = _ddp_cls.__init__

                def _patched_init(self, module, *args, **kwargs):
                    _walk = module
                    _seen: set[int] = set()
                    while _walk is not None and id(_walk) not in _seen:
                        _seen.add(id(_walk))
                        if getattr(_walk, "_protrain_ddp_skip_init_sync", False):
                            kwargs["init_sync"] = False
                            LOG.info(
                                "ProTrain: "
                                "DistributedDataParallel.__init__ "
                                "patched-injection of init_sync=False "
                                "for chunk-managed model — "
                                "_sync_module_states broadcast and "
                                "_verify_param_shape_across_processes "
                                "are bypassed (every rank already "
                                "agreed on init state via "
                                "materialize_offload's deterministic "
                                "partition).",
                            )
                            break
                        _walk = getattr(_walk, "module", None)
                    return _orig_init(self, module, *args, **kwargs)

                _ddp_cls.__init__ = _patched_init
                _ddp_cls._protrain_init_sync_patched = True

            model._protrain_ddp_skip_init_sync = True  # type: ignore[attr-defined]
        except Exception as _patch_exc:  # noqa: BLE001 — defensive
            LOG.warning(
                "ProTrain: failed to install "
                "DistributedDataParallel init_sync bypass patch: %s. "
                "Multi-GPU sharded path may still trip the shared-"
                "storage hazard at DDP construction time.",
                _patch_exc,
            )

        ignore = chunk_manager.chunk_managed_param_names()
        # Cross-check: ignore names must resolve through model.named_parameters().
        live_names = {n for n, _ in model.named_parameters()}
        unmatched = ignore - live_names
        if unmatched:
            LOG.warning(
                "ProTrain: %d/%d chunk-managed names do NOT "
                "match model.named_parameters() — DDP broadcast filter "
                "will MISS them. Sample mismatches: %s",
                len(unmatched),
                len(ignore),
                sorted(unmatched)[:5],
            )
        existing = getattr(model, "_ddp_params_and_buffers_to_ignore", None)
        if existing is None:
            model._ddp_params_and_buffers_to_ignore = list(ignore)  # type: ignore[attr-defined]
        else:
            merged = set(existing) | ignore
            model._ddp_params_and_buffers_to_ignore = list(merged)  # type: ignore[attr-defined]
        LOG.info(
            "ProTrain: registered %d chunk-managed param "
            "names in model._ddp_params_and_buffers_to_ignore (live "
            "match: %d/%d) so DDP's _sync_module_states broadcast "
            "skips the shape-preserving expand placeholders (write "
            "would trip the shared-storage hazard on the expanded "
            "view).",
            len(ignore),
            len(ignore - unmatched),
            len(ignore),
        )
        # Diagnostic: ignore-list state at registration site for v53 OOM investigation.
        try:
            _diag_ignore_list = list(
                getattr(model, "_ddp_params_and_buffers_to_ignore", []) or []
            )
            _diag_total_params = len(list(model.parameters()))
            _diag_sample = _diag_ignore_list[:3]
            _diag_truncated = [
                (n[:80] + "...") if len(n) > 80 else n for n in _diag_sample
            ]
            LOG.warning(
                "[protrain-diag] ignore-list post-register: size=%d "
                "total_params=%d first3=%s",
                len(_diag_ignore_list),
                _diag_total_params,
                _diag_truncated,
            )
        except Exception as _diag_exc:  # noqa: BLE001
            LOG.warning("[protrain-diag] ignore-list logging failed: %s", _diag_exc)
    else:
        # Rebuild path: strip stale DDP-skip state from a prior shape-preserving wrap.
        if getattr(model, "_protrain_ddp_skip_init_sync", False):
            try:
                delattr(model, "_protrain_ddp_skip_init_sync")
            except AttributeError:
                pass
        if hasattr(model, "_protrain_ddp_original_ignore"):
            try:
                _original = model._protrain_ddp_original_ignore  # type: ignore[attr-defined]
                if _original is None:
                    if hasattr(model, "_ddp_params_and_buffers_to_ignore"):
                        try:
                            delattr(model, "_ddp_params_and_buffers_to_ignore")
                        except AttributeError:
                            pass
                else:
                    model._ddp_params_and_buffers_to_ignore = list(_original)  # type: ignore[attr-defined]
                try:
                    delattr(model, "_protrain_ddp_original_ignore")
                except AttributeError:
                    pass
                LOG.info(
                    "ProTrain: rebuild path detected — stripped stale "
                    "DDP skip state from model so the rebuilt "
                    "runtime (non-shape-preserving) receives normal "
                    "init_sync + backward allreduce semantics."
                )
            except Exception as _exc:  # noqa: BLE001 — defensive
                LOG.warning(
                    "ProTrain: failed to strip stale DDP skip state on rebuild: %s",
                    _exc,
                )

    # CPU FusedAdam built post-offload so it references post-shard nn.Parameter objects.
    # Sharded chunks: adapter updates per-region shard_param (one per _DtypeRegion).
    cpu_params_per_chunk_for_optim: dict = {}
    for cid, chunk_params in cpu_params_per_chunk.items():
        shard_state = chunk_manager._chunk_shards.get(cid)  # type: ignore[attr-defined]
        if shard_state is not None and shard_state.regions:
            cpu_params_per_chunk_for_optim[cid] = [
                r.shard_param for r in shard_state.regions
            ]
        else:
            cpu_params_per_chunk_for_optim[cid] = chunk_params

    cpu_optim: CpuFusedAdamAdapter | None = None
    if any(params for params in cpu_params_per_chunk_for_optim.values()):
        try:
            cpu_optim = CpuFusedAdamAdapter(
                params_per_chunk=cpu_params_per_chunk_for_optim,
                lr=1e-4,
            )
        except (ImportError, Exception) as err:  # noqa: BLE001 - DS raises CUDAMismatchException
            # Render err to string before logging — live exception leaks frame locals (GPU params).
            err_repr = f"{type(err).__name__}: {err}"
            LOG.warning(
                "ProTrain: CPU FusedAdam unavailable (%s); non-persistent chunks "
                "will not get async CPU Adam. Install DeepSpeed with a matching "
                "CUDA toolkit (or set DS_SKIP_CUDA_CHECK=1) for full coverage.",
                err_repr,
            )
            del err
            cpu_optim = None
    chunk_manager.cpu_optim = cpu_optim

    eff_h2d, eff_d2h = effective_bw(result.cfg, hardware_profile)

    scheduler = Scheduler(
        chunk_manager=chunk_manager,
        block_map=result.block_map,
        layout=layout,
        effective_h2d_bps=eff_h2d,
        effective_d2h_bps=eff_d2h,
    )
    # Back-ref so _ProTrainOptimizer.step() can call scheduler.drain() at the step boundary.
    chunk_manager._scheduler_ref = scheduler

    # Encoder-decoder models have two ModuleLists; _find_block_parent_map returns one per block.
    block_parent_map = _find_block_parent_map(model, blocks)
    for idx, block in enumerate(blocks):
        mode = result.block_map.get(BlockId(idx))
        if mode is None:
            continue
        wrapped_block = wrap_block(block, mode)
        if wrapped_block is not block:
            parent = block_parent_map.get(id(block))
            if parent is not None:
                # idx is global; within-tree slot may differ for decoder blocks.
                for slot, child in enumerate(parent):
                    if child is block:
                        parent[slot] = wrapped_block
                        break
            blocks[idx] = wrapped_block

    # Build the activation SWAP pool when n_swap > 0.
    if result.cfg.n_swap > 0:
        from axolotl.integrations.protrain.types import BlockMode as _BM_swap

        # Slot must hold the largest SAVED tensor (output | intra-op transient | saved weight).
        max_act_bytes = 0
        max_param_bytes = 0
        max_intra_delta_bytes = 0
        swap_block_ids: set[int] = set()
        for bid, mode in result.block_map.items():
            if mode is _BM_swap.SWAP:
                swap_block_ids.add(int(bid))
                act = trace.activation_sizes.get(bid, 0)
                if act > max_act_bytes:
                    max_act_bytes = int(act)
        # Largest forward-op intra delta across SWAP blocks.
        if swap_block_ids:
            for op in trace.op_order:
                if not op.is_forward:
                    continue
                if op.block_id is None:
                    continue
                if int(op.block_id) not in swap_block_ids:
                    continue
                delta = int(trace.intra_op_delta.get(op.op_id, 0))
                if delta > max_intra_delta_bytes:
                    max_intra_delta_bytes = delta
        # Largest individual parameter tensor across SWAP blocks (covers F.linear saved weight).
        for idx, block in enumerate(blocks):
            if int(idx) not in swap_block_ids:
                continue
            inner = getattr(block, "block", block)
            for p in inner.parameters(recurse=True):
                pb = int(p.numel() * p.element_size())
                if pb > max_param_bytes:
                    max_param_bytes = pb
        slot_bytes_required = max(
            int(max_act_bytes),
            int(max_intra_delta_bytes),
            int(max_param_bytes),
        )
        if slot_bytes_required <= 0:
            LOG.warning(
                "ProTrain: result.cfg.n_swap=%d but no SWAP block has "
                "non-zero activation_sizes/params; skipping swap-pool "
                "construction",
                result.cfg.n_swap,
            )
        else:
            from axolotl.integrations.protrain.block.swap_pool import (
                DEFAULT_SLOTS_PER_BLOCK,
                ActivationSwapPool,
            )
            from axolotl.integrations.protrain.cost.memory import (
                SWAP_PREFETCH_DEPTH,
            )

            slots_per_block = DEFAULT_SLOTS_PER_BLOCK
            # Floor at 1 byte for the pool's positive-size invariant.
            per_slot = max(1, slot_bytes_required)
            swap_pool = ActivationSwapPool(
                n_swap=result.cfg.n_swap,
                slot_bytes=per_slot,
                prefetch_depth=SWAP_PREFETCH_DEPTH,
                slots_per_block=slots_per_block,
            )
            scheduler.swap_pool = swap_pool
            for block in blocks:
                if getattr(block, "_protrain_wrapped_mode", None) is _BM_swap.SWAP:
                    block.attach_runtime(swap_pool, scheduler.swap_stream)
            LOG.info(
                "ProTrain: SWAP pool wired — %d slots x %d bytes = %.2f MB "
                "pinned (slot sized from max(act=%.2f MB, intra_op=%.2f MB, "
                "param=%.2f MB))",
                swap_pool.n_slot,
                swap_pool.slot_bytes,
                swap_pool.total_bytes / (1 << 20),
                max_act_bytes / (1 << 20),
                max_intra_delta_bytes / (1 << 20),
                max_param_bytes / (1 << 20),
            )

    handles = install_hooks(
        model=model,
        chunk_manager=chunk_manager,
        block_map=result.block_map,
        scheduler=scheduler,
    )

    # Block wrappers inject .block. infix into named_parameters paths; rebuild ignore set by id.
    if _shape_preserving:
        try:
            # Restrict to NON-persistent chunks — persistent need normal DDP broadcast/allreduce.
            chunk_managed_param_ids: set[int] = set()
            for _cid in chunk_manager._non_persistent_ids:
                _slots = chunk_manager._cpu_slots.get(_cid)
                if not _slots:
                    continue
                for _cpu_slot in _slots:
                    # Renamed from ``slot`` to avoid shadowing the int binding above.
                    _p = chunk_manager._params_by_id.get(_cpu_slot.param_id)
                    if _p is not None:
                        chunk_managed_param_ids.add(id(_p))
            post_wrap_ignore: set[str] = {
                live_name
                for live_name, live_param in model.named_parameters()
                if id(live_param) in chunk_managed_param_ids
            }
            _original = getattr(model, "_protrain_ddp_original_ignore", None)
            if _original is None:
                model._ddp_params_and_buffers_to_ignore = list(post_wrap_ignore)  # type: ignore[attr-defined]
            else:
                model._ddp_params_and_buffers_to_ignore = list(  # type: ignore[attr-defined]
                    set(_original) | post_wrap_ignore
                )
            LOG.info(
                "ProTrain: re-registered "
                "%d chunk-managed param names in "
                "model._ddp_params_and_buffers_to_ignore using "
                "post-block-wrap named_parameters() (DDP's backward "
                "allreduce filter sees the .block.-infixed names).",
                len(post_wrap_ignore),
            )
            # Diagnostic: post-block-wrap ignore-list state for v53 OOM investigation.
            try:
                _diag_final = list(
                    getattr(model, "_ddp_params_and_buffers_to_ignore", []) or []
                )
                _diag_total = len(list(model.parameters()))
                LOG.warning(
                    "[protrain-diag] ignore-list post-block-wrap: "
                    "size=%d total_params=%d non_persistent_chunks=%d",
                    len(_diag_final),
                    _diag_total,
                    len(chunk_manager._non_persistent_ids),
                )
            except Exception as _diag_exc:  # noqa: BLE001
                LOG.warning(
                    "[protrain-diag] post-block-wrap logging failed: %s",
                    _diag_exc,
                )
        except Exception as _exc:  # noqa: BLE001 — defensive
            LOG.warning(
                "ProTrain: failed to "
                "re-register _ddp_params_and_buffers_to_ignore after "
                "block-wrap: %s. DDP's backward allreduce may attempt "
                "to reduce chunk-managed param gradients.",
                _exc,
            )

    # capacity_bytes kept in signature for future per-capacity derating.
    del capacity_bytes

    return chunk_manager, scheduler, list(handles), result


def protrain_model_wrapper(
    model: nn.Module,
    model_config: object,  # noqa: ARG001 — accepted for API symmetry with the plan
    hardware_profile: HardwareProfile,
    *,
    batch_size: int,
    seq_len: int,
    capacity_bytes: int | None = None,
    cpu_capacity_bytes: int | None = None,
    cache_dir: str | None = None,
    force_all_persistent: bool = False,
    n_persist_override: int | None = None,
    n_buffer_override: int | None = None,
    n_swap_override: int | None = None,
    n_checkpoint_override: int | None = None,
    n_offload_override: int | None = None,
    zero3_shard: bool | None = None,
    auto_mode: bool = False,
    target_device: "torch.device | str | int | None" = None,
    forbid_activation_offload: bool = False,
    prefer_no_offload_on_non_nvlink: bool = True,
) -> WrappedModel:
    """Compose the ProTrain runtime around a standard ``nn.Module``.

    ``forbid_activation_offload``: when True, the searcher refuses any
    CostConfig with ``n_offload > 0``. Set from ``cfg.lora_mlp_kernel`` —
    the fused MLP backward kernel is incompatible with chunk-storage
    placeholders.

    ``prefer_no_offload_on_non_nvlink``: defensive searcher tie-break;
    auto-prefers ``n_offload=0`` configs within a 5% noise band when the
    hardware profile reports multi-rank without NVLink. See PR #17c.
    """
    import torch

    # Device precedence: target_device > model._protrain_target_device > model param device > cuda:0/cpu.
    resolved_target: torch.device | None = None
    if target_device is not None:
        resolved_target = torch.device(target_device)
    else:
        attr_target = getattr(model, "_protrain_target_device", None)
        if attr_target is not None:
            resolved_target = torch.device(attr_target)

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = None

    if resolved_target is not None:
        device = resolved_target
    elif model_device is not None:
        device = model_device
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Move model to the resolved CUDA target when needed; never swallow OOM here.
    if (
        device.type == "cuda"
        and model_device is not None
        and model_device != device
        and getattr(model, "hf_device_map", None) is None
    ):
        LOG.info(
            "ProTrain: moving model from %s to %s before profiling "
            "(target_device hint).",
            model_device,
            device,
        )
        model.to(device)

    # Force use_cache=False — HF KV cache breaks CKPT recompute (shape mismatch on past_key_values).
    cfg_obj = getattr(model, "config", None)
    if cfg_obj is not None and getattr(cfg_obj, "use_cache", False):
        LOG.info(
            "ProTrain: forcing model.config.use_cache=False for CKPT compatibility"
        )
        cfg_obj.use_cache = False

    cache_key = ProfilerCacheKey(
        arch_hash=_arch_hash(model),
        bs=batch_size,
        seq=seq_len,
        sku=_sku(device),
        world=hardware_profile.gpu_count,
    )
    # Override-skip: the un-offloaded trace OOMs on big-model offload configs before chunk offload engages.
    # Synthetic trace is NOT saved — analytical placeholders must not poison future non-override runs.
    _override_skip_trace = (
        n_persist_override is not None
        and n_buffer_override is not None
        and n_swap_override is not None
        and n_checkpoint_override is not None
    )
    trace = load_cached_trace(cache_key, cache_dir=cache_dir)
    if trace is None and _override_skip_trace:
        import sys as _sys

        LOG.info(
            "ProTrain: explicit knob override path with cache miss — "
            "synthesizing ProfilerTrace from defaults and SKIPPING the "
            "trace pass (n_persist=%s n_buffer=%s n_swap=%s n_checkpoint=%s "
            "n_offload=%s). This avoids the trace-pass OOM on big-model "
            "offload configurations where the un-offloaded forward+backward "
            "exceeds device memory before chunk offload can engage.",
            n_persist_override,
            n_buffer_override,
            n_swap_override,
            n_checkpoint_override,
            n_offload_override,
        )
        _sys.stderr.write(
            "[protrain] override path: skipping trace pass, "
            "synthesizing ProfilerTrace from defaults\n"
        )
        _sys.stderr.flush()
        trace = synth_trace_from_overrides(
            model,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            world_size=int(hardware_profile.gpu_count),
        )
        _sys.stderr.write(
            f"[protrain] synth trace done: {len(trace.activation_sizes)} blocks "
            f"(no op_order, no measured activations)\n"
        )
        _sys.stderr.flush()
    elif trace is None:
        import sys as _sys

        LOG.info(
            "ProTrain profiler cache miss for %s — running trace (bs=%d seq=%d)",
            cache_key.fingerprint()[:12],
            batch_size,
            seq_len,
        )
        _sys.stderr.write(
            "[protrain] profiler cache miss — running backward-aware trace\n"
        )
        _sys.stderr.flush()
        # Backward-aware: peak memory typically occurs in backward; OnDemandTensorMgr guards OOM.
        profiler_cfg = ProfilerConfig(
            batch_size=batch_size,
            seq_len=seq_len,
            device=str(device),
            include_backward=True,
            on_demand=True,
            force_all_persistent=bool(force_all_persistent),
            world_size=int(hardware_profile.gpu_count),
        )
        batch = _dummy_batch(model, batch_size, seq_len, device)
        trace = run_trace(model, batch, profiler_cfg)
        _sys.stderr.write(
            f"[protrain] trace done: {len(trace.op_order)} ops, "
            f"{len(trace.activation_sizes)} blocks\n"
        )
        _sys.stderr.flush()
        save_cached_trace(cache_key, trace, cache_dir=cache_dir)
    else:
        LOG.info("ProTrain profiler cache hit for %s", cache_key.fingerprint()[:12])

    import sys as _sys2

    _sys2.stderr.write("[protrain] building layout\n")
    _sys2.stderr.flush()
    blocks, block_spans = _build_block_spans(model)
    exec_order = _param_exec_order(model, block_spans, trace)

    # Pass exec_order + block_spans so pick_S_chunk's simulation matches build_layout's placement.
    param_bytes: dict[ParamId, int] = {
        cast(ParamId, name): int(p.numel()) * int(p.element_size())
        for name, p in model.named_parameters()
    }
    s_chunk = pick_S_chunk(
        param_bytes,
        exec_order=exec_order,
        block_spans=block_spans,
    )

    layout = build_layout(
        model=model,
        exec_order=exec_order,
        S_chunk=s_chunk,
        block_spans=block_spans,
    )
    _sys2.stderr.write(
        f"[protrain] layout built: S_chunk={layout.S_chunk} N_chunk={layout.N_chunk}\n"
    )
    _sys2.stderr.flush()

    if capacity_bytes is None:
        capacity_bytes = max(
            0, int(hardware_profile.gpu_memory_bytes) - _DEFAULT_HEADROOM_BYTES
        )

    # Search-time CPU feasibility budget; None means filter disabled.
    if cpu_capacity_bytes is None:
        cpu_capacity_bytes = _default_cpu_capacity_for_search(
            hardware_profile.gpu_count
        )

    # Early world-size probe for mode selector + zero3_shard plumbing.
    _ws_early = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        _ws_early = int(torch.distributed.get_world_size())

    # Snapshot user's raw intent before auto-selector potentially rewrites flags.
    _user_force_all_persistent = bool(force_all_persistent)
    _user_zero3_shard = zero3_shard

    if auto_mode:
        # Auto path: let the searcher pick; flip mode post-search via _select_mode.
        if _user_force_all_persistent:
            LOG.info(
                "ProTrain auto-mode: user set force_all_persistent=True "
                "but auto-mode overrides explicit flags. Running searcher "
                "— will pick Mode A naturally if the workload fits on "
                "GPU. Set ``protrain_auto_mode: false`` to force-honour "
                "force_all_persistent=True."
            )
        force_all_persistent = False
        zero3_shard = False

    # Search-time zero3 is the most-permissive choice so the CPU gate doesn't preempt Mode C.
    if auto_mode and _ws_early > 1:
        _zero3_for_hw = True
    elif zero3_shard is None:
        _zero3_for_hw = (_ws_early > 1) and (not force_all_persistent)
    else:
        _zero3_for_hw = bool(zero3_shard) and (_ws_early > 1)
    from dataclasses import replace as _replace

    _hw_updates: dict = {}
    if _zero3_for_hw != hardware_profile.zero3_shard:
        _hw_updates["zero3_shard"] = _zero3_for_hw
    # Only overwrite caller-provided Adam rates when they're absent (zero).
    if (
        hardware_profile.cpu_adam_bytes_per_sec <= 0.0
        and trace.cpu_adam_bytes_per_sec > 0.0
    ):
        _hw_updates["cpu_adam_bytes_per_sec"] = trace.cpu_adam_bytes_per_sec
    if (
        hardware_profile.gpu_adam_bytes_per_sec <= 0.0
        and trace.gpu_adam_bytes_per_sec > 0.0
    ):
        _hw_updates["gpu_adam_bytes_per_sec"] = trace.gpu_adam_bytes_per_sec
    # Live SKU compute rate for cross-SKU per-op latency scaling.
    if hardware_profile.gpu_compute_tflops <= 0.0:
        try:
            _live_tflops = measure_compute_rate(int(getattr(device, "index", 0) or 0))
            if _live_tflops > 0.0:
                _hw_updates["gpu_compute_tflops"] = _live_tflops
        except Exception as _e:  # noqa: BLE001 - defensive
            LOG.debug(
                "measure_compute_rate live failed (%s); skipping SKU calibration", _e
            )
    # PCIe: overwrite the conservative 13 GB/s default with measured H2D/D2H.
    if hardware_profile.pcie_h2d_bps <= 13e9 + 1e6 and trace.pcie_h2d_bps > 13e9 + 1e6:
        _hw_updates["pcie_h2d_bps"] = trace.pcie_h2d_bps
    if hardware_profile.pcie_d2h_bps <= 13e9 + 1e6 and trace.pcie_d2h_bps > 13e9 + 1e6:
        _hw_updates["pcie_d2h_bps"] = trace.pcie_d2h_bps
    # Per-dtype alpha lookup: only stamp when detection differs AND caller passed default.
    _detected_bpe = _detect_dominant_param_bytes_per_element(model)
    if (
        abs(hardware_profile.dominant_param_bytes_per_element - 2.0) < 1e-9
        and abs(_detected_bpe - 2.0) > 1e-9
    ):
        _hw_updates["dominant_param_bytes_per_element"] = _detected_bpe
    if _hw_updates:
        hardware_profile = _replace(hardware_profile, **_hw_updates)

    # Phase-2 re-search must keep the permissive search-time profile to avoid filtering Mode-C-only cfgs.
    search_hw_profile = hardware_profile

    n_block = max(1, len(trace.activation_sizes))

    all_overrides_set = all(
        v is not None
        for v in (
            n_persist_override,
            n_buffer_override,
            n_swap_override,
            n_checkpoint_override,
        )
    )

    if force_all_persistent:
        # Synthetic all-persistent + all-CKPT cfg; cost model skipped, predictions zeroed.
        synth_cfg = CostConfig(
            n_persist=layout.N_chunk,
            n_buffer=min_n_buffer_for(layout, layout.N_chunk),
            n_swap=0,
            n_checkpoint=n_block,
        )
        block_map = assign_modes(n_swap=0, n_checkpoint=n_block, N_block=n_block)
        result = SearchResult(
            cfg=synth_cfg,
            block_map=block_map,
            predicted_peak_bytes=0,
            predicted_iter_s=0.0,
        )
        LOG.warning(
            "ProTrain: force_all_persistent=True — bypassing searcher. "
            "n_persist=%d n_buffer=%d n_swap=0 n_checkpoint=%d. "
            "All model state stays GPU-resident; activations rely on CKPT. "
            "This is the documented workaround for the M4.5 runtime gaps.",
            synth_cfg.n_persist,
            synth_cfg.n_buffer,
            synth_cfg.n_checkpoint,
        )
        _sys2.stderr.write(f"[protrain] force_all_persistent: cfg={result.cfg}\n")
        _sys2.stderr.flush()
    elif all_overrides_set:
        # Explicit 4-tuple override; bounds-check manually since searcher is skipped.
        assert n_persist_override is not None
        assert n_buffer_override is not None
        assert n_swap_override is not None
        assert n_checkpoint_override is not None

        n_persist = int(n_persist_override)
        n_buffer = int(n_buffer_override)
        n_swap = int(n_swap_override)
        n_checkpoint = int(n_checkpoint_override)
        n_offload = int(n_offload_override) if n_offload_override is not None else 0

        if not (0 <= n_persist <= layout.N_chunk):
            raise ValueError(
                f"n_persist_override={n_persist} out of range [0, {layout.N_chunk}]"
            )
        if n_buffer < 0:
            raise ValueError(f"n_buffer_override must be >= 0, got {n_buffer}")
        if not (0 <= n_swap <= n_block):
            raise ValueError(f"n_swap_override={n_swap} out of range [0, {n_block}]")
        if not (0 <= n_checkpoint <= n_block - n_swap):
            raise ValueError(
                f"n_checkpoint_override={n_checkpoint} incompatible "
                f"with n_swap_override={n_swap} (N_block={n_block})"
            )
        if not (0 <= n_offload <= n_block - n_swap - n_checkpoint):
            raise ValueError(
                f"n_offload_override={n_offload} incompatible with "
                f"n_swap_override={n_swap} + "
                f"n_checkpoint_override={n_checkpoint} (N_block={n_block}); "
                f"valid range is [0, {n_block - n_swap - n_checkpoint}]"
            )
        synth_cfg = CostConfig(
            n_persist=n_persist,
            n_buffer=n_buffer,
            n_swap=n_swap,
            n_checkpoint=n_checkpoint,
            n_offload=n_offload,
        )
        block_map = assign_modes(
            n_swap=n_swap,
            n_checkpoint=n_checkpoint,
            N_block=n_block,
            n_offload=n_offload,
        )

        # Replicate searcher's two runtime-safety invariants (buffer floor + admissible block_map).
        min_buffer = min_n_buffer_for(layout, n_persist)
        if n_buffer < min_buffer:
            raise ValueError(
                f"n_buffer_override={n_buffer} below scheduler minimum "
                f"{min_buffer} for n_persist={n_persist} on this layout "
                f"(N_chunk={layout.N_chunk}). The lookahead prefetch "
                "needs the union of current+next non-persistent chunks "
                "to fit in the pool simultaneously."
            )
        if not block_map_runtime_admissible(layout, block_map, n_persist):
            raise ValueError(
                f"override block_map for n_swap={n_swap} n_checkpoint={n_checkpoint} "
                f"n_offload={n_offload} is runtime-unsafe at n_persist={n_persist}: "
                "at least one block owns non-persistent chunks but is in NONE mode. "
                "NONE installs no activation-save hooks, so PyTorch's autograd "
                "saved-tensors hold direct GPU storage refs that the chunk pool's "
                "slot reuse will clobber. Use CKPT (recompute), OFFLOAD (saved-"
                "tensors-hook re-gather), or SWAP (saved tensors persisted to a "
                "pinned-CPU pool decoupled from param.data) for non-persistent "
                "blocks — or raise n_persist to make those blocks fully resident."
            )

        result = SearchResult(
            cfg=synth_cfg,
            block_map=block_map,
            predicted_peak_bytes=0,
            predicted_iter_s=0.0,
        )
        LOG.warning(
            "ProTrain: explicit knob override path — bypassing searcher. cfg=%s",
            synth_cfg,
        )
        _sys2.stderr.write(f"[protrain] explicit override: cfg={result.cfg}\n")
        _sys2.stderr.flush()
    else:
        _sys2.stderr.write(
            f"[protrain] running exhaustive search (N_chunk={layout.N_chunk}, "
            f"N_block={n_block})\n"
        )
        _sys2.stderr.flush()
        result = search(
            trace,
            layout,
            int(capacity_bytes),
            hardware_profile,
            cpu_capacity_bytes=cpu_capacity_bytes,
            forbid_activation_offload=forbid_activation_offload,
            prefer_no_offload_on_non_nvlink=prefer_no_offload_on_non_nvlink,
        )
        _sys2.stderr.write(
            f"[protrain] search done: cfg={result.cfg} "
            f"peak={result.predicted_peak_bytes / 1e9:.2f}GB "
            f"iter={result.predicted_iter_s:.3f}s\n"
        )
        _sys2.stderr.flush()

    # Auto-mode selection.
    if auto_mode:
        cpu_ram = _cpu_ram_per_rank_bytes(_ws_early)
        if cpu_ram == 0 and _ws_early > 1:
            LOG.warning(
                "ProTrain auto-mode: could not probe CPU RAM via psutil or "
                "/proc/meminfo. Treating per-rank RAM as 0 bytes — the "
                "selector will prefer Mode A (force_all_persistent) and "
                "raise if the model needs offload. Set "
                "``protrain_auto_mode: false`` and pick the mode "
                "explicitly on exotic topologies."
            )
        auto_force_persistent, auto_zero3 = _select_mode(
            search_result=result,
            layout=layout,
            hw=hardware_profile,
            world_size=_ws_early,
            cpu_ram_per_rank_bytes=cpu_ram,
            auto_mode=True,
            user_force_all_persistent=_user_force_all_persistent,
            user_zero3_shard=_user_zero3_shard,
        )

        # Warn if explicit zero3 flag is being overridden — Mode A is typically faster.
        if _user_zero3_shard is True and not auto_zero3 and _ws_early > 1:
            LOG.warning(
                "ProTrain auto-mode: user set zero3_shard=True but the "
                "workload fits in Mode A (force_all_persistent). "
                "Auto-mode picked Mode A for better throughput — on a "
                "non-NVLink 4x RTX 3090 rig, DDP+Mode_A gives ~3.6x "
                "scaling vs ZeRO-3's ~0.7x. Set ``protrain_auto_mode: "
                "false`` to force-honour zero3_shard=True."
            )

        if auto_force_persistent:
            if _ws_early > 1:
                LOG.info(
                    "ProTrain auto-mode: picking Mode A "
                    "(force_all_persistent=True). On a non-NVLink 4x RTX "
                    "3090 rig, DDP+Mode_A gives ~3.6x scaling vs ZeRO-3's "
                    "~0.7x — see DESIGN.md §Multi-GPU for benchmark data."
                )
            else:
                LOG.info(
                    "ProTrain auto-mode: picking Mode A "
                    "(force_all_persistent=True, single-rank)."
                )
        elif not auto_zero3:
            LOG.info(
                "ProTrain auto-mode: picking Mode B (CPU-offload, "
                "replicated). Per-rank CPU RAM sufficient for the full "
                "non-persistent chunk set."
            )
        else:
            LOG.info(
                "ProTrain auto-mode: picking Mode C (CPU-offload, "
                "ZeRO-3 sharded). Per-rank CPU RAM too tight for "
                "replication — falling back to 1/world_size shard."
            )

        force_all_persistent = auto_force_persistent
        zero3_shard = auto_zero3
        # Re-stamp runtime profile; search_hw_profile (snapshot) stays permissive for phase-2.
        if zero3_shard != hardware_profile.zero3_shard:
            from dataclasses import replace as _replace

            hardware_profile = _replace(hardware_profile, zero3_shard=bool(zero3_shard))

    # Phase-2: build under conservative bootstrap, measure, splice trace, re-search.
    # Optimizer slots aren't wired into the trainer yet; rebuild here is safe.
    n_block = len(trace.activation_sizes)
    use_phase2 = (
        torch.cuda.is_available()
        and trace.steady_bwd_chunked_wall_s == 0.0
        and n_block > 0
        # Skip on explicit-override and force_all_persistent — nothing to refine.
        and not force_all_persistent
        and not all_overrides_set
    )
    if use_phase2:
        from axolotl.integrations.protrain.profiler.phase2 import (
            estimate_per_block_recompute_s,
            measure_chunked_steady,
            select_bootstrap_config,
        )

        boot_cfg, boot_block_map = select_bootstrap_config(
            initial_result=result,
            layout=layout,
            n_block=n_block,
            capacity_bytes=capacity_bytes,
            trace=trace,
            hw=hardware_profile,
        )
        boot_result = SearchResult(
            cfg=boot_cfg,
            block_map=boot_block_map,
            predicted_peak_bytes=result.predicted_peak_bytes,
            predicted_iter_s=result.predicted_iter_s,
        )
        chunk_manager, scheduler, handles, boot_result = _construct_runtime(
            model=model,
            blocks=blocks,
            layout=layout,
            result=boot_result,
            hardware_profile=hardware_profile,
            capacity_bytes=capacity_bytes,
            trace=trace,
            zero3_shard=zero3_shard,
            device=device,
        )
        boot_wrapped = WrappedModel(
            module=model,
            search_result=boot_result,
            chunk_manager=chunk_manager,
            scheduler=scheduler,
            _hook_handles=list(handles),
        )
        from axolotl.integrations.protrain.api.optim_wrapper import (
            protrain_optimizer_wrapper,
        )

        boot_optim = protrain_optimizer_wrapper(boot_wrapped, lr=1e-4)
        boot_batch = _dummy_batch(model, batch_size, seq_len, device)

        measurement_failed = False
        fwd_s = 0.0
        bwd_s = 0.0
        step_s = 0.0
        phase2_peak_bytes = 0
        try:
            fwd_s, bwd_s, step_s, phase2_peak_bytes = measure_chunked_steady(
                model=model, batch=boot_batch, optimizer=boot_optim
            )
        except Exception as exc:  # noqa: BLE001 — measurement is best-effort
            exc_repr = f"{type(exc).__name__}: {exc}"
            LOG.warning(
                "Phase-2 chunked measurement raised %s; falling back to "
                "the v8 cost-model path under the searcher's original "
                "pick. Tighten or disable the phase-2 gate if the "
                "failure is reproducible.",
                exc_repr,
            )
            del exc
            measurement_failed = True

        if measurement_failed:
            # Tear down bootstrap and rebuild under original cfg; unwrap blocks to restore param names.
            for h in handles:
                try:
                    h.remove()  # type: ignore[attr-defined]
                except Exception as exc:  # noqa: BLE001 — best-effort
                    LOG.debug(
                        "phase-2 fallback teardown: hook handle remove failed: %s",
                        exc,
                    )
            block_parent_map_unwrap = _find_block_parent_map(model, blocks)
            for idx, block in enumerate(blocks):
                unwrapped = unwrap_block(block)
                if unwrapped is not block:
                    parent = block_parent_map_unwrap.get(id(block))
                    if parent is not None:
                        for slot, child in enumerate(parent):
                            if child is block:
                                parent[slot] = unwrapped
                                break
                    blocks[idx] = unwrapped
            chunk_manager.restore_to_gpu()
            del boot_wrapped, boot_optim, chunk_manager, scheduler, handles
            chunk_manager, scheduler, handles, result = _construct_runtime(
                model=model,
                blocks=blocks,
                layout=layout,
                result=result,
                hardware_profile=hardware_profile,
                capacity_bytes=capacity_bytes,
                trace=trace,
                zero3_shard=zero3_shard,
                device=device,
            )
        if not measurement_failed:
            # Pre-splice ordering mirrors v10; per_block is the same shape pre/post.
            per_block_recompute_s = estimate_per_block_recompute_s(trace, n_block)

            # Capture analytical baselines BEFORE splicing — chunked-wall override gates fall through.
            from axolotl.integrations.protrain.cost.memory import (
                estimate_peak as _estimate_peak,
            )
            from axolotl.integrations.protrain.cost.runtime import (
                _estimate_runtime_components,
                estimate_runtime as _estimate_runtime,
            )

            phase2_analytical_iter_s_val = float(
                _estimate_runtime(
                    boot_cfg, trace, layout, boot_block_map, hardware_profile
                )
            )
            # Per-component decomposition at boot cfg for per-component alpha calibration.
            (
                t_fwd_boot,
                t_bwd_boot,
                t_gpu_optim_boot,
                t_cpu_optim_boot,
                _fwd_used_boot,
                _bwd_used_boot,
            ) = _estimate_runtime_components(
                boot_cfg, trace, layout, boot_block_map, hardware_profile
            )
            # Additive analytical step matches the alphaopt ratio.
            phase2_analytical_fwd_s_val = float(t_fwd_boot)
            phase2_analytical_bwd_s_val = float(t_bwd_boot)
            phase2_analytical_step_s_val = float(t_gpu_optim_boot + t_cpu_optim_boot)
            phase2_analytical_peak_bytes_val = int(
                _estimate_peak(
                    boot_cfg, trace, layout, boot_block_map, hardware_profile
                )
            )
            phase2_iter_s_val = float(fwd_s + bwd_s + step_s)

            # Residual-alpha anchor: per-component analytical path under derived alphas.
            from axolotl.integrations.protrain.cost.runtime import (
                _PHASE2_ALPHA_CLAMP_MAX,
                _PHASE2_ALPHA_CLAMP_MIN,
            )

            def _clamp_for_anchor(x: float) -> float:
                return max(_PHASE2_ALPHA_CLAMP_MIN, min(_PHASE2_ALPHA_CLAMP_MAX, x))

            if (
                phase2_analytical_fwd_s_val > 0.0
                and phase2_analytical_bwd_s_val > 0.0
                and phase2_analytical_step_s_val > 0.0
            ):
                a_fwd_boot = _clamp_for_anchor(
                    float(fwd_s) / phase2_analytical_fwd_s_val
                )
                a_bwd_boot = _clamp_for_anchor(
                    float(bwd_s) / phase2_analytical_bwd_s_val
                )
                a_opt_boot = _clamp_for_anchor(
                    float(step_s) / phase2_analytical_step_s_val
                )
                t_fwd_anchor = a_fwd_boot * float(t_fwd_boot)
                t_bwd_anchor = a_bwd_boot * float(t_bwd_boot)
                t_gpu_anchor = a_opt_boot * float(t_gpu_optim_boot)
                t_cpu_anchor = a_opt_boot * float(t_cpu_optim_boot)
                phase2_per_comp_pred_iter_s_val = float(
                    t_fwd_anchor
                    + t_bwd_anchor
                    + t_gpu_anchor
                    + max(0.0, t_cpu_anchor - t_bwd_anchor)
                )
            else:
                phase2_per_comp_pred_iter_s_val = 0.0

            from dataclasses import replace as _replace

            new_trace = _replace(
                trace,
                steady_fwd_chunked_wall_s=fwd_s,
                steady_bwd_chunked_wall_s=bwd_s,
                steady_step_overlap_s=step_s,
                steady_phase2_peak_bytes=phase2_peak_bytes,
                phase2_n_persist=boot_result.cfg.n_persist,
                phase2_n_buffer=boot_result.cfg.n_buffer,
                phase2_n_checkpoint=boot_result.cfg.n_checkpoint,
                phase2_per_block_recompute_s=per_block_recompute_s,
                phase2_iter_s=phase2_iter_s_val,
                phase2_analytical_iter_s=phase2_analytical_iter_s_val,
                phase2_analytical_peak_bytes=phase2_analytical_peak_bytes_val,
                phase2_fwd_s=float(fwd_s),
                phase2_bwd_s=float(bwd_s),
                phase2_step_s=float(step_s),
                phase2_analytical_fwd_s=phase2_analytical_fwd_s_val,
                phase2_analytical_bwd_s=phase2_analytical_bwd_s_val,
                phase2_analytical_step_s=phase2_analytical_step_s_val,
                phase2_per_comp_pred_iter_s=phase2_per_comp_pred_iter_s_val,
            )
            try:
                save_cached_trace(cache_key, new_trace, cache_dir=cache_dir)
            except OSError as exc:
                LOG.warning(
                    "Phase-2: failed to persist updated trace (%s); the "
                    "in-memory trace is still updated for this run.",
                    exc,
                )
            trace = new_trace

            # Re-run search with phase-2 fields populated; reuse CPU budget (phase-2 only refines runtime).
            # Use search_hw_profile (permissive) so the CPU gate doesn't drop Mode-C-only candidates.
            new_result = search(
                trace,
                layout,
                capacity_bytes,
                search_hw_profile,
                cpu_capacity_bytes=cpu_capacity_bytes,
                forbid_activation_offload=forbid_activation_offload,
                prefer_no_offload_on_non_nvlink=prefer_no_offload_on_non_nvlink,
            )

            # Re-pick runtime mode for the post-measurement cfg.
            mode_changed = False
            if auto_mode:
                cpu_ram_re = _cpu_ram_per_rank_bytes(_ws_early)
                new_force_persistent, new_zero3 = _select_mode(
                    search_result=new_result,
                    layout=layout,
                    hw=search_hw_profile,
                    world_size=_ws_early,
                    cpu_ram_per_rank_bytes=cpu_ram_re,
                    auto_mode=True,
                    user_force_all_persistent=_user_force_all_persistent,
                    user_zero3_shard=_user_zero3_shard,
                )
                # Mode flip MUST trigger rebuild even when cfg/block_map match — runtime built under old mode.
                mode_changed = (
                    new_force_persistent != force_all_persistent
                    or new_zero3 != zero3_shard
                )
                if mode_changed:
                    LOG.info(
                        "Phase-2: post-measurement _select_mode changed "
                        "the runtime mode (force_all_persistent %s -> %s, "
                        "zero3_shard %s -> %s); rebuilding the runtime.",
                        force_all_persistent,
                        new_force_persistent,
                        zero3_shard,
                        new_zero3,
                    )
                force_all_persistent = new_force_persistent
                zero3_shard = new_zero3
                if zero3_shard != hardware_profile.zero3_shard:
                    hardware_profile = _replace(
                        hardware_profile, zero3_shard=bool(zero3_shard)
                    )
            # Compare against raw boot_cfg for symmetry / robustness against future calibration knobs.
            cfg_changed = (
                new_result.cfg != boot_cfg
                or new_result.block_map != boot_block_map
                or mode_changed
            )
            if not cfg_changed:
                calibrated_peak = _calibrate_peak_with_actual_chunk_bytes(
                    original_peak=new_result.predicted_peak_bytes,
                    layout=layout,
                    chunk_manager=chunk_manager,
                    cfg=new_result.cfg,
                    trace=trace,
                    block_map=new_result.block_map,
                    hw=hardware_profile,
                )
                # Reuse bootstrap-time init transient — post-offload _chunk_bytes sees placeholders.
                init_transient_peak = boot_result.predicted_init_transient_peak_bytes
                if (
                    calibrated_peak != new_result.predicted_peak_bytes
                    or init_transient_peak
                    != new_result.predicted_init_transient_peak_bytes
                ):
                    new_result = SearchResult(
                        cfg=CostConfig(
                            n_persist=new_result.cfg.n_persist,
                            n_buffer=new_result.cfg.n_buffer,
                            n_swap=new_result.cfg.n_swap,
                            n_checkpoint=new_result.cfg.n_checkpoint,
                            n_offload=new_result.cfg.n_offload,
                        ),
                        block_map=new_result.block_map,
                        predicted_peak_bytes=calibrated_peak,
                        predicted_iter_s=new_result.predicted_iter_s,
                        predicted_init_transient_peak_bytes=init_transient_peak,
                    )
                LOG.info(
                    "Phase-2: post-measurement search picked the same cfg "
                    "(predicted_iter_s %.4f -> %.4f); keeping bootstrap "
                    "runtime in place.",
                    boot_result.predicted_iter_s,
                    new_result.predicted_iter_s,
                )
                result = new_result
                wrapped = boot_wrapped
                wrapped.search_result = result
            else:
                LOG.info(
                    "Phase-2: post-measurement search picked a different "
                    "cfg (%s -> %s); tearing down bootstrap runtime and "
                    "rebuilding under the new pick.",
                    boot_result.cfg,
                    new_result.cfg,
                )
                # Unwrap blocks so rebuild's _build_block_spans sees param names matching layout.chunks.
                for h in handles:
                    try:
                        h.remove()  # type: ignore[attr-defined]
                    except Exception as exc:  # noqa: BLE001 — best-effort
                        LOG.debug(
                            "phase-2 teardown: hook handle remove failed: %s",
                            exc,
                        )
                block_parent_map_unwrap = _find_block_parent_map(model, blocks)
                for idx, block in enumerate(blocks):
                    unwrapped = unwrap_block(block)
                    if unwrapped is not block:
                        parent = block_parent_map_unwrap.get(id(block))
                        if parent is not None:
                            for slot, child in enumerate(parent):
                                if child is block:
                                    parent[slot] = unwrapped
                                    break
                        blocks[idx] = unwrapped
                chunk_manager.restore_to_gpu()
                del boot_wrapped, boot_optim, chunk_manager, scheduler, handles
                chunk_manager, scheduler, handles, result = _construct_runtime(
                    model=model,
                    blocks=blocks,
                    layout=layout,
                    result=new_result,
                    hardware_profile=hardware_profile,
                    capacity_bytes=capacity_bytes,
                    trace=trace,
                    zero3_shard=zero3_shard,
                    device=device,
                )
    else:
        chunk_manager, scheduler, handles, result = _construct_runtime(
            model=model,
            blocks=blocks,
            layout=layout,
            result=result,
            hardware_profile=hardware_profile,
            capacity_bytes=capacity_bytes,
            trace=trace,
            zero3_shard=zero3_shard,
            device=device,
        )

    LOG.info(
        "ProTrain config: n_persist=%d n_buffer=%d n_swap=%d n_checkpoint=%d "
        "S_chunk=%d N_chunk=%d peak=%.2f GiB iter1_transient=%.2f GiB "
        "iter=%.3f s capacity=%.2f GiB",
        result.cfg.n_persist,
        result.cfg.n_buffer,
        result.cfg.n_swap,
        result.cfg.n_checkpoint,
        layout.S_chunk,
        layout.N_chunk,
        result.predicted_peak_bytes / (1 << 30),
        result.predicted_init_transient_peak_bytes / (1 << 30),
        result.predicted_iter_s,
        capacity_bytes / (1 << 30),
    )

    wrapped = WrappedModel(
        module=model,
        search_result=result,
        chunk_manager=chunk_manager,
        scheduler=scheduler,
        _hook_handles=list(handles),
    )
    # Stash searcher inputs for the plugin's post_trainer_create NCCL re-search.
    wrapped._trace = trace  # type: ignore[attr-defined]
    wrapped._layout = layout  # type: ignore[attr-defined]
    wrapped._capacity_bytes = int(capacity_bytes)  # type: ignore[attr-defined]
    wrapped._cpu_capacity_bytes = (  # type: ignore[attr-defined]
        int(cpu_capacity_bytes) if cpu_capacity_bytes is not None else None
    )
    wrapped._hardware_profile = hardware_profile  # type: ignore[attr-defined]
    wrapped._cache_key = cache_key  # type: ignore[attr-defined]
    wrapped._cache_dir = cache_dir  # type: ignore[attr-defined]
    wrapped._override_skip_trace = bool(_override_skip_trace)  # type: ignore[attr-defined]
    return wrapped


def auto_wrap(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    *,
    capacity_bytes: int | None = None,
    cpu_capacity_bytes: int | None = None,
    cache_dir: str | None = None,
) -> WrappedModel:
    """Drop-in ProTrain wrapper with auto-derived ``HardwareProfile``."""
    import torch

    from axolotl.integrations.protrain.api.hardware import build_hardware_profile

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ProTrain.auto_wrap requires CUDA; torch.cuda.is_available() is False. "
            "Construct a HardwareProfile manually and call protrain_model_wrapper "
            "directly for non-CUDA test harnesses."
        )

    # Refuse a CPU-resident model; surface contract here rather than at the profiler.
    try:
        param_device = next(model.parameters()).device
    except StopIteration:
        param_device = None
    if param_device is not None and param_device.type != "cuda":
        raise RuntimeError(
            f"ProTrain.auto_wrap requires the model to be on GPU before the call "
            f"(found device={param_device}). Move it first via "
            f"`model.cuda()` or `model.to('cuda:N')`, then re-invoke."
        )

    hw = build_hardware_profile()

    return protrain_model_wrapper(
        model,
        model_config=getattr(model, "config", None),
        hardware_profile=hw,
        batch_size=batch_size,
        seq_len=seq_len,
        capacity_bytes=capacity_bytes,
        cpu_capacity_bytes=cpu_capacity_bytes,
        cache_dir=cache_dir,
        auto_mode=True,
    )


def _find_block_parent_map(
    model: nn.Module, blocks: list[nn.Module]
) -> dict[int, "nn.ModuleList"]:
    """Map ``id(block)`` to the ``nn.ModuleList`` containing it."""
    out: dict[int, "nn.ModuleList"] = {}
    if not blocks:
        return out
    target_ids = {id(b) for b in blocks}
    for module in model.modules():
        if not isinstance(module, nn.ModuleList):
            continue
        for child in module:
            cid = id(child)
            if cid in target_ids and cid not in out:
                out[cid] = module
    return out


__all__ = [
    "auto_wrap",
    "predict_init_transient_peak_bytes",
    "protrain_model_wrapper",
]
