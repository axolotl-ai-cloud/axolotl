"""Phase-2 chunked-runtime profiler — measures bwd/step under the chunk manager."""

from __future__ import annotations

import copy
import statistics
from typing import TYPE_CHECKING, Any

from axolotl.integrations.protrain.types import (
    ChunkId,
    CostConfig,
    SearchResult,
)
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch
    from torch import nn

    from axolotl.integrations.protrain.types import (
        BlockStrategyMap,
        ChunkLayout,
        HardwareProfile,
        ProfilerTrace,
    )

LOG = get_logger(__name__)


# 3 warmups settle buffer-pool LRU + CPU-Adam lazy init.
_PHASE2_N_WARMUP = 3
# 5 timed iters give a stable median on 7B-LoRA (~5% per-iter variance).
_PHASE2_N_ITERS = 5


def _min_n_buffer_for_layout(layout: "ChunkLayout", n_persist: int) -> int:
    """Minimum pool size for adjacent-block prefetch at ``n_persist``."""
    if n_persist >= layout.N_chunk:
        return 0
    persistent: set[ChunkId] = set(layout.effective_persistent_ids(n_persist))
    block_ids = sorted(layout.block_to_chunks.keys())
    if not block_ids:
        return 0
    need = 0
    for i, bid in enumerate(block_ids):
        cur_np = [c for c in layout.block_to_chunks.get(bid, ()) if c not in persistent]
        nxt_np: list[ChunkId] = []
        if i + 1 < len(block_ids):
            nxt_np = [
                c
                for c in layout.block_to_chunks.get(block_ids[i + 1], ())
                if c not in persistent
            ]
        need = max(need, len({*cur_np, *nxt_np}))
    # Return raw need; an all-persistent layout legitimately needs zero buffers.
    return need


def select_bootstrap_config(
    *,
    initial_result: SearchResult,
    layout: "ChunkLayout",
    n_block: int,
    capacity_bytes: int,
    trace: "ProfilerTrace",
    hw: "HardwareProfile",
) -> tuple[CostConfig, "BlockStrategyMap"]:
    """Pick a conservative bootstrap config that's guaranteed to fit.

    Spec: ``n_persist=0``, ``n_swap=0``, ``n_checkpoint=N_block``,
    ``n_buffer=min(layout.N_chunk, max(initial_result.cfg.n_buffer,
    _min_n_buffer_for_layout(layout, n_persist=0)))``. This biases hard
    toward memory savings (zero persistence, full activation
    checkpointing) while keeping ``n_buffer`` large enough to satisfy
    Zero-persist + all-CKPT calibration baseline; falls back to initial pick on peak overflow."""
    from axolotl.integrations.protrain.block.layout_rules import assign_modes
    from axolotl.integrations.protrain.cost.memory import estimate_peak

    # n_buffer floor = max(searcher pick, layout's adjacent-block prefetch need).
    min_buffer = _min_n_buffer_for_layout(layout, 0)
    bootstrap_cfg = CostConfig(
        n_persist=0,
        n_buffer=min(
            layout.N_chunk,
            max(initial_result.cfg.n_buffer, min_buffer),
        ),
        n_swap=0,
        n_checkpoint=n_block,
    )
    bootstrap_block_map = assign_modes(0, n_block, n_block)

    candidate_peak = estimate_peak(
        bootstrap_cfg, trace, layout, bootstrap_block_map, hw
    )
    if candidate_peak <= capacity_bytes:
        LOG.info(
            "Phase-2 bootstrap config: n_persist=%d n_buffer=%d "
            "n_checkpoint=%d (peak %.2f GB <= capacity %.2f GB)",
            bootstrap_cfg.n_persist,
            bootstrap_cfg.n_buffer,
            bootstrap_cfg.n_checkpoint,
            candidate_peak / (1 << 30),
            capacity_bytes / (1 << 30),
        )
        return bootstrap_cfg, bootstrap_block_map

    LOG.warning(
        "Phase-2 bootstrap formula (n_persist=%d n_buffer=%d "
        "n_checkpoint=%d) predicts peak %.2f GB > capacity %.2f GB; "
        "falling back to the searcher's first pick which passed the "
        "capacity gate by construction.",
        bootstrap_cfg.n_persist,
        bootstrap_cfg.n_buffer,
        bootstrap_cfg.n_checkpoint,
        candidate_peak / (1 << 30),
        capacity_bytes / (1 << 30),
    )
    return initial_result.cfg, initial_result.block_map


def _clone_state_dict(state, target_device=None):
    """Recursively clone every tensor in a (possibly nested) state_dict."""
    import torch

    if torch.is_tensor(state):
        if target_device is not None:
            target = torch.device(target_device)
            if state.device == target:
                return state.detach().clone()
            # to(target) yields independent storage when devices differ; no extra clone needed.
            return state.detach().to(target)
        return state.detach().clone()
    if isinstance(state, dict):
        cloned_items = {
            k: _clone_state_dict(v, target_device=target_device)
            for k, v in state.items()
        }
        # Preserve dict subclass identity (e.g. OrderedDict).
        try:
            cloned_dict = type(state)(cloned_items)
        except TypeError:
            cloned_dict = dict(cloned_items)
        # Module.state_dict _metadata carries per-module version info for load.
        metadata = getattr(state, "_metadata", None)
        if metadata is not None:
            try:
                cloned_dict._metadata = copy.copy(metadata)  # type: ignore[attr-defined]
            except (AttributeError, TypeError):
                pass
        return cloned_dict
    if isinstance(state, (list, tuple)):
        cloned = [_clone_state_dict(v, target_device=target_device) for v in state]
        return type(state)(cloned) if isinstance(state, tuple) else cloned
    return copy.deepcopy(state)


def measure_chunked_steady(
    *,
    model: "nn.Module",
    batch: dict,
    optimizer: "torch.optim.Optimizer",
    n_warmup: int = _PHASE2_N_WARMUP,
    n_iters: int = _PHASE2_N_ITERS,
) -> tuple[float, float, float, int]:
    """Run a chunked steady-state fwd→bwd→step loop; returns (fwd_s, bwd_s, step_s, peak_bytes)."""
    import contextlib

    import torch

    if n_warmup < 0 or n_iters <= 0:
        raise ValueError("n_warmup must be >= 0 and n_iters must be > 0")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Phase-2 measurement requires CUDA; got torch.cuda.is_available() == False"
        )

    # Per-module training flags snapshot for exact restore (top-level recurse would clobber eval submodules).
    module_training: dict[int, bool] = {id(m): m.training for m in model.modules()}
    cpu_rng = torch.get_rng_state()
    cuda_rngs = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    # Bind every CUDA call to the model's device against current-device drift.
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(f"Phase-2 measurement expected a CUDA model, got {device!r}")

    # Match Trainer.autocast so non-quantized fp32 layers (e.g. Qwen3.5 linear_attn.conv1d) accept BF16 activations.
    # Look explicitly for BF16/FP16 params; ignore fp32 (e.g. RMSNorm, Conv1d) and uint8 (bnb 4-bit packed storage).
    autocast_dtype: torch.dtype | None = None
    for _p in model.parameters():
        if _p.dtype in (torch.bfloat16, torch.float16):
            autocast_dtype = _p.dtype
            break
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else contextlib.nullcontext()
    )

    # Sentinels for safe partial-failure restore.
    model_state: dict[str, Any] | None = None
    optim_state: dict[str, Any] | None = None
    # Placeholder keys filtered from snapshot; non-listed missing keys → hard error.
    expected_missing_keys: frozenset[str] = frozenset()
    # bnb 4-bit/8-bit ``Linear`` modules emit companion entries
    # (``weight.absmax``, ``weight.quant_map``, ``weight.nested_absmax``,
    # ``weight.nested_quant_map``, ``weight.quant_state.<algo>``) via
    # ``_save_to_state_dict`` but do NOT consume them in
    # ``_load_from_state_dict`` — ``QuantState`` is stored as a Python
    # attribute on the param, not a registered buffer. The companion
    # bytes round-trip implicitly through ``chunk/manager.py``'s
    # ``param.data``-only rebind path. Track these so the post-restore
    # gate doesn't false-positive on a benign bnb asymmetry.
    expected_unexpected_keys: frozenset[str] = frozenset()
    # state_dict() misses offloaded chunk bytes; use chunk-manager snapshot for Mode-C correctness.
    chunk_state: dict[Any, Any] | None = None
    chunk_manager = getattr(optimizer, "_chunk_manager", None)
    with torch.cuda.device(device):
        try:
            model.train()
            # Deep-clone state before warmup so step() mutations don't advance the snapshot.
            torch.cuda.synchronize(device)
            # Host snapshot to avoid doubling GPU footprint during the timed region.
            # land back on each parameter's original device — no
            # device drift on rollback.
            #
            # Mode-C / PEFT correctness: filter out entries whose live
            # tensor is the zero-element offloaded placeholder
            # (``param.data = empty(0)`` after
            # ``ChunkManager.materialize_offload``). Including those
            # keys is doubly wrong:
            #
            #   * The placeholder carries no real bytes — the actual
            #     weights live in ``chunk_manager._cpu_slots`` /
            #     ``_chunk_shards`` and are restored via the
            #     ``chunk_state`` path below; snapshotting the
            #     placeholder ADDS NOTHING to the rollback contract.
            #   * On restore, ``Module.load_state_dict`` shape-checks
            #     each entry. A snapshot captured AT empty-placeholder
            #     time but a live param momentarily rebound to its
            #     real GPU buffer (e.g. the last timed step left a
            #     chunk gathered, or a PEFT-wrapped backbone has
            #     adapter modules whose ``state_dict`` walks see the
            #     gathered shape) raises
            #     ``RuntimeError: Error(s) in loading state_dict for
            #     PeftModelForCausalLM: size mismatch for ...``.
            #
            # Filtering here keeps the rollback minimal: only
            # GPU-resident persistent / LoRA-adapter / non-chunked
            # tensors round-trip through ``state_dict()`` (where they
            # belong); offloaded chunks round-trip via ``chunk_state``
            # (where their bytes actually live). Pair this with
            # ``strict=False`` on the restore so the skipped keys
            # don't trip ``load_state_dict``'s missing-keys check.
            #
            # ``v.numel() > 0`` is the placeholder filter:
            # ``_empty_placeholder`` returns ``torch.empty(0)`` exactly,
            # so a non-trivial parameter cannot collide with the
            # filter. Buffers whose source-of-truth is ``param.data``
            # already-empty (rare but valid — e.g. a deliberate empty
            # tensor in a checkpoint) are also skipped, but that's
            # benign: if the live tensor is empty before AND after
            # the timed loop, there is nothing to restore.
            full_state = model.state_dict()
            filtered_state = {
                k: v
                for k, v in full_state.items()
                if not torch.is_tensor(v) or v.numel() > 0
            }
            # Track offloaded-placeholder keys to distinguish expected vs real missing in restore.
            expected_missing_keys = frozenset(full_state.keys() - filtered_state.keys())
            # bnb companion-buffer suffix list (post the param name).
            # Pattern: ``<param>.absmax``, ``<param>.quant_map``,
            # ``<param>.nested_absmax``, ``<param>.nested_quant_map``,
            # ``<param>.quant_state.<algo>``. The set is small + stable
            # across bnb >= 0.42 (when QuantState landed) and bnb 0.49.x.
            _BNB_COMPANION_SUFFIXES = (
                ".absmax",
                ".quant_map",
                ".nested_absmax",
                ".nested_quant_map",
            )
            expected_unexpected_keys = frozenset(
                k
                for k in full_state.keys()
                if any(k.endswith(s) for s in _BNB_COMPANION_SUFFIXES)
                or ".quant_state." in k
            )
            model_state = _clone_state_dict(
                filtered_state, target_device=torch.device("cpu")
            )
            # Use _protrain_snapshot_inner_state to bypass the hollow public state_dict shell.
            if hasattr(optimizer, "_protrain_snapshot_inner_state"):
                optim_state = _clone_state_dict(
                    optimizer._protrain_snapshot_inner_state(),
                    target_device=torch.device("cpu"),
                )
            else:
                optim_state = _clone_state_dict(
                    optimizer.state_dict(), target_device=torch.device("cpu")
                )
            # snapshot_cpu_state captures non-persistent chunk bytes the state_dict misses.
            if chunk_manager is not None and hasattr(
                chunk_manager, "snapshot_cpu_state"
            ):
                chunk_state = chunk_manager.snapshot_cpu_state()
            # Snapshot/restore doesn't cover .grad; reject pre-existing grads.
            if any(param.grad is not None for param in model.parameters()):
                raise RuntimeError(
                    "measure_chunked_steady requires gradients to be "
                    "cleared before profiling: at least one parameter "
                    "has a non-None .grad and the helper does not "
                    "snapshot/restore grads. Call "
                    "optimizer.zero_grad(set_to_none=True) on the "
                    "caller side before invoking measure_chunked_steady."
                )
            optimizer.zero_grad(set_to_none=True)
            # Warmup — discard timings.
            for _ in range(n_warmup):
                with autocast_ctx:
                    out = model(**batch)
                loss = _extract_loss(out)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)

            fwd_times_s: list[float] = []
            bwd_times_s: list[float] = []
            step_times_s: list[float] = []
            for _ in range(n_iters):
                fwd_start = torch.cuda.Event(enable_timing=True)
                fwd_end = torch.cuda.Event(enable_timing=True)
                bwd_start = torch.cuda.Event(enable_timing=True)
                bwd_end = torch.cuda.Event(enable_timing=True)
                step_end = torch.cuda.Event(enable_timing=True)

                fwd_start.record()
                with autocast_ctx:
                    out = model(**batch)
                loss = _extract_loss(out)
                fwd_end.record()

                bwd_start.record()
                loss.backward()
                bwd_end.record()
                optimizer.step()
                step_end.record()

                torch.cuda.synchronize(device)
                fwd_times_s.append(fwd_start.elapsed_time(fwd_end) / 1000.0)
                bwd_times_s.append(bwd_start.elapsed_time(bwd_end) / 1000.0)
                step_times_s.append(bwd_end.elapsed_time(step_end) / 1000.0)

                optimizer.zero_grad(set_to_none=True)

            fwd_median = statistics.median(fwd_times_s)
            bwd_median = statistics.median(bwd_times_s)
            step_median = statistics.median(step_times_s)
            peak_bytes = int(torch.cuda.max_memory_allocated(device))
        finally:
            # Defer rollback errors so every restore step runs; re-raise the first at the end.
            restore_error: Exception | None = None
            torch.cuda.synchronize(device)
            if model_state is not None:
                # strict=False tolerates offloaded placeholders; non-listed missing → hard error.
                try:
                    _result = model.load_state_dict(model_state, strict=False)
                    extra_missing = set(_result.missing_keys) - expected_missing_keys
                    if extra_missing:
                        raise RuntimeError(
                            "Phase-2 state_dict restore missed "
                            f"{len(extra_missing)} unexpected keys "
                            f"(first 3: {sorted(extra_missing)[:3]}). "
                            "The live model's state-dict surface "
                            "changed during the timed measurement; "
                            "investigate the harness or the model for "
                            "a parameter add / submodule rebuild."
                        )
                    extra_unexpected = (
                        set(_result.unexpected_keys) - expected_unexpected_keys
                    )
                    if extra_unexpected:
                        # unexpected_keys = snapshot keys absent in live model → incomplete rollback.
                        raise RuntimeError(
                            "Phase-2 state_dict restore saw "
                            f"{len(extra_unexpected)} unexpected snapshot "
                            f"keys (first 3: {sorted(extra_unexpected)[:3]}). "
                            "The live model dropped or renamed state during "
                            "the timed measurement, so rollback is incomplete."
                        )
                except Exception as exc:  # noqa: BLE001 — re-raised below
                    restore_error = restore_error or exc
            # Restore chunk-manager CPU-shadow bytes (paired with param state_dict restore).
            if (
                chunk_state is not None
                and chunk_manager is not None
                and hasattr(chunk_manager, "restore_cpu_state")
            ):
                try:
                    chunk_manager.restore_cpu_state(chunk_state)
                except Exception as exc:  # noqa: BLE001 — re-raised below
                    restore_error = restore_error or exc
            if optim_state is not None:
                # Route through _protrain_restore_inner_state to bypass the hollow public load_state_dict.
                try:
                    if hasattr(optimizer, "_protrain_restore_inner_state"):
                        optimizer._protrain_restore_inner_state(optim_state)
                    else:
                        optimizer.load_state_dict(optim_state)
                    optimizer.zero_grad(set_to_none=True)
                except Exception as exc:  # noqa: BLE001 — re-raised below
                    restore_error = restore_error or exc
            torch.cuda.synchronize(device)
            # RNG restored last (state_dict restore may consume RNG).
            torch.set_rng_state(cpu_rng)
            if cuda_rngs is not None:
                torch.cuda.set_rng_state_all(cuda_rngs)
            # Per-module training flags restored last so nothing else re-flips them.
            for m in model.modules():
                saved = module_training.get(id(m))
                if saved is None:
                    continue
                if saved:
                    m.train()
                else:
                    m.eval()
            if restore_error is not None:
                raise restore_error
    LOG.info(
        "Phase-2 chunked-runtime measurement: "
        "steady_fwd_chunked_wall_s=%.4f (n=%d, samples=%s) "
        "steady_bwd_chunked_wall_s=%.4f (samples=%s) "
        "steady_step_overlap_s=%.4f (samples=%s) "
        "steady_phase2_peak_bytes=%.2f GB",
        fwd_median,
        n_iters,
        ["%.4f" % t for t in fwd_times_s],
        bwd_median,
        ["%.4f" % t for t in bwd_times_s],
        step_median,
        ["%.4f" % t for t in step_times_s],
        peak_bytes / (1 << 30),
    )
    return fwd_median, bwd_median, step_median, peak_bytes


def estimate_per_block_recompute_s(trace: "ProfilerTrace", n_block: int) -> float:
    """Mean per-block forward compute time (≡ recompute under CKPT)."""
    from axolotl.integrations.protrain.cost.runtime import (
        _fwd_compute_time_from_trace,
    )

    if n_block <= 0:
        return 0.0
    t_fwd_total, per_block_compute, _used_measured, _fwd_compute_base = (
        _fwd_compute_time_from_trace(trace)
    )
    if per_block_compute:
        return sum(per_block_compute.values()) / max(1, len(per_block_compute))
    if t_fwd_total > 0.0:
        return t_fwd_total / n_block
    return 0.0


def _extract_loss(out) -> "torch.Tensor":
    """Pull a backwards-able scalar loss out of a HuggingFace forward output."""
    from axolotl.integrations.protrain.profiler.trace import (
        _extract_loss as _trace_extract_loss,
    )

    return _trace_extract_loss(out)


__all__ = [
    "estimate_per_block_recompute_s",
    "measure_chunked_steady",
    "select_bootstrap_config",
]
