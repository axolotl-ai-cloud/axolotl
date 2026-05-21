"""Single-iteration forward/backward trace driver for the ProTrain profiler."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from axolotl.integrations.protrain.profiler.hw_bench import (
    measure_compute_rate,
    measure_cpu_adam,
    measure_gpu_adam,
    measure_nccl,
    measure_pcie,
)
from axolotl.integrations.protrain.profiler.memory_deltas import (
    MemoryDeltaTracker,
    inter_op_delta,
    intra_op_delta,
)
from axolotl.integrations.protrain.profiler.on_demand import OnDemandTensorMgr
from axolotl.integrations.protrain.types import (
    BlockId,
    OpId,
    OpRecord,
    ProfilerConfig,
    ProfilerTrace,
)
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch
    from torch import nn
    from torch.cuda import Event as CudaEvent

LOG = get_logger(__name__)


# fp16 param+grad+fp32 master+m+v under mixed-precision Adam.
DEFAULT_OPTIM_STATE_BYTES_PER_PARAM = 12
DEFAULT_PARAM_GRAD_BYTES_PER_PARAM = 4  # fp16 param + fp16 grad

# Auto-engage on-demand mode when full model state exceeds 60% of device memory.
ON_DEMAND_STATE_BYTES_FRACTION: float = 0.60


@dataclass
class _OpFrame:
    """Mutable per-op bookkeeping while a forward hook pair is live."""

    op_id: OpId
    module_path: str
    qualified_name: str
    shape_signature: tuple[tuple[int, ...], ...]
    block_id: BlockId | None
    is_forward: bool
    pre_peak_bytes: int
    prev_end_peak_bytes: int
    parent_id: int | None = None
    children_peak_contribution: int = 0
    # CUDA Events for per-op timing; elapsed_time read lazily after final sync.
    pre_event: "CudaEvent | None" = None
    post_event: "CudaEvent | None" = None


def _infer_block_id(module_path: str) -> BlockId | None:
    """Extract a transformer-block index from a dotted module path, if present."""
    parts = module_path.split(".")
    for prev, cur in zip(parts, parts[1:], strict=False):
        if prev in {"h", "layers", "blocks", "block", "layer"} and cur.isdigit():
            return BlockId(int(cur))
    return None


def _shape_sig(inputs: Any) -> tuple[tuple[int, ...], ...]:
    """Best-effort input-shape signature. Non-tensor inputs become ``()``."""
    out: list[tuple[int, ...]] = []
    if not isinstance(inputs, (list, tuple)):
        inputs = (inputs,)
    for arg in inputs:
        shape = getattr(arg, "shape", None)
        if shape is not None:
            try:
                out.append(tuple(int(d) for d in shape))
            except TypeError:
                out.append(())
        else:
            out.append(())
    return tuple(out)


def _count_model_state_bytes(
    model: "nn.Module",
    *,
    param_byte_size: int | None = None,
    param_grad_bytes_per_param: int = DEFAULT_PARAM_GRAD_BYTES_PER_PARAM,
    optim_state_bytes_per_param: int = DEFAULT_OPTIM_STATE_BYTES_PER_PARAM,
) -> int:
    """Constant-size model-state footprint: params + grads + optimizer states."""
    trainable_params = 0
    frozen_param_bytes = 0
    for _, p in model.named_parameters():
        n = int(p.numel())
        if p.requires_grad:
            trainable_params += n
        else:
            if param_byte_size is None:
                frozen_param_bytes += n * int(p.element_size())
            else:
                frozen_param_bytes += n * int(param_byte_size)
    return frozen_param_bytes + trainable_params * (
        int(param_grad_bytes_per_param) + int(optim_state_bytes_per_param)
    )


def _arch_hash(model: "nn.Module") -> str:
    """Deterministic hash of model architecture; includes requires_grad to invalidate on freeze toggle."""
    parts: list[str] = [type(model).__name__]
    for name, p in model.named_parameters():
        parts.append(
            f"{name}:{tuple(p.shape)}:{p.dtype}:requires_grad={p.requires_grad}"
        )
    for name, b in model.named_buffers():
        parts.append(f"B:{name}:{tuple(b.shape)}:{b.dtype}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _sku(device: "torch.device | str") -> str:
    import torch

    try:
        return torch.cuda.get_device_name(device)
    except Exception:  # pragma: no cover - defensive
        return "cpu"


def run_trace(
    model: "nn.Module",
    batch: dict,
    cfg: ProfilerConfig,
    *,
    param_grad_bytes_per_param: int = DEFAULT_PARAM_GRAD_BYTES_PER_PARAM,
    optim_state_bytes_per_param: int = DEFAULT_OPTIM_STATE_BYTES_PER_PARAM,
) -> ProfilerTrace:
    """Run a single forward (+optional backward) pass and record memory deltas."""
    import torch

    device = torch.device(cfg.device)
    cuda_available_for_bench = device.type == "cuda" and torch.cuda.is_available()

    # Adam microbenches BEFORE tracker so reserved-but-free bytes fold into the baseline.
    try:
        cpu_adam_bps = measure_cpu_adam()
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("measure_cpu_adam failed (%s); recording 0.0", exc)
        cpu_adam_bps = 0.0
    try:
        if cuda_available_for_bench:
            dev_idx_for_bench = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            gpu_adam_bps = measure_gpu_adam(dev_idx_for_bench)
        else:
            gpu_adam_bps = 0.0
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("measure_gpu_adam failed (%s); recording 0.0", exc)
        gpu_adam_bps = 0.0

    # No empty_cache(): keep reserved-but-free blocks for the traced forward+backward.
    if cuda_available_for_bench:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    tracker = MemoryDeltaTracker(device)
    # Seed baseline at current allocated to exclude resident model weights from first inter-op delta.
    tracker.mark_end(tracker.snapshot().allocated_bytes)

    # --- per-op accumulators -------------------------------------------
    op_records: list[OpRecord] = []
    intra_deltas: dict[OpId, int] = {}
    inter_deltas: dict[OpId, int] = {}
    activation_sizes: dict[BlockId, int] = {}

    # Eager-record / lazy-read cuda.Event pairs per op. Populated by the
    # post-forward hook after recording the "post" event; resolved into
    # ``op_latencies`` (seconds) after ``torch.cuda.synchronize()`` so that
    # ``Event.elapsed_time`` reads never stall the hook path.
    #
    # ``parent_op_id`` is captured at post-hook time and used during lazy
    # resolution to convert each op's INCLUSIVE event-pair elapsed (parent
    # span covers all of its descendants' work) into an EXCLUSIVE
    # self-time. Without that subtraction, ``cost/runtime.py``'s
    # ``_fwd_compute_time_from_trace`` — which sums ``op_latencies`` for
    # every op carrying the same ``block_id`` — double-counts every
    # composite span (block compute grows with nesting depth instead of
    # tracking real runtime, which then poisons CKPT recompute costing).
    pending_events: "list[tuple[OpId, OpId | None, CudaEvent | None, CudaEvent | None]]" = []

    # Stack of in-flight _OpFrames keyed by the calling module id. Submodules
    # fire pre-hooks before their parent's post-hook; a dict keyed on id()
    # matches that LIFO nesting without needing a real stack type.
    live_frames: dict[int, _OpFrame] = {}
    # Ordered list of in-flight module ids in pre-hook arrival order. The
    # top of the stack BEFORE we push a new frame IS the direct parent;
    # used to roll up child inclusive intra into the parent's
    # ``children_peak_contribution`` so each frame reports an EXCLUSIVE
    # intra delta (own allocation work, descendants subtracted).
    live_frame_stack: list[int] = []

    next_op_id = 0

    cuda_available = device.type == "cuda" and torch.cuda.is_available()
    # Bind every ``torch.cuda.Event`` and ``synchronize`` to ``cfg.device``'s
    # index. ``Event()`` infers its device from the ambient
    # ``current_device()`` at construction time, so under multi-GPU or
    # ``CUDA_VISIBLE_DEVICES`` masking a stale current device would silently
    # bind events to the wrong stream and produce bogus ``elapsed_time``
    # Gate device_idx behind cuda_available; consumers are all inside cuda_available branches.
    device_idx: int | None = None
    if cuda_available:
        device_idx = (
            device.index if device.index is not None else torch.cuda.current_device()
        )

    # Path → global BlockId registry from discover_blocks; disambiguates encoder vs decoder trees.
    path_to_global_bid: dict[str, BlockId] = {}
    block_path_prefixes: tuple[str, ...] = ()
    # Maps each BlockId to its forward-order tree (encoder=0, decoder=1).
    block_tree_index: dict[BlockId, int] = {}
    try:
        from axolotl.integrations.protrain.block.layout_rules import (
            block_id_path_map,
            discover_blocks as _discover_blocks_for_trace,
        )

        _trees_for_trace = _discover_blocks_for_trace(model)
        path_to_global_bid = block_id_path_map(model, _trees_for_trace)
        # Descending length: longest-prefix match wins for nested submodules.
        block_path_prefixes = tuple(
            sorted(path_to_global_bid.keys(), key=len, reverse=True)
        )
        _flat_idx = 0
        for _tree in sorted(_trees_for_trace, key=lambda t: t.forward_order):
            for _ in _tree.blocks:
                block_tree_index[BlockId(_flat_idx)] = int(_tree.forward_order)
                _flat_idx += 1
    except Exception as exc:  # pragma: no cover - defensive
        LOG.debug(
            "trace: block_id_path_map unavailable (%s); falling back "
            "to single-tree path-fragment heuristic",
            exc,
        )

    def _resolve_block_id(path: str) -> BlockId | None:
        """Map ``path`` to its global ``BlockId`` via the registry."""
        if block_path_prefixes:
            for prefix in block_path_prefixes:
                if path == prefix or path.startswith(prefix + "."):
                    return path_to_global_bid[prefix]
            return None
        return _infer_block_id(path)

    # Precompute id(module) → path map once so pre-hook does O(1) lookup not O(N_modules) walk.
    path_by_id: dict[int, str] = {}
    for name, candidate in model.named_modules():
        path_by_id[id(candidate)] = name or type(candidate).__name__

    def _module_path(m: "nn.Module") -> str:
        """Dotted path of ``m`` inside ``model`` (root -> '')."""
        cached = path_by_id.get(id(m))
        if cached is not None:
            return cached
        return type(m).__name__  # unreachable in practice

    def _pre_forward(module: "nn.Module", inputs):
        nonlocal next_op_id
        op_id = OpId(next_op_id)
        next_op_id += 1
        # Do NOT reset_peak_memory_stats here — child pre-hooks would clobber parent windows.
        if cuda_available:
            pre_peak_bytes = int(torch.cuda.max_memory_allocated(device))
        else:
            pre_peak_bytes = tracker.snapshot().allocated_bytes
        path = _module_path(module)
        pre_event = None
        if cuda_available:
            with torch.cuda.device(device_idx):
                pre_event = torch.cuda.Event(enable_timing=True)
                pre_event.record()
        parent_id = live_frame_stack[-1] if live_frame_stack else None
        frame = _OpFrame(
            op_id=op_id,
            module_path=path,
            qualified_name=type(module).__name__,
            shape_signature=_shape_sig(inputs),
            block_id=_resolve_block_id(path),
            is_forward=True,
            pre_peak_bytes=pre_peak_bytes,
            prev_end_peak_bytes=tracker.last_end_bytes,
            parent_id=parent_id,
            pre_event=pre_event,
        )
        live_frames[id(module)] = frame
        live_frame_stack.append(id(module))
        # Append at PRE-time for execution order; post-hook fires children-first which is wrong.
        op_records.append(
            OpRecord(
                op_id=frame.op_id,
                module_path=frame.module_path,
                qualified_name=frame.qualified_name,
                shape_signature=frame.shape_signature,
                block_id=frame.block_id,
                is_forward=True,
            )
        )

    def _post_forward(module: "nn.Module", inputs, output):
        frame = live_frames.pop(id(module), None)
        if frame is None:
            return
        # Defensive: re-entrant hooks may not match top of stack.
        if live_frame_stack and live_frame_stack[-1] == id(module):
            live_frame_stack.pop()
        elif id(module) in live_frame_stack:
            live_frame_stack.remove(id(module))
        # Re-sample cumulative peak; intra becomes exclusive by subtracting child rollup.
        if cuda_available:
            post_peak_bytes = int(torch.cuda.max_memory_allocated(device))
        else:
            post_peak_bytes = tracker.snapshot().allocated_bytes
        intra_inclusive = intra_op_delta(frame.pre_peak_bytes, post_peak_bytes)
        if frame.parent_id is not None:
            parent = live_frames.get(frame.parent_id)
            if parent is not None:
                parent.children_peak_contribution += intra_inclusive
        intra = max(0, intra_inclusive - frame.children_peak_contribution)
        inter = inter_op_delta(frame.prev_end_peak_bytes, post_peak_bytes)
        tracker.mark_end(post_peak_bytes)

        if cuda_available and frame.pre_event is not None:
            with torch.cuda.device(device_idx):
                post_event = torch.cuda.Event(enable_timing=True)
                post_event.record()
            # Capture parent op_id for exclusive self-time computation; parent frame is still alive.
            parent_op_id: "OpId | None" = None
            if frame.parent_id is not None:
                parent_frame = live_frames.get(frame.parent_id)
                if parent_frame is not None:
                    parent_op_id = parent_frame.op_id
            pending_events.append(
                (frame.op_id, parent_op_id, frame.pre_event, post_event)
            )

        intra_deltas[frame.op_id] = intra
        inter_deltas[frame.op_id] = inter

        # Record block-output bytes only at the canonical block-root to avoid double-counting.
        if frame.block_id is not None:
            is_block_root = (
                not path_to_global_bid
                or path_to_global_bid.get(frame.module_path) == frame.block_id
            )
            if is_block_root:
                out_bytes = _output_bytes(output)
                activation_sizes[frame.block_id] = max(
                    activation_sizes.get(frame.block_id, 0), out_bytes
                )

    def _output_bytes(output: Any) -> int:
        total = 0
        stack: list[Any] = [output]
        while stack:
            item = stack.pop()
            if isinstance(item, torch.Tensor):
                total += item.numel() * item.element_size()
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
            elif isinstance(item, dict):
                stack.extend(item.values())
        return total

    # Decide on-demand engagement before warmups; 13B+ models OOM on un-offloaded forward.
    engage_on_demand = False
    if cfg.force_all_persistent:
        # force_all_persistent bypasses the on-demand gate to honour Mode A.
        LOG.info(
            "Profiler force_all_persistent=True; skipping on-demand "
            "engagement gate. Trace pass will run the trainable "
            "forward+backward fully on GPU."
        )
    elif cfg.on_demand and cuda_available:
        try:
            gpu_total = int(torch.cuda.get_device_properties(device).total_memory)
            # Full model state (params + grads + Adam) — params alone misses optimizer state which OOMs warmup.
            state_bytes = _count_model_state_bytes(
                model,
                param_grad_bytes_per_param=param_grad_bytes_per_param,
                optim_state_bytes_per_param=optim_state_bytes_per_param,
            )
            if state_bytes > ON_DEMAND_STATE_BYTES_FRACTION * gpu_total:
                engage_on_demand = True
                LOG.info(
                    "Profiler engaging on-demand mode: model state=%.2f GB "
                    "(param + grad + optim) exceeds %.0f%% of %.2f GB device "
                    "memory; offloading params + saved-for-backward tensors "
                    "to CPU between modules.",
                    state_bytes / 1e9,
                    ON_DEMAND_STATE_BYTES_FRACTION * 100,
                    gpu_total / 1e9,
                )
        except Exception as exc:  # pragma: no cover - defensive
            LOG.debug(
                "On-demand size check failed (%s); falling back to fast path",
                exc,
            )

    # Warmup passes JIT-compile kernels so steady-state per-op latencies aren't cold.
    N_WARMUP = 0 if engage_on_demand else 2
    if cuda_available and N_WARMUP > 0:
        for _i in range(N_WARMUP):
            try:
                torch.cuda.synchronize(device)
                warm_out = model(**batch)
                if cfg.include_backward:
                    warm_loss = _extract_loss(warm_out)
                    warm_loss.backward()
                    model.zero_grad(set_to_none=True)
                del warm_out
                torch.cuda.synchronize(device)
                # No empty_cache: keep warm allocator state for the traced iter.
            except Exception as exc:  # pragma: no cover - defensive
                LOG.debug("profiler warmup pass failed (%s); continuing cold", exc)
                # On warmup OOM, clear cache for a clean steady-state baseline.
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001 — best-effort cleanup
                    pass
                break

    # Steady-state wall captured pre-hook for hook-dispatch calibration scale.
    # Per-block peak hooks (block-granularity, not per-leaf) are cheap and give the cost model a ground-truth cap.
    steady_fwd_wall_s = 0.0
    steady_bwd_wall_s = 0.0
    steady_fwd_peak_bytes = 0
    steady_bwd_peak_bytes = 0
    steady_fwd_block_peak_bytes: dict[BlockId, int] = {}
    steady_bwd_block_peak_bytes: dict[BlockId, int] = {}
    # Skip steady-state when on-demand engaged; cost model falls back to identity scale.
    if cuda_available and not engage_on_demand:
        # Skip per-block capture when block discovery fails on non-standard shapes.
        block_handles: list[Any] = []
        try:
            from axolotl.integrations.protrain.block.layout_rules import (
                discover_blocks,
                flatten_block_trees,
            )

            blocks = flatten_block_trees(discover_blocks(model))
        except Exception as exc:  # pragma: no cover - defensive
            LOG.debug(
                "profiler: discover_blocks failed (%s); skipping per-block "
                "peak capture, aggregate cap only",
                exc,
            )
            blocks = []

        # Whole-iter peak recovered post-iter as max(iter_block_peaks); no extra reads in hot pre-hook path.
        iter_block_peaks: list[int] = []

        def _make_pre(_dev):
            def _pre(_mod, _inputs):
                torch.cuda.reset_peak_memory_stats(_dev)

            return _pre

        def _make_post(bid, _dev):
            def _post(_mod, _inputs, _output):
                block_peak = int(torch.cuda.max_memory_allocated(_dev))
                steady_fwd_block_peak_bytes[bid] = max(
                    steady_fwd_block_peak_bytes.get(bid, 0), block_peak
                )
                iter_block_peaks.append(block_peak)

            return _post

        # Backward per-block peak hooks; asymmetric with forward because lm_head bwd fires before first block.
        # that peak. To preserve it, ``_bwd_pre`` reads
        # ``max_memory_allocated`` IMMEDIATELY BEFORE resetting and
        # rolls the reading into ``iter_bwd_pre_peaks``, capturing any
        # pre-block-bwd peak (lm_head, CE, embedding-grad reductions
        # that fire late, etc.). One extra allocator read per block
        # per backward — negligible relative to the backward kernel
        # cost itself.
        iter_bwd_block_peaks: list[int] = []
        iter_bwd_pre_peaks: list[int] = []

        def _make_bwd_pre(_dev):
            def _bwd_pre(_mod, _grad_output):
                # Capture cumulative peak BEFORE the reset so peaks from
                # bwd ops that ran prior to this block (lm_head, CE
                # loss, neighbouring blocks' transients) are not lost.
                iter_bwd_pre_peaks.append(int(torch.cuda.max_memory_allocated(_dev)))
                torch.cuda.reset_peak_memory_stats(_dev)

            return _bwd_pre

        def _make_bwd_post(bid, _dev):
            def _bwd_post(_mod, _grad_input, _grad_output):
                block_peak = int(torch.cuda.max_memory_allocated(_dev))
                steady_bwd_block_peak_bytes[bid] = max(
                    steady_bwd_block_peak_bytes.get(bid, 0), block_peak
                )
                iter_bwd_block_peaks.append(block_peak)

            return _bwd_post

        for idx, block in enumerate(blocks):
            bid = BlockId(idx)
            block_handles.append(block.register_forward_pre_hook(_make_pre(device)))
            block_handles.append(block.register_forward_hook(_make_post(bid, device)))
            # Backward hooks are best-effort: only fire when the block has
            # at least one tensor input that requires grad. For tiny test
            # models that pass non-grad inputs they're a no-op — the
            # forward-only peak still suffices as the bound.
            block_handles.append(
                block.register_full_backward_pre_hook(_make_bwd_pre(device))
            )
            block_handles.append(
                block.register_full_backward_hook(_make_bwd_post(bid, device))
            )

        # Multi-iter hot-loop measurement. A single forward still carries
        # allocator-settle cost that a real steady-state training loop
        # wouldn't pay. Run N=4 un-hooked iters and take the median of
        # iters 2-3 as the steady value; iter 0/1 soak up any residual
        # warmup. Per-block peak bytes take the max across all measured
        # iters to capture the true high-water mark.
        # Best-effort steady backward: runs inside the same loop (after
        # each forward) IFF the trace config allows it. Backward on a
        # 7B-class model without chunking engaged will OOM, so guard
        # with try/except per-iter and fall back to 0.0 on any failure
        # (cost model then uses the default bwd_fwd ratio).
        N_STEADY_ITERS = 4
        N_STEADY_WARMUP = 2
        fwd_iter_s: list[float] = []
        bwd_iter_s: list[float] = []
        try:
            for i in range(N_STEADY_ITERS):
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                # Clear the per-iter block-peak collector; the per-block
                # post-hooks below will append each block's peak as they
                # fire and the whole-iter aggregate is recovered as
                # ``max(iter_block_peaks)`` AFTER the forward completes.
                iter_block_peaks.clear()
                with torch.cuda.device(device_idx):
                    pre_sf = torch.cuda.Event(enable_timing=True)
                    post_sf = torch.cuda.Event(enable_timing=True)
                    pre_sf.record()
                steady_out = model(**batch)
                with torch.cuda.device(device_idx):
                    post_sf.record()
                torch.cuda.synchronize(device)
                fwd_iter_s.append(pre_sf.elapsed_time(post_sf) / 1000.0)
                # Recover whole-iter peak via max(per-block peaks); avoids extra hot-path reads.
                whole_iter_peak = max(iter_block_peaks) if iter_block_peaks else 0
                steady_fwd_peak_bytes = max(
                    steady_fwd_peak_bytes,
                    whole_iter_peak,
                    int(torch.cuda.max_memory_allocated(device)),
                )

                if cfg.include_backward:
                    try:
                        steady_loss = _extract_loss(steady_out)
                        torch.cuda.synchronize(device)
                        iter_bwd_block_peaks.clear()
                        iter_bwd_pre_peaks.clear()
                        torch.cuda.reset_peak_memory_stats(device)
                        with torch.cuda.device(device_idx):
                            pre_sb = torch.cuda.Event(enable_timing=True)
                            post_sb = torch.cuda.Event(enable_timing=True)
                            pre_sb.record()
                        steady_loss.backward()
                        with torch.cuda.device(device_idx):
                            post_sb.record()
                        torch.cuda.synchronize(device)
                        bwd_iter_s.append(pre_sb.elapsed_time(post_sb) / 1000.0)
                        # Combine per-block post/pre peaks + counter snapshot for the iter high-water.
                        whole_bwd_iter_peak = (
                            max(iter_bwd_block_peaks) if iter_bwd_block_peaks else 0
                        )
                        bwd_pre_peak = (
                            max(iter_bwd_pre_peaks) if iter_bwd_pre_peaks else 0
                        )
                        steady_bwd_peak_bytes = max(
                            steady_bwd_peak_bytes,
                            whole_bwd_iter_peak,
                            bwd_pre_peak,
                            int(torch.cuda.max_memory_allocated(device)),
                        )
                        model.zero_grad(set_to_none=True)
                    except Exception as bwd_exc:  # pragma: no cover
                        LOG.debug(
                            "profiler steady backward iter %d failed (%s); "
                            "cost model falls back to bwd_fwd ratio",
                            i,
                            bwd_exc,
                        )
                        bwd_iter_s.clear()  # drop partial measurements
                        # Drop partial per-block bwd peaks and grads so the next iter starts clean.
                        steady_bwd_block_peak_bytes.clear()
                        steady_bwd_peak_bytes = 0
                        model.zero_grad(set_to_none=True)
                del steady_out
                torch.cuda.synchronize(device)

            # Steady value = median of post-warmup iters.
            import statistics

            steady_slice = fwd_iter_s[N_STEADY_WARMUP:]
            if steady_slice:
                steady_fwd_wall_s = statistics.median(steady_slice)
            bwd_slice = bwd_iter_s[N_STEADY_WARMUP:] if bwd_iter_s else []
            if bwd_slice:
                steady_bwd_wall_s = statistics.median(bwd_slice)
            # No empty_cache: keep warm allocator state for the hooked trace.
        except Exception as exc:  # pragma: no cover - defensive
            LOG.debug(
                "profiler hook-less steady-state measurement failed (%s); "
                "cost model will fall back to identity scale",
                exc,
            )
            steady_fwd_wall_s = 0.0
            steady_bwd_wall_s = 0.0
            steady_fwd_peak_bytes = 0
            steady_bwd_peak_bytes = 0
            steady_fwd_block_peak_bytes = {}
            steady_bwd_block_peak_bytes = {}
        finally:
            for h in block_handles:
                h.remove()

    # --- install hooks on every nn.Module (leaves + composites) --------
    handles: list[Any] = []
    for sub in model.modules():
        handles.append(sub.register_forward_pre_hook(_pre_forward))
        handles.append(sub.register_forward_hook(_post_forward))

    model_state_bytes = _count_model_state_bytes(
        model,
        param_grad_bytes_per_param=param_grad_bytes_per_param,
        optim_state_bytes_per_param=optim_state_bytes_per_param,
    )

    # Wrapper honours engage decision; fast path stays a no-op context.
    on_demand_mgr = OnDemandTensorMgr(
        device=device, disabled=not engage_on_demand, model=model
    )

    # Event-timed hooked forward wall: captures dispatch gaps the per-op sum misses.
    hooked_fwd_wall_s = 0.0
    hooked_fwd_pre_event = None
    hooked_fwd_post_event = None

    try:
        if cuda_available:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            # Re-seed baseline after peak reset so first op's inter delta excludes resident weights.
            tracker.mark_end(int(torch.cuda.max_memory_allocated(device)))
        with on_demand_mgr:
            if cuda_available:
                with torch.cuda.device(device_idx):
                    hooked_fwd_pre_event = torch.cuda.Event(enable_timing=True)
                    hooked_fwd_pre_event.record()
            output = model(**batch)
            if cuda_available and hooked_fwd_pre_event is not None:
                with torch.cuda.device(device_idx):
                    hooked_fwd_post_event = torch.cuda.Event(enable_timing=True)
                    hooked_fwd_post_event.record()

            if cfg.include_backward:
                loss = _extract_loss(output)
                # Synthetic backward op for intra/inter maps; fwd ops then bwd ops.
                next_op_id_local = next_op_id
                bwd_op_id = OpId(next_op_id_local)
                next_op_id = next_op_id_local + 1
                # Reset CUDA peak counter so the bwd snapshot reflects only bwd.
                if cuda_available:
                    torch.cuda.reset_peak_memory_stats(device)
                before = tracker.snapshot()
                prev_end = tracker.last_end_bytes
                bwd_pre_event = None
                if cuda_available:
                    with torch.cuda.device(device_idx):
                        bwd_pre_event = torch.cuda.Event(enable_timing=True)
                        bwd_pre_event.record()
                loss.backward()
                if cuda_available and bwd_pre_event is not None:
                    with torch.cuda.device(device_idx):
                        bwd_post_event = torch.cuda.Event(enable_timing=True)
                        bwd_post_event.record()
                    # Top-level synthetic op: inclusive == exclusive (no children).
                    pending_events.append(
                        (bwd_op_id, None, bwd_pre_event, bwd_post_event)
                    )
                snap = tracker.snapshot()
                intra_deltas[bwd_op_id] = intra_op_delta(
                    before.allocated_bytes, snap.peak_allocated_bytes
                )
                inter_deltas[bwd_op_id] = inter_op_delta(
                    prev_end, snap.peak_allocated_bytes
                )
                tracker.mark_end(snap.allocated_bytes)
                op_records.append(
                    OpRecord(
                        op_id=bwd_op_id,
                        module_path="<backward>",
                        qualified_name="<backward>",
                        shape_signature=(),
                        block_id=None,
                        is_forward=False,
                    )
                )
                # Release loss + autograd graph before PCIe/compute probes to avoid OOM fallback.
                del loss
        del output
        # Clear grad tensors so probes don't see a model-sized inflated baseline.
        model.zero_grad(set_to_none=True)
        if cuda_available:
            torch.cuda.synchronize(device)
    finally:
        for h in handles:
            h.remove()

    # Resolve event pairs to exclusive self-time (parent elapsed minus child rollup).
    op_latencies: dict[OpId, float] = {}
    if cuda_available:
        inclusive_ms: dict[OpId, float] = {}
        children_ms: dict[OpId, float] = {}
        for op_id, parent_op_id, pre_ev, post_ev in pending_events:
            if pre_ev is None or post_ev is None:
                continue
            try:
                elapsed_ms = pre_ev.elapsed_time(post_ev)
            except Exception as exc:  # pragma: no cover - defensive
                LOG.debug("Event.elapsed_time failed for op %s: %s", op_id, exc)
                continue
            if elapsed_ms < 0:
                continue
            inclusive_ms[op_id] = elapsed_ms
            if parent_op_id is not None:
                children_ms[parent_op_id] = (
                    children_ms.get(parent_op_id, 0.0) + elapsed_ms
                )
        for op_id, elapsed_ms in inclusive_ms.items():
            self_ms = elapsed_ms - children_ms.get(op_id, 0.0)
            if self_ms < 0.0:
                self_ms = 0.0
            op_latencies[op_id] = self_ms / 1000.0

        # Whole-forward hooked wall from the wrapping events.
        if hooked_fwd_pre_event is not None and hooked_fwd_post_event is not None:
            try:
                hooked_fwd_wall_s = (
                    hooked_fwd_pre_event.elapsed_time(hooked_fwd_post_event) / 1000.0
                )
            except Exception as exc:  # pragma: no cover - defensive
                LOG.debug("hooked forward Event.elapsed_time failed: %s", exc)
                hooked_fwd_wall_s = 0.0

    # PCIe measured post-trace; copy engines unaffected by Adam benches.
    try:
        if cuda_available:
            dev_idx = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            pcie_h2d_bps, pcie_d2h_bps = measure_pcie(dev_idx)
        else:
            pcie_h2d_bps = pcie_d2h_bps = 0.0
    except Exception as exc:  # pragma: no cover - defensive, GPU-only
        LOG.warning("measure_pcie failed (%s); recording zeros", exc)
        pcie_h2d_bps = pcie_d2h_bps = 0.0

    # Trainable-param fraction drives cost model's bwd/fwd-ratio fallback (LoRA ~1x, full-FT ~2x).
    try:
        n_trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
        n_total = sum(int(p.numel()) for p in model.parameters())
        trainable_param_fraction = n_trainable / n_total if n_total > 0 else 0.0
    except Exception as exc:  # pragma: no cover - defensive
        LOG.debug("trainable_param_fraction probe failed (%s)", exc)
        trainable_param_fraction = 0.0

    # Per-SKU compute rate enables cross-SKU per-op latency scaling.
    try:
        if cuda_available:
            dev_idx_for_compute = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            compute_rate_tflops = measure_compute_rate(dev_idx_for_compute)
        else:
            compute_rate_tflops = 0.0
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning(
            "measure_compute_rate failed (%s); recording 0.0 — cost model "
            "will skip SKU calibration",
            exc,
        )
        compute_rate_tflops = 0.0

    resolved_world = cfg.world_size
    if resolved_world is None:
        try:
            import torch.distributed as _dist

            resolved_world = _dist.get_world_size() if _dist.is_initialized() else 1
        except Exception:  # noqa: BLE001 - defensive
            resolved_world = 1

    try:
        gather_table, reduce_table = measure_nccl(world_size=resolved_world)
    except Exception as exc:  # pragma: no cover - distributed-only paths
        LOG.warning(
            "measure_nccl failed (%s); recording empty collective tables. "
            "Cost model's communication term will degrade to 0.",
            exc,
        )
        gather_table, reduce_table = ({}, {})

    return ProfilerTrace(
        op_order=tuple(op_records),
        intra_op_delta=intra_deltas,
        inter_op_delta=inter_deltas,
        activation_sizes=activation_sizes,
        model_state_bytes=model_state_bytes,
        pcie_h2d_bps=pcie_h2d_bps,
        pcie_d2h_bps=pcie_d2h_bps,
        nccl_gather_s=gather_table,
        nccl_reduce_s=reduce_table,
        arch_hash=_arch_hash(model),
        bs=cfg.batch_size,
        seq=cfg.seq_len,
        sku=_sku(device),
        world=resolved_world,
        op_latencies=op_latencies,
        cpu_adam_bytes_per_sec=cpu_adam_bps,
        gpu_adam_bytes_per_sec=gpu_adam_bps,
        hooked_fwd_wall_s=hooked_fwd_wall_s,
        steady_fwd_wall_s=steady_fwd_wall_s,
        steady_bwd_wall_s=steady_bwd_wall_s,
        steady_fwd_peak_bytes=steady_fwd_peak_bytes,
        steady_bwd_peak_bytes=steady_bwd_peak_bytes,
        steady_fwd_block_peak_bytes=steady_fwd_block_peak_bytes,
        steady_bwd_block_peak_bytes=steady_bwd_block_peak_bytes,
        compute_rate_tflops=compute_rate_tflops,
        trainable_param_fraction=trainable_param_fraction,
        block_tree_index=block_tree_index,
    )


def _extract_loss(output: Any) -> "torch.Tensor":
    """Pull a scalar loss out of a HuggingFace-style output or raw tensor."""
    import torch

    loss = getattr(output, "loss", None)
    if isinstance(loss, torch.Tensor):
        return loss
    if isinstance(output, dict) and isinstance(output.get("loss"), torch.Tensor):
        return output["loss"]
    if isinstance(output, torch.Tensor):
        return output.sum()
    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.dim() == 0:
                return item
        # fall back to summing the first tensor we can find
        for item in output:
            if isinstance(item, torch.Tensor):
                return item.sum()
    raise TypeError(
        f"run_trace: unable to extract a loss from output of type {type(output)}"
    )


def _infer_hidden_size(model: "nn.Module") -> int:
    """Best-effort hidden-size inference; falls back to 2048 so synthetic SWAP slot sizing stays finite."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("hidden_size", "d_model", "n_embd"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int) and v > 0:
                return v
    return 2048


def _infer_intermediate_size(model: "nn.Module", hidden_size: int) -> int:
    """Best-effort FFN intermediate size; sized larger than hidden so synthetic SWAP slot sizing doesn't under-shoot the largest saved activation."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("intermediate_size", "ffn_hidden_size", "d_ff", "n_inner"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int) and v > 0:
                return v
    return 4 * int(hidden_size)


def synth_trace_from_overrides(
    model: "nn.Module",
    *,
    batch_size: int,
    seq_len: int,
    device: "torch.device | str",
    world_size: int,
    measure_pcie_bps: bool = True,
    param_grad_bytes_per_param: int = DEFAULT_PARAM_GRAD_BYTES_PER_PARAM,
    optim_state_bytes_per_param: int = DEFAULT_OPTIM_STATE_BYTES_PER_PARAM,
) -> ProfilerTrace:
    """Synthesize a minimal ProfilerTrace for the explicit-override skip path."""
    import torch

    from axolotl.integrations.protrain.block.layout_rules import (
        block_id_path_map,
        discover_blocks,
        flatten_block_trees,
    )

    dev = torch.device(device) if not isinstance(device, torch.device) else device

    try:
        trees = discover_blocks(model)
        blocks = flatten_block_trees(trees)
        block_count = max(1, len(blocks))
        path_map = block_id_path_map(model, trees)
        block_tree_index: dict[BlockId, int] = {}
        flat_idx = 0
        for tree in sorted(trees, key=lambda t: t.forward_order):
            for _ in tree.blocks:
                block_tree_index[BlockId(flat_idx)] = int(tree.forward_order)
                flat_idx += 1
        # path_map sanity check only.
        del path_map
    except Exception as exc:  # pragma: no cover - defensive
        LOG.debug(
            "synth_trace_from_overrides: discover_blocks failed (%s); "
            "falling back to single-block placeholder",
            exc,
        )
        block_count = 1
        block_tree_index = {BlockId(0): 0}

    hidden_size = _infer_hidden_size(model)
    intermediate_size = _infer_intermediate_size(model, hidden_size)
    # Size off FFN intermediate: dominates block-output for autograd's saved tensors.
    per_block_act_bytes = int(batch_size) * int(seq_len) * int(intermediate_size) * 2
    activation_sizes: dict[BlockId, int] = {
        BlockId(i): per_block_act_bytes for i in range(block_count)
    }

    model_state_bytes = _count_model_state_bytes(
        model,
        param_grad_bytes_per_param=param_grad_bytes_per_param,
        optim_state_bytes_per_param=optim_state_bytes_per_param,
    )

    # Conservative PCIe Gen3 fallback prior.
    pcie_h2d_bps = 13e9
    pcie_d2h_bps = 13e9
    if measure_pcie_bps and dev.type == "cuda" and torch.cuda.is_available():
        try:
            dev_idx = (
                dev.index if dev.index is not None else torch.cuda.current_device()
            )
            pcie_h2d_bps, pcie_d2h_bps = measure_pcie(int(dev_idx))
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning(
                "synth_trace_from_overrides: measure_pcie failed (%s); "
                "falling back to 13 GB/s Gen3 prior",
                exc,
            )

    return ProfilerTrace(
        op_order=(),
        intra_op_delta={},
        inter_op_delta={},
        activation_sizes=activation_sizes,
        model_state_bytes=int(model_state_bytes),
        pcie_h2d_bps=float(pcie_h2d_bps),
        pcie_d2h_bps=float(pcie_d2h_bps),
        nccl_gather_s={},
        nccl_reduce_s={},
        arch_hash=_arch_hash(model),
        bs=int(batch_size),
        seq=int(seq_len),
        sku=_sku(dev),
        world=int(world_size),
        op_latencies={},
        cpu_adam_bytes_per_sec=0.0,
        gpu_adam_bytes_per_sec=0.0,
        block_tree_index=block_tree_index,
    )


__all__ = ["run_trace", "synth_trace_from_overrides"]
