"""Optimizer-state checkpoint/resume for the ProTrain runtime."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from typing import TYPE_CHECKING, Any

import torch

from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from transformers.trainer_callback import (
        TrainerCallback,
        TrainerControl,
        TrainerState,
    )
    from transformers.training_args import TrainingArguments

LOG = get_logger(__name__)

PROTRAIN_OPTIM_DIRNAME = "protrain_optim"
METADATA_FILENAME = "metadata.json"
GPU_OPTIM_FILENAME = "gpu_optim.pt"
CPU_OPTIM_DIRNAME = "cpu_optim"
# Mode-B: chunk_<N>.pt (no rank suffix). Mode-C: chunk_<N>_rank_<R>.pt.
CHUNK_FILE_RE = re.compile(r"^chunk_(\d+)\.pt$")
CHUNK_SHARD_FILE_RE = re.compile(r"^chunk_(\d+)_rank_(\d+)\.pt$")
# v3 persistent partition: gpu_optim_rank_<R>.pt (one per rank).
GPU_OPTIM_RANK_FILE_RE = re.compile(r"^gpu_optim_rank_(\d+)\.pt$")
SCHEMA_FORMAT_VERSION = 4
SAVE_MODE_REPLICATED = "replicated"
SAVE_MODE_SHARDED = "sharded"
DEFAULT_SAVE_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB; mirrors args.py default

# Sentinel grepped by external bench gating; see CHECKPOINT_DESIGN_PHASE2.md §13.
_CROSS_WORLD_NCCL_CPU_BRIDGE = "v1"

# torch.dtype -> str(dtype) round-trip. JSON cannot serialize dtype
# objects directly, and pickling them defeats the "human-readable
# metadata" goal. We persist ``str(dtype)`` (e.g. "torch.float16") and
# convert back on load via this mapping. Only dtypes that can land in a
# DtypeRegion (i.e. anything ChunkLayout might bundle) need an entry.
_DTYPE_NAME_TO_TORCH: dict[str, "torch.dtype"] = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float": torch.float32,
    "torch.half": torch.float16,
    "torch.double": torch.float64,
}


# ---------------------------------------------------------------------------
# Distributed helpers — no-op on single-rank
# ---------------------------------------------------------------------------


def _dist_is_active() -> bool:
    return bool(torch.distributed.is_available() and torch.distributed.is_initialized())


def _broadcast_object_list_or_noop(obj_list: list, src: int = 0) -> None:
    """Broadcast a list of picklable objects from ``src`` to every rank."""
    if not _dist_is_active():
        return
    torch.distributed.broadcast_object_list(obj_list, src=src)


def _barrier_or_noop() -> None:
    """``dist.barrier()`` if dist is active; else no-op."""
    if not _dist_is_active():
        return
    torch.distributed.barrier()


def _dist_status_tensor(status: int) -> torch.Tensor:
    """Build a 0/1 status tensor on the right device for the active backend."""
    return _dist_backend_tensor(int(status), dtype=torch.int64)


def _dist_backend_tensor(value: int, *, dtype: torch.dtype) -> torch.Tensor:
    """Allocate a scalar collective payload on the device the active PG supports.

    Under accelerate's ``MULTI_GPU`` distributed_type the default PG is
    NCCL-only, and passing a CPU tensor to ``dist.all_reduce``/``all_gather``
    raises ``RuntimeError: No backend type associated with device type cpu``.
    """
    device = torch.device("cpu")
    if _dist_is_active() and torch.distributed.get_backend() == "nccl":
        device = torch.device("cuda", torch.cuda.current_device())
    return torch.tensor([value], dtype=dtype, device=device)


def _broadcast_status_or_raise(status: int, *, src: int, op: str) -> None:
    """Broadcast a 0/1 status flag from ``src``; raise on every rank if non-zero."""
    if not _dist_is_active():
        return
    flag = _dist_status_tensor(status)
    torch.distributed.broadcast(flag, src=src)
    if int(flag.item()) != 0:
        my_rank = int(torch.distributed.get_rank())
        if my_rank == src:
            # Source rank raises its own original exception in the caller's
            # ``finally``-bracketed try/except; do not stomp on it here.
            return
        raise RuntimeError(
            f"ProTrain optimizer {op}: rank {src} failed during the "
            "single-rank-writes phase (see rank "
            f"{src}'s traceback for the underlying error). Aborting on "
            f"rank {my_rank} so the cluster fails in lockstep instead of "
            "deadlocking on the trailing barrier."
        )


def _allreduce_status_or_raise(status: int, *, op: str) -> None:
    """All-reduce SUM a status flag; raise everywhere if any rank failed."""
    if not _dist_is_active():
        return
    flag = _dist_status_tensor(status)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.SUM)
    total = int(flag.item())
    if total != 0:
        my_rank = int(torch.distributed.get_rank())
        if status != 0:
            # Local rank raises its own original exception in the caller's
            # try/except; do not stomp on it here.
            return
        raise RuntimeError(
            f"ProTrain optimizer {op}: {total} rank(s) failed during the "
            f"per-rank phase (see those ranks' tracebacks for the "
            f"underlying error). Aborting on rank {my_rank} so the cluster "
            "fails in lockstep instead of deadlocking on the trailing barrier."
        )


def _allreduce_visibility_consensus(present: bool, *, what: str, path: str) -> bool:
    """Reach cross-rank consensus on whether a path is visible; mixed → raise everywhere."""
    if not _dist_is_active():
        return bool(present)
    flag = _dist_status_tensor(1 if present else 0)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.SUM)
    total = int(flag.item())
    world = int(torch.distributed.get_world_size())
    if total == 0:
        return False
    if total == world:
        return True
    my_rank = int(torch.distributed.get_rank())
    raise RuntimeError(
        f"ProTrain optimizer load: {what} {path!r} is visible on "
        f"{total}/{world} ranks (rank {my_rank} reports "
        f"{'present' if present else 'absent'}). This usually means "
        "``output_dir`` is not actually a shared filesystem across all "
        "ranks, so some ranks would skip the ProTrain shard while others "
        "restore it -- a silent split-brain. Refusing to load; aborting "
        "on every rank so the cluster fails in lockstep."
    )


def _read_metadata_lockstep(path: str) -> dict[str, Any]:
    """Read + parse ``metadata.json`` with cluster-wide failure synchronization."""
    status = 0
    captured_exc: Exception | None = None
    metadata: dict[str, Any] | None = None
    try:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise RuntimeError(
                f"ProTrain optimizer load: metadata at {path!r} is not a JSON object."
            )
        metadata = loaded
    except Exception as exc:
        status = 1
        captured_exc = exc
    try:
        _allreduce_status_or_raise(status, op="load (metadata read)")
    except Exception:
        # Local failures fall through to captured re-raise so the original traceback wins.
        if captured_exc is None:
            raise
    if captured_exc is not None:
        raise captured_exc
    assert metadata is not None
    # Cross-rank fingerprint to catch divergent per-node metadata (split-brain on non-shared FS).
    if _dist_is_active():
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        gathered: list[str] = [""] * int(torch.distributed.get_world_size())
        torch.distributed.all_gather_object(gathered, payload)
        if any(item != gathered[0] for item in gathered[1:]):
            raise RuntimeError(
                f"ProTrain optimizer load: metadata at {path!r} differs across ranks. "
                "This usually means the checkpoint path is not a single shared tree."
            )
    return metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _current_world_size() -> int:
    """Return the active ``torch.distributed`` world size, or 1 if uninitialized."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    return 1


def _effective_persistent_ids(chunk_manager: Any) -> list[int]:
    """Sorted list of persistent ChunkIds — the post-non-block-pin set."""
    return sorted(int(cid) for cid in chunk_manager._persistent_ids)


def _build_layout_fingerprint(
    chunk_manager: Any, world_size: int, zero3_shard: bool
) -> dict[str, Any]:
    """Raw fingerprint dict whose SHA-256 is :func:`_layout_signature`."""
    layout = chunk_manager.layout
    return {
        "S_chunk": int(layout.S_chunk),
        "N_chunk": int(layout.N_chunk),
        "chunks": [list(map(str, c)) for c in layout.chunks],
        "persistent_ids": _effective_persistent_ids(chunk_manager),
        "world_size": int(world_size),
        "zero3_shard": bool(zero3_shard),
    }


def _layout_signature_from_fingerprint(fingerprint: dict[str, Any]) -> str:
    """SHA-256 over a layout fingerprint dict (deterministic, JSON-canonical)."""
    payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _layout_signature(chunk_manager: Any, world_size: int, zero3_shard: bool) -> str:
    """SHA-256 over load-bearing layout fields; Mode-B is world-size-independent."""
    if not zero3_shard:
        # Replicated: drop world_size so signature is rank-count-independent.
        layout = chunk_manager.layout
        fp = {
            "S_chunk": int(layout.S_chunk),
            "N_chunk": int(layout.N_chunk),
            "chunks": [list(map(str, c)) for c in layout.chunks],
            "persistent_ids": _effective_persistent_ids(chunk_manager),
            "zero3_shard": False,
        }
        return _layout_signature_from_fingerprint(fp)
    return _layout_signature_from_fingerprint(
        _build_layout_fingerprint(chunk_manager, world_size, zero3_shard)
    )


def _estimate_optim_state_bytes(optim: Any) -> int:
    """Estimated bytes for the optimizer's persisted Adam state (cluster-wide under Mode-C)."""
    import torch

    replicated = 0
    local_shard = 0

    def _add_inner(inner_optim: Any, accumulator: str) -> None:
        nonlocal replicated, local_shard
        delta = 0
        for state in getattr(inner_optim, "state", {}).values():
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    delta += int(v.numel()) * int(v.element_size())
        if accumulator == "replicated":
            replicated += delta
        else:
            local_shard += delta

    # v3 round-robin partition: GPU adapter holds only 1/W of persistent state on each rank.
    # v4 within-shard fallback: when only the huge-param path is active (small set
    # empty + huge set non-empty), partition is still active — each rank holds
    # its 1/W shard of the huge-param Adam state.
    persistent_world_size = int(getattr(optim, "_persistent_world_size", 1) or 1)
    persistent_params_full = getattr(optim, "_persistent_params_full", None) or []
    persistent_huge_originals = getattr(optim, "_persistent_huge_originals", None) or []
    partition_active = persistent_world_size > 1 and (
        len(persistent_params_full) > 0 or len(persistent_huge_originals) > 0
    )

    gpu_optim = getattr(optim, "_gpu_optim", None)
    if gpu_optim is not None:
        inner = getattr(gpu_optim, "_optim", None)
        if inner is not None:
            _add_inner(inner, "local_shard" if partition_active else "replicated")

    cpu_optim = getattr(optim, "_cpu_optim", None)
    if cpu_optim is not None:
        for inner in getattr(cpu_optim, "_optims", {}).values():
            _add_inner(inner, "local_shard")

    global_sharded_bytes = local_shard
    try:
        import torch.distributed as _dist

        if _dist.is_available() and _dist.is_initialized():
            shard_tensor = _dist_backend_tensor(local_shard, dtype=torch.long)
            _dist.all_reduce(shard_tensor, op=_dist.ReduceOp.SUM)
            global_sharded_bytes = int(shard_tensor.item())
    except ImportError:
        pass

    return replicated + global_sharded_bytes


def _build_regions_per_chunk(chunk_manager: Any) -> dict[str, list[dict[str, Any]]]:
    """Capture the per-chunk dtype-region layout from ``_chunk_shards``."""
    out: dict[str, list[dict[str, Any]]] = {}
    chunk_shards = getattr(chunk_manager, "_chunk_shards", None) or {}
    for cid, shard_state in chunk_shards.items():
        regions: list[dict[str, Any]] = []
        for region in shard_state.regions:
            regions.append(
                {
                    "chunk_offset": int(region.chunk_offset),
                    "region_bytes": int(region.region_bytes),
                    "region_bytes_padded": int(region.region_bytes_padded),
                    "shard_bytes": int(region.shard_bytes),
                    "dtype": str(region.dtype),
                }
            )
        out[str(int(cid))] = regions
    return out


def _validate_regions_match(
    saved: dict[str, list[dict[str, Any]]],
    current: dict[str, list[dict[str, Any]]],
) -> None:
    """Raise RuntimeError if Mode-C region layouts differ."""
    saved_ids = set(saved.keys())
    current_ids = set(current.keys())
    if saved_ids != current_ids:
        missing = sorted(current_ids - saved_ids, key=lambda s: int(s))
        extra = sorted(saved_ids - current_ids, key=lambda s: int(s))
        raise RuntimeError(
            "ProTrain optimizer load: regions_per_chunk chunk-id mismatch — "
            f"missing on disk: {missing}, extra on disk: {extra}. "
            "The non-persistent chunk partition differs between save and load."
        )

    for cid in sorted(saved_ids, key=lambda s: int(s)):
        saved_regions = saved[cid]
        current_regions = current[cid]
        if len(saved_regions) != len(current_regions):
            raise RuntimeError(
                "ProTrain optimizer load: regions_per_chunk region count "
                f"mismatch on chunk {cid} — saved={len(saved_regions)}, "
                f"current={len(current_regions)}. Likely a dtype-mix change "
                "(e.g. an fp32 layernorm appearing/disappearing in a chunk)."
            )
        for idx, (s, c) in enumerate(zip(saved_regions, current_regions, strict=True)):
            for field in (
                "chunk_offset",
                "region_bytes",
                "region_bytes_padded",
                "shard_bytes",
                "dtype",
            ):
                sv = s.get(field)
                cv = c.get(field)
                # ``dtype`` is compared as string; numeric fields are
                # compared as ints. Any mismatch is fatal.
                if field != "dtype":
                    sv = int(sv) if sv is not None else sv
                    cv = int(cv) if cv is not None else cv
                if sv != cv:
                    raise RuntimeError(
                        "ProTrain optimizer load: regions_per_chunk field "
                        f"mismatch on chunk {cid} region {idx} field "
                        f"{field!r} — saved={sv!r} current={cv!r}. The "
                        "saved per-rank shard tensors will not fit the "
                        "rebuilt shard_param; refusing to load."
                    )


def _huge_param_shards_metadata(optim: Any, world_size: int) -> list[dict[str, Any]]:
    """Return per-original metadata for the within-param shard fallback.

    Empty list iff the optimizer has no huge-param within-shard partition
    active (which is the common case). One entry per huge original
    persistent param; the ``shard_shape`` is the rank-local view shape,
    which is identical across ranks because dim-0 divides world_size by
    construction.
    """
    originals = getattr(optim, "_persistent_huge_originals", None) or []
    shards = getattr(optim, "_persistent_huge_shards", None) or []
    if not originals:
        return []
    out: list[dict[str, Any]] = []
    for orig, shard in zip(originals, shards, strict=True):
        out.append(
            {
                "param_shape": list(int(d) for d in orig.shape),
                "shard_shape": list(int(d) for d in shard.shape),
                "shard_dtype": str(shard.dtype),
                "world_size": int(world_size),
            }
        )
    return out


def _hyperparam_snapshot(optim: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group in optim.param_groups:
        out.append(
            {
                k: v
                for k, v in group.items()
                if k in ("lr", "betas", "eps", "weight_decay")
            }
        )
    return out


def _normalize_hp(hp: dict[str, Any]) -> dict[str, Any]:
    """Normalize hyperparameter dict for save/load drift comparison."""
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in hp.items()}


def _is_raw_protrain_optimizer(optim: Any) -> bool:
    """Duck-type for the raw _ProTrainOptimizer (avoids a circular import)."""
    return (
        hasattr(optim, "_gpu_optim")
        and hasattr(optim, "_cpu_optim")
        and hasattr(optim, "_chunk_manager")
    )


def _unwrap_protrain_optim(optim: Any) -> Any:
    """Return the raw _ProTrainOptimizer or None (unwraps AcceleratedOptimizer)."""
    if optim is None:
        return None
    if _is_raw_protrain_optimizer(optim):
        return optim
    inner = getattr(optim, "optimizer", None)
    if inner is not None and _is_raw_protrain_optimizer(inner):
        return inner
    return None


def _is_protrain_optimizer(optim: Any) -> bool:
    """Truthy iff ``optim`` is (or wraps) a _ProTrainOptimizer."""
    return _unwrap_protrain_optim(optim) is not None


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def _hash_state_dict(sd: dict) -> bytes:
    """Recursively hash a state_dict-like nested structure deterministically."""
    h = hashlib.sha256()

    def _emit(obj: Any) -> None:
        if isinstance(obj, dict):
            h.update(b"dict:")
            for k in sorted(obj, key=repr):
                h.update(repr(k).encode("utf-8"))
                h.update(b"=")
                _emit(obj[k])
                h.update(b";")
        elif isinstance(obj, (list, tuple)):
            h.update(b"seq:")
            for item in obj:
                _emit(item)
                h.update(b",")
        elif isinstance(obj, torch.Tensor):
            t = obj.detach().contiguous().cpu()
            h.update(b"t:")
            h.update(str(t.dtype).encode("utf-8"))
            h.update(b":")
            h.update(repr(tuple(t.shape)).encode("utf-8"))
            h.update(b":")
            # uint8 view handles bf16/etc; flatten first because 0-dim view rejects mismatched esize.
            if t.numel() > 0:
                h.update(t.flatten().view(torch.uint8).numpy().tobytes())
        else:
            # Scalar: int, float, bool, str, None, etc.
            h.update(repr(obj).encode("utf-8"))

    _emit(sd)
    return h.digest()


def _hash_inner_state_dicts(optim: Any) -> str:
    """SHA-256 over the rank's inner optimizer state dicts."""
    h = hashlib.sha256()
    if optim._gpu_optim is not None:
        h.update(b"gpu:")
        h.update(_hash_state_dict(optim._gpu_optim._optim.state_dict()))
    if optim._cpu_optim is not None:
        for cid in sorted(optim._cpu_optim._optims):
            h.update(f"cpu:{int(cid)}:".encode("utf-8"))
            h.update(_hash_state_dict(optim._cpu_optim._optims[cid].state_dict()))
    return h.hexdigest()


def _verify_replicated_state_across_ranks(optim: Any, *, world_size: int) -> None:
    """Cross-rank state-equality check for Mode-B (opt-in, single shot)."""
    if world_size <= 1 or not _dist_is_active():
        return
    local_hash = ""
    local_exc: BaseException | None = None
    try:
        local_hash = _hash_inner_state_dicts(optim)
    except BaseException as exc:  # noqa: BLE001 - re-raised after collective
        local_exc = exc
    # Surface failures BEFORE all_gather_object so a rank-0 exception can't wedge peers.
    _allreduce_status_or_raise(
        1 if local_exc is not None else 0,
        op="verify-replicated-state (local hash)",
    )
    if local_exc is not None:
        # Re-raise the original exception so the actionable traceback is preserved.
        raise local_exc
    gathered: list[str] = [""] * world_size
    torch.distributed.all_gather_object(gathered, local_hash)
    rank0 = gathered[0]
    diverged = [(r, h) for r, h in enumerate(gathered) if h != rank0]
    if diverged:
        raise RuntimeError(
            "ProTrain Mode-B precondition violated: optimizer state "
            "diverges across ranks (rank-0's state does not represent "
            f"the cluster). rank-0 hash={rank0!r}, divergent ranks: {diverged!r}"
        )


def _save_protrain_optim_dir(
    optim: Any,
    output_dir: str,
    *,
    step: int,
    save_max_bytes: int,
    rank: int = 0,
    world_size: int | None = None,
    _skip_size_gate: bool = False,
) -> bool:
    """Write the protrain_optim/ subdirectory. Returns True iff written."""
    chunk_manager = optim._chunk_manager
    if world_size is None:
        world_size = _current_world_size()
    zero3_shard = bool(getattr(chunk_manager, "zero3_shard", False))

    # Drain async CPU Adam so the size-gate sees post-step state.
    chunk_manager.wait_cpu_optim_all()

    estimate = _estimate_optim_state_bytes(optim)
    # Callback already broadcast a rank-0 gate; skip the redundant per-rank check on that path.
    if not _skip_size_gate and estimate > save_max_bytes:
        LOG.warning(
            "ProTrain optimizer save: estimated %d bytes (~%.2f GiB) exceeds "
            "protrain_optim_save_max_bytes=%d (~%.2f GiB) — skipping save. "
            "Raise protrain_optim_save_max_bytes to opt in to larger saves.",
            estimate,
            estimate / 1024**3,
            save_max_bytes,
            save_max_bytes / 1024**3,
        )
        return False

    target = os.path.join(output_dir, PROTRAIN_OPTIM_DIRNAME)

    # v3 round-robin partition: active iff multi-rank AND persistent set non-empty on this build.
    persistent_partition_active = bool(
        int(getattr(optim, "_persistent_world_size", 1) or 1) > 1
        and len(getattr(optim, "_persistent_params_full", None) or []) > 0
    )

    if zero3_shard:
        # Mode-C sharded save. Rank-0 metadata; per-rank gpu+cpu shards when partition active.
        rank0_status = 0
        try:
            if rank == 0:
                # Reset dir so stale files don't trip the load-side hard mismatch.
                shutil.rmtree(target, ignore_errors=True)
                os.makedirs(target, exist_ok=False)

                _fp = _build_layout_fingerprint(chunk_manager, world_size, zero3_shard)
                metadata = {
                    "format_version": SCHEMA_FORMAT_VERSION,
                    "protrain_layout_signature": _layout_signature_from_fingerprint(
                        _fp
                    ),
                    # Persisted so the offline reshard tool can recompute the signature.
                    "layout_fingerprint": _fp,
                    "protrain_persistent_ids": _effective_persistent_ids(chunk_manager),
                    "protrain_n_buffer": int(getattr(chunk_manager, "n_buffer", 0)),
                    "protrain_world_size": int(world_size),
                    "protrain_zero3_shard": zero3_shard,
                    "protrain_save_mode": SAVE_MODE_SHARDED,
                    "saving_rank": int(rank),
                    "param_groups_meta": _hyperparam_snapshot(optim),
                    "saved_at_step": int(step),
                    "torch_version": str(torch.__version__),
                    "estimated_optim_state_bytes": int(estimate),
                    "regions_per_chunk": _build_regions_per_chunk(chunk_manager),
                }
                if persistent_partition_active:
                    metadata["protrain_persistent_partition_version"] = 1
                    metadata["protrain_persistent_owner_world_size"] = int(world_size)
                huge_meta = _huge_param_shards_metadata(optim, world_size)
                if huge_meta:
                    metadata["protrain_persistent_huge_param_shards"] = huge_meta
                with open(os.path.join(target, METADATA_FILENAME), "w") as f:
                    json.dump(metadata, f, indent=2, sort_keys=True)

                if not persistent_partition_active and optim._gpu_optim is not None:
                    torch.save(
                        optim._gpu_optim._optim.state_dict(),
                        os.path.join(target, GPU_OPTIM_FILENAME),
                    )

                cpu_dir = os.path.join(target, CPU_OPTIM_DIRNAME)
                if optim._cpu_optim is not None and optim._cpu_optim._optims:
                    os.makedirs(cpu_dir, exist_ok=True)
        except Exception:
            rank0_status = 1
            raise
        finally:
            # Broadcast rank-0 status before the barrier so failure doesn't deadlock.
            _broadcast_status_or_raise(
                rank0_status, src=0, op="save (rank-0 metadata/gpu_optim)"
            )

        # Barrier so non-zero ranks see metadata + cpu_optim/ before writing.
        _barrier_or_noop()

        per_rank_status = 0
        try:
            # Each rank writes its own GPU optim shard when partition active.
            if persistent_partition_active and optim._gpu_optim is not None:
                if not os.path.isdir(target):
                    raise RuntimeError(
                        f"ProTrain optimizer save: checkpoint directory "
                        f"{target!r} is not visible on rank {rank}. Mode-C "
                        "saves require a shared filesystem across all ranks."
                    )
                torch.save(
                    optim._gpu_optim._optim.state_dict(),
                    os.path.join(target, f"gpu_optim_rank_{int(rank)}.pt"),
                )

            if optim._cpu_optim is not None and optim._cpu_optim._optims:
                cpu_dir = os.path.join(target, CPU_OPTIM_DIRNAME)
                # Cross-rank visibility check: catch node-local FS pretending to be shared.
                if not os.path.isdir(target):
                    raise RuntimeError(
                        f"ProTrain optimizer save: checkpoint directory "
                        f"{target!r} is not visible on rank {rank}. Mode-C "
                        "saves require a shared filesystem across all ranks."
                    )
                # Defensive mkdir for single-rank test-mode runs.
                os.makedirs(cpu_dir, exist_ok=True)
                for cid, inner in optim._cpu_optim._optims.items():
                    path = os.path.join(
                        cpu_dir, f"chunk_{int(cid)}_rank_{int(rank)}.pt"
                    )
                    torch.save(inner.state_dict(), path)
        except Exception:
            per_rank_status = 1
            raise
        finally:
            _allreduce_status_or_raise(
                per_rank_status, op="save (per-rank shard write)"
            )

        if rank == 0:
            LOG.info(
                "ProTrain optimizer save: wrote %s (estimate=%d bytes, "
                "persistent=%d chunks, cpu_chunks=%d, step=%d, "
                "world_size=%d, save_mode=%s, partition=%s)",
                target,
                estimate,
                len(_effective_persistent_ids(chunk_manager)),
                len(optim._cpu_optim._optims) if optim._cpu_optim is not None else 0,
                step,
                world_size,
                SAVE_MODE_SHARDED,
                "v1-round-robin" if persistent_partition_active else "none",
            )
        return True

    # ---------- Mode-B replicated save (rank-0-only write) ----------
    # Rank-0 writes; broadcast status flag so peers fail in lockstep on rank-0 failure.
    persistent_ids = _effective_persistent_ids(chunk_manager)
    rank0_status = 0
    try:
        if rank == 0:
            # Reset dir so stale files from a partial save can't trip the load-side hard mismatch.
            shutil.rmtree(target, ignore_errors=True)
            os.makedirs(target, exist_ok=False)

            metadata = {
                "format_version": SCHEMA_FORMAT_VERSION,
                "protrain_layout_signature": _layout_signature(
                    chunk_manager, world_size, zero3_shard
                ),
                "protrain_persistent_ids": persistent_ids,
                "protrain_n_buffer": int(getattr(chunk_manager, "n_buffer", 0)),
                "protrain_world_size": int(world_size),
                "protrain_zero3_shard": zero3_shard,
                "protrain_save_mode": SAVE_MODE_REPLICATED,
                "saving_rank": int(rank),
                "param_groups_meta": _hyperparam_snapshot(optim),
                "saved_at_step": int(step),
                "torch_version": str(torch.__version__),
                "estimated_optim_state_bytes": int(estimate),
            }
            if persistent_partition_active:
                metadata["protrain_persistent_partition_version"] = 1
                metadata["protrain_persistent_owner_world_size"] = int(world_size)
            huge_meta = _huge_param_shards_metadata(optim, world_size)
            if huge_meta:
                metadata["protrain_persistent_huge_param_shards"] = huge_meta
            with open(os.path.join(target, METADATA_FILENAME), "w") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)

            if not persistent_partition_active and optim._gpu_optim is not None:
                torch.save(
                    optim._gpu_optim._optim.state_dict(),
                    os.path.join(target, GPU_OPTIM_FILENAME),
                )

            if optim._cpu_optim is not None and optim._cpu_optim._optims:
                cpu_dir = os.path.join(target, CPU_OPTIM_DIRNAME)
                os.makedirs(cpu_dir, exist_ok=True)
                for cid, inner in optim._cpu_optim._optims.items():
                    torch.save(
                        inner.state_dict(),
                        os.path.join(cpu_dir, f"chunk_{int(cid)}.pt"),
                    )
    except Exception:
        rank0_status = 1
        raise
    finally:
        _broadcast_status_or_raise(
            rank0_status, src=0, op="save (replicated rank-0 write)"
        )

    # Under partition every rank writes its own gpu_optim_rank_<R>.pt.
    if persistent_partition_active:
        _barrier_or_noop()
        per_rank_status = 0
        try:
            if optim._gpu_optim is not None:
                if not os.path.isdir(target):
                    raise RuntimeError(
                        f"ProTrain optimizer save: checkpoint directory "
                        f"{target!r} is not visible on rank {rank}. "
                        "Per-rank persistent-partition save requires a "
                        "shared filesystem across all ranks."
                    )
                torch.save(
                    optim._gpu_optim._optim.state_dict(),
                    os.path.join(target, f"gpu_optim_rank_{int(rank)}.pt"),
                )
        except Exception:
            per_rank_status = 1
            raise
        finally:
            _allreduce_status_or_raise(
                per_rank_status, op="save (replicated per-rank gpu_optim)"
            )

    if rank == 0:
        LOG.info(
            "ProTrain optimizer save: wrote %s (estimate=%d bytes, "
            "persistent=%d chunks, cpu_chunks=%d, step=%d, "
            "world_size=%d, save_mode=%s, partition=%s)",
            target,
            estimate,
            len(persistent_ids),
            len(optim._cpu_optim._optims) if optim._cpu_optim is not None else 0,
            step,
            world_size,
            SAVE_MODE_REPLICATED,
            "v1-round-robin" if persistent_partition_active else "none",
        )
    return True


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def _load_persistent_gpu_optim(
    optim: Any,
    target: str,
    *,
    metadata: dict[str, Any],
    current_rank: int,
    current_world: int,
) -> None:
    """Load the persistent GPU optimizer state from disk.

    Three cases:
      * ``protrain_persistent_partition_version`` in metadata: each rank
        loads its own ``gpu_optim_rank_<R>.pt``. world_size match was
        already enforced earlier.
      * Legacy ``gpu_optim.pt`` (v2 or v3 no-partition save): rank-0
        loads + broadcasts to peers when distributed; otherwise rank-0
        loads directly.
      * No file on disk + no GPU adapter currently: legal no-op.
    """
    partition_active = metadata.get("protrain_persistent_partition_version") is not None
    legacy_path = os.path.join(target, GPU_OPTIM_FILENAME)
    rank_path = os.path.join(target, f"gpu_optim_rank_{int(current_rank)}.pt")
    if partition_active:
        if optim._gpu_optim is None:
            return
        if not os.path.isfile(rank_path):
            raise RuntimeError(
                "ProTrain optimizer load: missing per-rank persistent "
                f"file {rank_path!r}. Saved with partition_version=1; "
                "expected one gpu_optim_rank_<R>.pt per rank."
            )
        loaded = torch.load(rank_path, map_location="cpu", weights_only=True)
        optim._gpu_optim._optim.load_state_dict(loaded)
        return
    # Legacy single-file or no GPU adapter present.
    if os.path.isfile(legacy_path):
        if optim._gpu_optim is None:
            raise RuntimeError(
                "ProTrain optimizer load: gpu_optim.pt present on disk but "
                "current optimizer has no persistent (GPU) inner — partition "
                "mismatch slipped past the layout-signature check."
            )
        # Single-rank or rank-0-only on-disk file. Rank-0 loads from
        # disk; non-zero ranks pull via broadcast (avoids per-rank
        # reads on a shared FS + matches the v2-into-v3 migration
        # path where the file came from a single-rank save).
        if current_world > 1 and _dist_is_active():
            payload: list[Any] = [None]
            if current_rank == 0:
                payload[0] = torch.load(
                    legacy_path, map_location="cpu", weights_only=True
                )
            torch.distributed.broadcast_object_list(payload, src=0)
            optim._gpu_optim._optim.load_state_dict(payload[0])
        else:
            loaded = torch.load(legacy_path, map_location="cpu", weights_only=True)
            optim._gpu_optim._optim.load_state_dict(loaded)
    elif optim._gpu_optim is not None:
        raise RuntimeError(
            "ProTrain optimizer load: current optimizer has a persistent "
            "(GPU) inner but no gpu_optim.pt / gpu_optim_rank_*.pt files "
            "are present on disk."
        )


def _perform_online_reshard(
    original_target: str,
    saved_world: int,
    current_world: int,
) -> str:
    """Run the online Mode-C reshard against a sibling temp dir."""
    # Source-of-truth import: the offline CLI also imports from here.
    from axolotl.integrations.protrain.api.reshard import (  # noqa: PLC0415
        reshard_mode_c_shards,
    )

    temp_dir = os.path.join(
        original_target,
        f".reshard_to_N{int(current_world)}",
    )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank_for_reshard = int(torch.distributed.get_rank())
    else:
        rank_for_reshard = 0

    # Lockstep failure protocol: rank-0 broadcasts status so peers fail with it.
    reshard_status = 0
    try:
        if rank_for_reshard == 0:
            LOG.info(
                "ProTrain optimizer load: online reshard "
                "saved_world=%d → current_world=%d (opt-in via "
                "protrain_allow_online_reshard). Writing to %s",
                saved_world,
                current_world,
                temp_dir,
            )
            # Pre-clean stale temp dir from a previous interrupted run.
            shutil.rmtree(temp_dir, ignore_errors=True)
            reshard_mode_c_shards(
                original_target,
                temp_dir,
                int(current_world),
                log_fn=LOG.info,
            )
    except Exception:
        reshard_status = 1
        raise
    finally:
        _broadcast_status_or_raise(
            reshard_status,
            src=0,
            op="load (online reshard)",
        )

    # Barrier so non-zero ranks see the resharded files before they read them.
    _barrier_or_noop()

    return temp_dir


def _load_protrain_optim_dir(
    optim: Any,
    checkpoint_dir: str,
    *,
    allow_online_reshard: bool = False,
) -> bool:
    """Load a previously saved protrain_optim/ subdirectory in-place."""
    original_target = os.path.join(checkpoint_dir, PROTRAIN_OPTIM_DIRNAME)
    target = original_target

    # Cross-rank visibility consensus: catch node-local FS pretending to be shared.
    has_dir = _allreduce_visibility_consensus(
        os.path.isdir(target),
        what="checkpoint directory",
        path=target,
    )
    if not has_dir:
        return False

    meta_path = os.path.join(target, METADATA_FILENAME)
    has_meta = _allreduce_visibility_consensus(
        os.path.isfile(meta_path),
        what="metadata file",
        path=meta_path,
    )
    if not has_meta:
        raise RuntimeError(
            f"ProTrain optimizer load: {target!r} exists but lacks "
            f"{METADATA_FILENAME}. Refusing to load partial checkpoint."
        )
    metadata = _read_metadata_lockstep(meta_path)

    fmt = int(metadata.get("format_version", 0))
    if fmt == 1:
        # v1 predates save_mode / saving_rank; known to be single-rank replicated.
        metadata.setdefault("protrain_save_mode", SAVE_MODE_REPLICATED)
        metadata.setdefault("saving_rank", 0)
        metadata.setdefault("protrain_world_size", 1)
        metadata.setdefault("protrain_zero3_shard", False)
    elif fmt == 2:
        # v2 (Phase 2 pre-partition): no partition metadata; gpu_optim.pt is full state on rank-0.
        if "protrain_save_mode" not in metadata:
            raise RuntimeError(
                "ProTrain optimizer load: v2 metadata missing required "
                "field 'protrain_save_mode'. Refusing to load."
            )
        if "saving_rank" not in metadata:
            raise RuntimeError(
                "ProTrain optimizer load: v2 metadata missing required "
                "field 'saving_rank'. Refusing to load."
            )
    elif fmt in (3, SCHEMA_FORMAT_VERSION):
        if "protrain_save_mode" not in metadata:
            raise RuntimeError(
                f"ProTrain optimizer load: v{fmt} metadata missing required "
                "field 'protrain_save_mode'. Refusing to load."
            )
        if "saving_rank" not in metadata:
            raise RuntimeError(
                f"ProTrain optimizer load: v{fmt} metadata missing required "
                "field 'saving_rank'. Refusing to load."
            )
    else:
        raise RuntimeError(
            f"ProTrain optimizer load: unknown format_version={fmt} "
            f"(this build expects {SCHEMA_FORMAT_VERSION}). Refusing to load."
        )

    # v3 round-robin partition: when present on disk, world_size must match exactly.
    saved_partition_version = metadata.get("protrain_persistent_partition_version")
    if saved_partition_version is not None:
        saved_world_for_partition = int(
            metadata.get(
                "protrain_persistent_owner_world_size",
                metadata.get("protrain_world_size", 0),
            )
        )
        current_world_for_partition = _current_world_size()
        if int(saved_world_for_partition) != int(current_world_for_partition):
            raise RuntimeError(
                "world_size mismatch on resume: persistent fp32 master is "
                "round-robin partitioned and offline reshard does not "
                "support repartitioning. Resume with the original "
                f"world_size of {int(saved_world_for_partition)}."
            )

    # v4 within-param shard fallback: when huge-param shards are recorded,
    # world_size must match identity — offline reshard does not yet support
    # repartitioning the within-shard dim-0 slices.
    saved_huge_shards = metadata.get("protrain_persistent_huge_param_shards")
    if saved_huge_shards:
        # Use the first shard's recorded world_size; all entries share the value by construction.
        saved_world_for_huge = int(saved_huge_shards[0].get("world_size", 0))
        current_world_for_huge = _current_world_size()
        if saved_world_for_huge != current_world_for_huge:
            raise RuntimeError(
                "protrain: cross-world-size resume not supported when "
                "huge-param within-shard partition is active. Resume with "
                f"the original world_size of {saved_world_for_huge}, or "
                "run an offline reshard."
            )

    chunk_manager = optim._chunk_manager
    current_world = _current_world_size()
    current_zero3 = bool(getattr(chunk_manager, "zero3_shard", False))
    saved_world = int(metadata["protrain_world_size"])
    saved_zero3 = bool(metadata["protrain_zero3_shard"])
    saved_mode = str(metadata["protrain_save_mode"])
    current_mode = SAVE_MODE_SHARDED if current_zero3 else SAVE_MODE_REPLICATED

    if saved_mode not in (SAVE_MODE_REPLICATED, SAVE_MODE_SHARDED):
        raise RuntimeError(
            f"ProTrain optimizer load: unknown protrain_save_mode="
            f"{saved_mode!r}. Refusing to load."
        )

    # Save-mode mismatch (§4.2). Hard error in either direction.
    if saved_mode != current_mode:
        raise RuntimeError(
            f"ProTrain optimizer load: save_mode mismatch — "
            f"saved={saved_mode!r} current={current_mode!r}. "
            "Replicated state cannot be loaded into a sharded run, and "
            "sharded state cannot be loaded into a replicated run; the "
            "on-disk shape doesn't match what the current run needs."
        )

    if saved_zero3 != current_zero3:
        raise RuntimeError(
            f"ProTrain optimizer load: zero3_shard mismatch — saved={saved_zero3} "
            f"current={current_zero3}."
        )

    if current_zero3:
        # Mode-C sharded load. Hard-error on world_size mismatch unless online reshard opted in.
        if saved_world != current_world:
            if not allow_online_reshard:
                raise RuntimeError(
                    "ProTrain optimizer load: Mode-C sharded resume "
                    f"requires identical world_size — saved={saved_world} "
                    f"current={current_world}. Two ways to recover:\n"
                    "  (a) Offline reshard via the CLI before resuming:\n"
                    "      ``python -m scripts.protrain.reshard_optim "
                    "--src <saved-protrain_optim-dir> "
                    "--dst <new-protrain_optim-dir> --target-world "
                    f"{current_world}``\n"
                    "  (b) Online reshard on load by setting "
                    "``protrain_allow_online_reshard: True`` in the "
                    "ProTrain config (off by default — opt-in because "
                    "online resharding writes a temp dir under the "
                    "checkpoint and silent automatic resharding can "
                    "mask configuration drift the user might want to "
                    "see). Both paths use the same reshard logic; "
                    "(a) is the conservative default. Alternatively, "
                    "resume with the original world_size or set "
                    "``protrain_save_optimizer_state=False`` to "
                    "discard the saved optimizer state."
                )

            # Rank-0 writes a sibling temp dir; trailing barrier ensures visibility.
            online_reshard_temp_dir = _perform_online_reshard(
                original_target,
                saved_world=saved_world,
                current_world=current_world,
            )

            # Re-point at the resharded dir; saved_world now == current_world by construction.
            target = online_reshard_temp_dir
            metadata = _read_metadata_lockstep(os.path.join(target, METADATA_FILENAME))
            saved_world = int(metadata["protrain_world_size"])
            assert saved_world == current_world, (
                "online reshard produced metadata with "
                f"protrain_world_size={saved_world}, expected "
                f"{current_world} — bug in reshard_mode_c_shards"
            )
        else:
            online_reshard_temp_dir = None

        # Region-layout match: drift means saved bytes won't fit the rebuilt shard_param.
        saved_regions = metadata.get("regions_per_chunk")
        if saved_regions is None:
            raise RuntimeError(
                "ProTrain optimizer load: sharded metadata missing "
                "required field 'regions_per_chunk'. The save predates "
                "Mode-C support or the file is corrupt."
            )
        current_regions = _build_regions_per_chunk(chunk_manager)
        _validate_regions_match(saved_regions, current_regions)

        # Signature comparison uses saved values; saved_world == current_world here.
        saved_sig = metadata["protrain_layout_signature"]
        expected_sig = _layout_signature(chunk_manager, saved_world, saved_zero3)
        if saved_sig != expected_sig:
            raise RuntimeError(
                "ProTrain optimizer load: layout signature mismatch.\n"
                f"  saved   = {saved_sig}\n"
                f"  current = {expected_sig}\n"
                "The model architecture, S_chunk, persistent_ids, "
                "world_size, or zero3_shard differs between save and "
                "load. Resume is unsafe."
            )

        saved_pids = list(metadata["protrain_persistent_ids"])
        current_pids = _effective_persistent_ids(chunk_manager)
        if saved_pids != current_pids:
            raise RuntimeError(
                "ProTrain optimizer load: persistent_ids set mismatch.\n"
                f"  saved   = {saved_pids}\n"
                f"  current = {current_pids}\n"
                "The search picked a different partition. Pin the saved "
                "set via protrain_n_persist_override (and related "
                "overrides) to resume."
            )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            current_rank = int(torch.distributed.get_rank())
        else:
            current_rank = 0

        # Per-rank shard load wrapped in lockstep failure protocol; stray files rejected up-front.
        cpu_dir = os.path.join(target, CPU_OPTIM_DIRNAME)
        expected_cpu_ids = (
            set(int(cid) for cid in optim._cpu_optim._optims)
            if optim._cpu_optim is not None
            else set()
        )
        load_status = 0
        try:
            _load_persistent_gpu_optim(
                optim,
                target,
                metadata=metadata,
                current_rank=current_rank,
                current_world=current_world,
            )

            if os.path.isdir(cpu_dir):
                for name in os.listdir(cpu_dir):
                    m = CHUNK_SHARD_FILE_RE.match(name)
                    if m is None:
                        raise RuntimeError(
                            "ProTrain optimizer load: unexpected file "
                            f"{name!r} in {cpu_dir!r} — Mode-C cpu_optim/ "
                            "must contain only chunk_<N>_rank_<R>.pt "
                            "shards. Refusing to load."
                        )
                    file_chunk_id = int(m.group(1))
                    file_rank = int(m.group(2))
                    if file_rank < 0 or file_rank >= current_world:
                        raise RuntimeError(
                            "ProTrain optimizer load: unexpected file "
                            f"{name!r} in {cpu_dir!r} — rank ordinal "
                            f"{file_rank} is outside the current "
                            f"world_size range [0, {current_world}). "
                            "Likely a leftover shard from a higher-"
                            "world_size save. Refusing to load."
                        )
                    if file_chunk_id not in expected_cpu_ids:
                        raise RuntimeError(
                            "ProTrain optimizer load: unexpected file "
                            f"{name!r} in {cpu_dir!r} — chunk ID "
                            f"{file_chunk_id} is not in the current set "
                            f"of non-persistent chunk IDs "
                            f"{sorted(expected_cpu_ids)}. Likely a "
                            "leftover shard from a different partition "
                            "or persistent_ids configuration. Refusing "
                            "to load."
                        )
            if optim._cpu_optim is not None and optim._cpu_optim._optims:
                for cid, inner in optim._cpu_optim._optims.items():
                    shard_path = os.path.join(
                        cpu_dir, f"chunk_{int(cid)}_rank_{current_rank}.pt"
                    )
                    if not os.path.isfile(shard_path):
                        raise RuntimeError(
                            "ProTrain optimizer load: missing rank shard "
                            f"{shard_path!r}. Expected per-rank file for "
                            f"rank {current_rank} chunk {int(cid)} — the "
                            "saved checkpoint is incomplete or was produced "
                            "by a different world_size."
                        )
                    loaded = torch.load(
                        shard_path, map_location="cpu", weights_only=True
                    )
                    inner.load_state_dict(loaded)
                    # Force CPU: torch load_state_dict auto-casts to param.device but DeepSpeedCPUAdam needs CPU.
                    for state in inner.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                                state[k] = v.cpu()
        except Exception:
            load_status = 1
            raise
        finally:
            _allreduce_status_or_raise(load_status, op="load (per-rank shard read)")

        # Hyperparam drift: warn but accept; zip strict=False mirrors the documented contract.
        saved_hp = metadata.get("param_groups_meta", [])
        current_hp = _hyperparam_snapshot(optim)
        if len(saved_hp) != len(current_hp):
            LOG.warning(
                "ProTrain optimizer load: param-group count mismatch "
                "(saved=%d, current=%d) — accepting partial restore; "
                "groups beyond min(saved, current) won't be compared.",
                len(saved_hp),
                len(current_hp),
            )
        for i, (s, c) in enumerate(zip(saved_hp, current_hp, strict=False)):
            if _normalize_hp(s) != _normalize_hp(c):
                LOG.warning(
                    "ProTrain optimizer load: param_groups[%d] "
                    "hyperparams drifted between save and load — "
                    "saved=%s current=%s. Continuing.",
                    i,
                    s,
                    c,
                )

        LOG.info(
            "ProTrain optimizer load: restored from %s (saved_at_step=%d, "
            "persistent=%d chunks, cpu_chunks=%d, save_mode=%s, rank=%d)",
            target,
            int(metadata.get("saved_at_step", -1)),
            len(saved_pids),
            len(optim._cpu_optim._optims) if optim._cpu_optim is not None else 0,
            SAVE_MODE_SHARDED,
            current_rank,
        )

        # Cleanup online-reshard temp dir; failure path leaves it for post-mortem.
        if online_reshard_temp_dir is not None:
            _barrier_or_noop()
            if current_rank == 0 and os.path.isdir(online_reshard_temp_dir):
                try:
                    shutil.rmtree(online_reshard_temp_dir)
                except OSError as cleanup_exc:
                    # Cleanup failure is non-fatal — load already succeeded.
                    LOG.warning(
                        "ProTrain optimizer load: failed to clean up "
                        "online reshard temp dir %s: %s",
                        online_reshard_temp_dir,
                        cleanup_exc,
                    )
        return True

    # Mode-B replicated load; world_size differences tolerated (state is rank-independent).
    if saved_world != current_world:
        LOG.info(
            "ProTrain optimizer load: replicated checkpoint saved with "
            "world_size=%d loading into world_size=%d. Replicated state "
            "is rank-independent, so this is supported.",
            saved_world,
            current_world,
        )

    # Recompute signature at current world_size; only chunk geometry + persistent_ids + zero3 matter here.
    saved_sig = metadata["protrain_layout_signature"]
    expected_sig = _layout_signature(chunk_manager, current_world, saved_zero3)
    if saved_sig != expected_sig:
        raise RuntimeError(
            "ProTrain optimizer load: layout signature mismatch.\n"
            f"  saved   = {saved_sig}\n"
            f"  current = {expected_sig}\n"
            "The model architecture, S_chunk, persistent_ids, world_size, or "
            "zero3_shard differs between save and load. Resume is unsafe."
        )

    saved_pids = list(metadata["protrain_persistent_ids"])
    current_pids = _effective_persistent_ids(chunk_manager)
    if saved_pids != current_pids:
        raise RuntimeError(
            "ProTrain optimizer load: persistent_ids set mismatch.\n"
            f"  saved   = {saved_pids}\n"
            f"  current = {current_pids}\n"
            "The search picked a different partition. Pin the saved set via "
            "protrain_n_persist_override (and related overrides) to resume."
        )

    # Lockstep failure protocol — local torch.load failure must not wedge peers at the barrier.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        current_rank_mb = int(torch.distributed.get_rank())
    else:
        current_rank_mb = 0
    load_status = 0
    captured_exc: Exception | None = None
    try:
        _load_persistent_gpu_optim(
            optim,
            target,
            metadata=metadata,
            current_rank=current_rank_mb,
            current_world=current_world,
        )

        # CPU optim: walk saved chunk files; require an exact match against the
        # current set of non-persistent chunk IDs.
        cpu_dir = os.path.join(target, CPU_OPTIM_DIRNAME)
        saved_chunks: dict[int, str] = {}
        if os.path.isdir(cpu_dir):
            for name in os.listdir(cpu_dir):
                m = CHUNK_FILE_RE.match(name)
                if m is None:
                    raise RuntimeError(
                        f"ProTrain optimizer load: unexpected file {name!r} in "
                        f"{cpu_dir!r} — refusing to load."
                    )
                saved_chunks[int(m.group(1))] = os.path.join(cpu_dir, name)

        current_cpu_ids = (
            set(int(cid) for cid in optim._cpu_optim._optims)
            if optim._cpu_optim is not None
            else set()
        )
        saved_cpu_ids = set(saved_chunks)
        if saved_cpu_ids != current_cpu_ids:
            missing_on_disk = current_cpu_ids - saved_cpu_ids
            extra_on_disk = saved_cpu_ids - current_cpu_ids
            raise RuntimeError(
                "ProTrain optimizer load: CPU chunk set mismatch — "
                f"missing on disk: {sorted(missing_on_disk)}, "
                f"extra on disk: {sorted(extra_on_disk)}."
            )

        if optim._cpu_optim is not None:
            for cid, inner in optim._cpu_optim._optims.items():
                loaded = torch.load(
                    saved_chunks[int(cid)], map_location="cpu", weights_only=True
                )
                inner.load_state_dict(loaded)
                # Force CPU: torch auto-casts to param.device but DeepSpeedCPUAdam needs CPU pointer.
                for state in inner.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                            state[k] = v.cpu()
    except Exception as exc:
        load_status = 1
        captured_exc = exc
    try:
        _allreduce_status_or_raise(load_status, op="load (replicated read)")
    except Exception:
        # Prefer the caller's original exception over the helper's synthesised one when present.
        if captured_exc is None:
            raise
    if captured_exc is not None:
        raise captured_exc

    # Cross-rank state-equality check: catches divergent restores on non-shared FS.
    # Skip under v3 round-robin partition: per-rank GPU state is intentionally different.
    if metadata.get("protrain_persistent_partition_version") is None:
        _verify_replicated_state_across_ranks(optim, world_size=current_world)

    # Hyperparam drift: warn but accept; normalize for JSON betas tuple→list round-trip.
    saved_hp = metadata.get("param_groups_meta", [])
    current_hp = _hyperparam_snapshot(optim)
    if len(saved_hp) != len(current_hp):
        LOG.warning(
            "ProTrain optimizer load: param-group count mismatch "
            "(saved=%d, current=%d) — accepting partial restore; "
            "groups beyond min(saved, current) won't be compared.",
            len(saved_hp),
            len(current_hp),
        )
    for i, (s, c) in enumerate(zip(saved_hp, current_hp, strict=False)):
        if _normalize_hp(s) != _normalize_hp(c):
            LOG.warning(
                "ProTrain optimizer load: param_groups[%d] hyperparams drifted "
                "between save and load — saved=%s current=%s. Continuing.",
                i,
                s,
                c,
            )

    LOG.info(
        "ProTrain optimizer load: restored from %s (saved_at_step=%d, "
        "persistent=%d chunks, cpu_chunks=%d)",
        target,
        int(metadata.get("saved_at_step", -1)),
        len(saved_pids),
        len(saved_chunks),
    )
    return True


# ---------------------------------------------------------------------------
# Public callback (save side)
# ---------------------------------------------------------------------------


def _make_callback_class():
    """Lazy-imported callback class — keeps transformers out of import path."""
    from transformers.trainer_callback import TrainerCallback

    class ProTrainOptimizerCheckpointCallback(TrainerCallback):
        """``on_save``: write protrain_optim/ beside HF's checkpoint dir."""

        def __init__(
            self,
            *,
            save_max_bytes: int,
            verify_replicated: bool = False,
        ) -> None:
            """Store save policy and one-shot replication-verify flag."""
            self._save_max_bytes = save_max_bytes
            self._verify_replicated = bool(verify_replicated)
            # Verify fires on the first save only; per-save would be expensive.
            self._verify_replicated_done = False

        def on_save(
            self,
            args: "TrainingArguments",
            state: "TrainerState",
            control: "TrainerControl",
            **kwargs: Any,
        ) -> "TrainerControl":
            """Persist the ProTrain optimizer state alongside the HF checkpoint dir."""
            # Trainer.optimizer is wrapped by AcceleratedOptimizer after prepare; unwrap first.
            raw = _unwrap_protrain_optim(kwargs.get("optimizer"))
            if raw is None:
                return control

            rank = int(getattr(args, "process_index", 0))
            world_size = int(getattr(args, "world_size", 1))
            chunk_manager = raw._chunk_manager
            zero3_shard = bool(getattr(chunk_manager, "zero3_shard", False))

            checkpoint_dir = os.path.join(
                args.output_dir, f"checkpoint-{state.global_step}"
            )
            # Only rank-0 sees the HF dir; non-zero ranks must still drain + participate in collectives.
            checkpoint_dir_missing = rank == 0 and not os.path.isdir(checkpoint_dir)

            # Pre-save preamble under lockstep protocol; commit state only after status check.
            preamble_status = 0
            skip = False
            verify_fired = False
            estimate = 0
            try:
                # Drain CPU adam on every rank.
                chunk_manager.wait_cpu_optim_all()

                # Estimate runs on every rank for the cluster-wide all_reduce; gate decision rank-0-only.
                estimate = _estimate_optim_state_bytes(raw)
                if rank == 0:
                    if checkpoint_dir_missing:
                        # Missing-dir takes precedence over the size estimate.
                        skip = True
                        LOG.warning(
                            "ProTrainOptimizerCheckpointCallback.on_save: "
                            "expected checkpoint dir %s does not exist on "
                            "rank-0; skipping ProTrain shard.",
                            checkpoint_dir,
                        )
                    else:
                        skip = estimate > self._save_max_bytes
                    if skip and not checkpoint_dir_missing:
                        LOG.warning(
                            "ProTrain optimizer save: estimated %d bytes "
                            "(~%.2f GiB) exceeds protrain_optim_save_max_bytes="
                            "%d (~%.2f GiB) — skipping save (decision "
                            "broadcast to %d ranks).",
                            estimate,
                            estimate / 1024**3,
                            self._save_max_bytes,
                            self._save_max_bytes / 1024**3,
                            world_size,
                        )

                # Cross-rank verify (opt-in, once per run, Mode-B only).
                if (
                    self._verify_replicated
                    and not self._verify_replicated_done
                    and world_size > 1
                    and not zero3_shard
                ):
                    _verify_replicated_state_across_ranks(raw, world_size=world_size)
                    verify_fired = True
            except Exception:
                preamble_status = 1
                raise
            finally:
                _allreduce_status_or_raise(
                    preamble_status, op="save (pre-save preamble)"
                )

            # Commit verify state only after the status check confirmed every rank succeeded.
            if verify_fired:
                self._verify_replicated_done = True

            # Broadcast rank-0's gate decision to every rank.
            skip_decision = [skip]
            _broadcast_object_list_or_noop(skip_decision, src=0)
            if skip_decision[0]:
                _barrier_or_noop()
                return control

            # Write per-mode (dispatcher inside _save_protrain_optim_dir handles B vs C).
            _save_protrain_optim_dir(
                raw,
                checkpoint_dir,
                step=int(state.global_step),
                save_max_bytes=self._save_max_bytes,
                rank=rank,
                world_size=world_size,
                # Inner per-rank gate must not re-trip — callback already broadcast.
                _skip_size_gate=True,
            )

            # Barrier so downstream code sees the dir.
            _barrier_or_noop()
            return control

    return ProTrainOptimizerCheckpointCallback


def make_checkpoint_callback(
    *,
    save_max_bytes: int,
    verify_replicated: bool = False,
) -> "TrainerCallback":
    """Return a fresh ProTrain optimizer-checkpoint TrainerCallback instance."""
    cls = _make_callback_class()
    return cls(
        save_max_bytes=save_max_bytes,
        verify_replicated=verify_replicated,
    )


# ---------------------------------------------------------------------------
# Load monkey-patch
# ---------------------------------------------------------------------------


def install_load_hook(
    trainer: Any, optim: Any, *, allow_online_reshard: bool = False
) -> None:
    """Wrap ``trainer._load_optimizer_and_scheduler`` to also load ProTrain."""
    raw_at_install = _unwrap_protrain_optim(optim)
    if raw_at_install is None:
        # Silent no-op — caller passed a non-ProTrain optimizer.
        return

    original = trainer._load_optimizer_and_scheduler

    def _patched(checkpoint: str | None) -> None:
        # Re-resolve from trainer.optimizer so cross-mode rebuilds load into the live instance.
        raw = _unwrap_protrain_optim(getattr(trainer, "optimizer", None))
        if raw is None:
            raw = raw_at_install
        # Wrap original load so a one-rank failure can't wedge peers at the trailing barrier.
        original_exc_info: Any = None
        hf_load_status = 0
        peer_hf_failure: Exception | None = None
        try:
            original(checkpoint)
        except Exception:
            hf_load_status = 1
            original_exc_info = sys.exc_info()

        # Synchronize HF-load result before any rank enters ProTrain's collectives.
        try:
            _allreduce_status_or_raise(
                hf_load_status, op="load (HF optimizer/scheduler)"
            )
        except Exception as exc:
            # Surviving ranks: peer failed; skip ProTrain load and hit the same barrier.
            if original_exc_info is None:
                peer_hf_failure = exc

        if (
            original_exc_info is None
            and peer_hf_failure is None
            and checkpoint is not None
        ):
            try:
                _load_protrain_optim_dir(
                    raw,
                    checkpoint,
                    allow_online_reshard=allow_online_reshard,
                )
            except Exception:
                LOG.exception(
                    "ProTrain optimizer load failed from %s — re-raising. "
                    "If you intended to discard the saved state, set "
                    "protrain_save_optimizer_state=False and remove the "
                    "protrain_optim/ subdirectory from the checkpoint.",
                    checkpoint,
                )
                # Barrier before re-raising so a one-rank failure doesn't wedge the cluster.
                _barrier_or_noop()
                raise
        # Defensive barrier to keep the cluster in lockstep through resume.
        _barrier_or_noop()
        if original_exc_info is not None:
            raise original_exc_info[1].with_traceback(original_exc_info[2])
        if peer_hf_failure is not None:
            # Surviving rank: peer's HF load failed; raise after barrier.
            raise peer_hf_failure

    trainer._load_optimizer_and_scheduler = _patched  # type: ignore[method-assign]


__all__ = [
    "CHUNK_SHARD_FILE_RE",
    "DEFAULT_SAVE_MAX_BYTES",
    "GPU_OPTIM_RANK_FILE_RE",
    "PROTRAIN_OPTIM_DIRNAME",
    "SAVE_MODE_REPLICATED",
    "SAVE_MODE_SHARDED",
    "SCHEMA_FORMAT_VERSION",
    "_barrier_or_noop",
    "_broadcast_object_list_or_noop",
    "_build_regions_per_chunk",
    "_effective_persistent_ids",
    "_estimate_optim_state_bytes",
    "_hash_inner_state_dicts",
    "_hash_state_dict",
    "_is_protrain_optimizer",
    "_is_raw_protrain_optimizer",
    "_layout_signature",
    "_load_protrain_optim_dir",
    "_save_protrain_optim_dir",
    "_unwrap_protrain_optim",
    "_validate_regions_match",
    "_verify_replicated_state_across_ranks",
    "install_load_hook",
    "make_checkpoint_callback",
]
