"""Optimizer-state checkpoint/resume for the ProTrain runtime.

Implements Phase 1 (CHECKPOINT_DESIGN.md) and Phase 2 Modes B and C
(CHECKPOINT_DESIGN_PHASE2.md). Save runs through
``ProTrainOptimizerCheckpointCallback.on_save`` after HF writes its
standard checkpoint files; load runs through a monkey-patched
``trainer._load_optimizer_and_scheduler`` (HF has no
``on_load_checkpoint`` callback, and ``on_train_begin`` fires after
the load slot, so the patch is the only correct hook).

On disk under ``{checkpoint_dir}/protrain_optim/``:

* ``metadata.json``                 — schema version, layout
                                      signature, effective
                                      persistent_ids set, world_size,
                                      zero3_shard, save_mode,
                                      saving_rank, hyperparam snapshot,
                                      step. Mode-C also stores
                                      ``regions_per_chunk`` describing
                                      every per-chunk dtype-region.
* ``gpu_optim.pt``                  — ``torch.save`` of the persistent
                                      inner optimizer's ``state_dict``
                                      (absent if no chunks are
                                      persistent). Replicated across
                                      ranks in both modes; rank-0 only
                                      writes.
* ``cpu_optim/chunk_<N>.pt``        — Mode-B replicated: one file per
                                      non-persistent chunk; rank-0
                                      writes. Bounds peak save-time
                                      RAM to one chunk's worth of
                                      state.
* ``cpu_optim/chunk_<N>_rank_<R>.pt``
                                    — Mode-C sharded: each rank writes
                                      its own per-rank-per-chunk file
                                      (per-rank state is genuinely
                                      different under ZeRO-3 sharding).

Mode-B (DDP-replicated) writes only on rank-0 — every rank has the
same state by DDP's grad-allreduce contract. Mode-C (ZeRO-3 sharded)
writes the persistent state and metadata on rank-0 (replicated
across ranks) and the per-rank chunk shards on every rank. Per-rank
filenames distinguish Mode-C shards from Mode-B's no-suffix files so
the two modes don't collide on disk.

Hard validation on load: zero3_shard, layout signature, save_mode,
and effective persistent_ids set must all match the current run. World
size is allowed to differ between save and load in Mode-B (replicated
state is shape-independent of world_size); Mode-C requires identical
world_size since the shard arithmetic depends on it (cross-world-size
resume needs a re-shard step that's out of scope for Phase 2). Mode-C
additionally requires the saved per-chunk dtype-region descriptors to
exactly match the current run's region layout — a mismatch implies
the saved bytes won't fit the rebuilt ``shard_param`` and we'd crash
deep in ``load_state_dict`` otherwise. All ``torch.load`` calls pin
``map_location='cpu'`` to defeat HF Trainer's hostile
``map_location=device`` default for CPU-offloaded adam state.
"""

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
SCHEMA_FORMAT_VERSION = 2
SAVE_MODE_REPLICATED = "replicated"
SAVE_MODE_SHARDED = "sharded"
DEFAULT_SAVE_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB; mirrors args.py default

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
    """Broadcast a list of picklable objects from ``src`` to every rank.

    No-op when ``torch.distributed`` is not initialized — preserves
    Phase 1 single-rank behavior. ``obj_list`` is mutated in place to
    match ``src``'s contents.
    """
    if not _dist_is_active():
        return
    torch.distributed.broadcast_object_list(obj_list, src=src)


def _barrier_or_noop() -> None:
    """``dist.barrier()`` if dist is active; else no-op."""
    if not _dist_is_active():
        return
    torch.distributed.barrier()


def _dist_status_tensor(status: int) -> torch.Tensor:
    """Build a 0/1 status tensor on the right device for the active backend.

    NCCL collectives reject CPU tensors, so when the process group is up
    and using NCCL we must place the flag on the current CUDA device.
    For Gloo / MPI / single-rank fall-back, CPU is correct.
    """
    device = torch.device("cpu")
    if _dist_is_active() and torch.distributed.get_backend() == "nccl":
        device = torch.device("cuda", torch.cuda.current_device())
    return torch.tensor([int(status)], dtype=torch.int64, device=device)


def _broadcast_status_or_raise(status: int, *, src: int, op: str) -> None:
    """Broadcast a 0/1 status flag from ``src`` and raise on every rank if non-zero.

    Used to guard barriers around single-rank-writes-only sections (Mode-C
    save: rank-0 writes ``metadata.json`` + ``gpu_optim.pt``). If ``src``
    raised mid-write, it must still call this with ``status=1`` from a
    ``finally`` block so the broadcast happens before the source rank
    re-raises its original exception. Non-source ranks receive the flag
    and synthesize a ``RuntimeError`` so the cluster fails in lockstep
    instead of deadlocking on the trailing barrier.

    No-op when dist is not initialised: in single-rank runs the local
    exception is already propagating from the caller's ``finally``-
    bracketed ``except: raise``, so synthesizing a generic RuntimeError
    here would only stomp the actionable underlying traceback.
    """
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
    """All-reduce SUM a status flag across the cluster; raise everywhere if any rank failed.

    Used to guard barriers around per-rank-writes/reads (Mode-C save's
    per-rank shard writes; Mode-C/B load's per-rank shard reads). Each
    rank contributes its local 0/1 status; if the sum is non-zero, every
    rank raises so the cluster fails in lockstep instead of deadlocking
    on the trailing barrier.

    No-op when dist is not initialised: in single-rank runs the local
    exception is already propagating from the caller's ``except: raise``,
    so synthesizing a generic RuntimeError here would only stomp the
    actionable underlying traceback.
    """
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
    """Reach cross-rank consensus on whether a path is visible.

    All-reduces a per-rank 0/1 ``present`` flag across the cluster and
    classifies the result into one of three states:

    * ``total == 0`` (every rank reports absent) → returns ``False``;
      caller treats the load as a no-op (e.g. first run, opt-out).
    * ``total == world_size`` (every rank reports present) → returns
      ``True``; caller proceeds with the read.
    * mixed (``0 < total < world_size``) → raises ``RuntimeError`` on
      every rank so the cluster fails in lockstep instead of letting one
      rank silently skip the ProTrain shard while others restore it (or
      vice versa). This is the load-side analogue of the Mode-C save
      path's per-rank ``os.path.isdir(target)`` visibility check.

    No-op when dist is not initialised: returns ``present`` as-is so
    single-rank runs preserve their original semantics.

    ``what``/``path`` are folded into the mixed-visibility error message
    to point the user at which file failed the cross-rank check.
    """
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
    """Read + parse ``metadata.json`` with the same all-reduced status protocol used for shard I/O.

    The metadata read sits between visibility consensus and the trailing
    collectives in the load hook (``_perform_online_reshard`` and the
    per-rank shard read). A rank-local read or parse failure here would
    otherwise let the failing rank unwind to the outer barrier in
    ``install_load_hook`` while surviving ranks march into those
    collectives and wedge the job. Mirror the per-rank-shard-read sync:
    every rank contributes a 0/1 status, the cluster all-reduces, and
    any non-zero total raises everywhere — local failures still surface
    their original exception (``_allreduce_status_or_raise`` returns
    without raising for them), so tracebacks aren't stomped.
    """
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
        # Another rank failed; this rank is the synthesized-error rank.
        # Local failures fall through to the captured re-raise below so
        # the original traceback wins.
        if captured_exc is None:
            raise
    if captured_exc is not None:
        raise captured_exc
    assert metadata is not None
    # Cross-rank fingerprint: every rank may have read a metadata.json at
    # the same path with *different contents* — e.g. when ``output_dir``
    # is a per-node local path rather than a shared tree. The status
    # all-reduce above only catches read/parse failures; byte-equal
    # success on divergent contents would otherwise leave the
    # compatibility checks running against rank-local metadata
    # (split-brain). Canonicalize the JSON so dict insertion order can't
    # cause spurious mismatches, all_gather, and raise everywhere if any
    # rank disagrees with rank-0.
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
    """Raw fingerprint dict whose SHA-256 is :func:`_layout_signature`.

    Exposed separately so the offline cross-world-size reshard tool
    (``scripts/protrain/reshard_optim.py``) can recompute the signature
    against a new ``world_size`` without re-deriving the model layout
    from scratch. Mode-C save persists the dict as ``layout_fingerprint``
    in metadata.json so the reshard tool can read it directly.
    """
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
    """SHA-256 over the load-bearing layout fields.

    The signature catches model/architecture drift between save and
    load: a checkpoint built against one chunk geometry must not be
    quietly loaded against a different geometry. Inputs include the
    full per-chunk param-name ordering, S_chunk, N_chunk, the
    effective persistent set, and zero3_shard.

    Mode-aware on ``world_size``:

    * Mode-B (``zero3_shard=False``, replicated): every rank holds the
      FULL optimizer state, so cross-world resume is legitimate. The
      ``world_size`` argument is IGNORED in the hash so a save at N
      ranks matches a load at M ranks.
    * Mode-C (``zero3_shard=True``, sharded): each rank holds a
      different shard, so ``world_size`` IS part of compatibility and
      gets mixed into the hash. Cross-world resume must go through
      the offline reshard tool.
    """
    if not zero3_shard:
        # Replicated: drop world_size from the fingerprint so the
        # signature is rank-count-independent. Build a fresh dict
        # (rather than reusing _build_layout_fingerprint and popping)
        # to keep the canonical-JSON payload deterministic.
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
    """Estimated bytes for the optimizer's persisted Adam state.

    Walks each INNER adapter's ``state`` dict (``_gpu_optim._optim`` and
    every entry in ``_cpu_optim._optims``) and sums tensor bytes —
    counting exactly what gets pickled to disk modulo Python object
    overhead.

    Walking the user-facing ``optim.param_groups`` is wrong here:
    after :meth:`ChunkManager.materialize_offload` runs, every
    offloaded param's ``.data`` is replaced with an empty placeholder
    (manager.py:706 / :1494), so ``p.numel()`` returns 0 between
    training steps and the estimate misses every offloaded chunk's
    optimizer state. For 7B full-FT that's the difference between a
    silent 84 GB write and a correct gate trip.

    Pre-first-step the inner state dicts are empty and this returns 0
    — that's correct: there is no state to save yet, so any save would
    produce small placeholder files that can pass the gate.
    """
    import torch

    total = 0

    def _add_inner(inner_optim: Any) -> None:
        nonlocal total
        for state in getattr(inner_optim, "state", {}).values():
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    total += int(v.numel()) * int(v.element_size())

    gpu_optim = getattr(optim, "_gpu_optim", None)
    if gpu_optim is not None:
        inner = getattr(gpu_optim, "_optim", None)
        if inner is not None:
            _add_inner(inner)

    cpu_optim = getattr(optim, "_cpu_optim", None)
    if cpu_optim is not None:
        for inner in getattr(cpu_optim, "_optims", {}).values():
            _add_inner(inner)

    return total


def _build_regions_per_chunk(chunk_manager: Any) -> dict[str, list[dict[str, Any]]]:
    """Capture the per-chunk dtype-region layout from ``_chunk_shards``.

    Walks ``chunk_manager._chunk_shards`` and emits one descriptor per
    region per chunk. Used by the save side to persist Mode-C metadata
    and by the load side to compute the current run's regions for
    comparison against the saved descriptors.

    Keys are stringified ``ChunkId`` (JSON only allows string keys);
    values are ordered lists of region descriptors, position-aligned to
    the runtime ``regions`` list. Each descriptor carries the five
    load-bearing fields described in :class:`_DtypeRegion`:

    * ``chunk_offset`` — byte offset within the chunk
    * ``region_bytes`` — un-padded bytes
    * ``region_bytes_padded`` — rank-evenly-divisible padding
    * ``shard_bytes`` — bytes per rank for this region
    * ``dtype`` — ``str(region.dtype)`` (e.g. ``"torch.float16"``)
    """
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
    """Raise RuntimeError if Mode-C region layouts differ.

    Every field of every region must match by position: chunk_id set,
    region count per chunk, and per-region ``chunk_offset``,
    ``region_bytes``, ``region_bytes_padded``, ``shard_bytes``, and
    ``dtype`` (string-compared). Mismatch implies the saved per-rank
    shard tensors won't fit the rebuilt ``shard_param`` — fail loud
    with a useful message instead of letting ``load_state_dict`` crash
    deep in torch with an unhelpful shape error.

    The error message names the differing chunk + region index + field
    so a user reading the trace can map straight back to the divergent
    config (dtype mix, world_size, alignment).
    """
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
    """Normalize hyperparameter dict for save/load drift comparison.

    JSON serialization turns ``betas`` tuples into lists; converting
    list values back to tuples here keeps round-tripped data from
    triggering a spurious mismatch warning.
    """
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in hp.items()}


def _is_raw_protrain_optimizer(optim: Any) -> bool:
    """Duck-type for the raw _ProTrainOptimizer (avoids a circular import)."""
    return (
        hasattr(optim, "_gpu_optim")
        and hasattr(optim, "_cpu_optim")
        and hasattr(optim, "_chunk_manager")
    )


def _unwrap_protrain_optim(optim: Any) -> Any:
    """Return the raw _ProTrainOptimizer or None.

    HF Trainer + Accelerate wrap ``trainer.optimizer`` with
    ``AcceleratedOptimizer`` after Accelerate's ``prepare`` runs, and
    every callback fired post-prepare receives the wrapped form (see
    accelerate/optimizer.py: AcceleratedOptimizer stores the raw
    optimizer at ``.optimizer``). Without this unwrap, the callback's
    duck-type check fails on the wrapper and the save silently no-ops
    in real Trainer runs.
    """
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
    """Recursively hash a state_dict-like nested structure deterministically.

    pickle.dumps is NOT cross-process-deterministic for torch tensors:
    the pickle stream embeds Python-level metadata (storage offsets,
    type-class object IDs in some torch builds) that can drift between
    the two mp.spawn workers' independent CUDA contexts even when the
    tensor *values* are identical. We instead walk the nested dict and
    feed only the load-bearing bytes (tensor element bytes, scalar
    values, sorted dict keys) into the hash.
    """
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
            # Hash raw storage bytes via a uint8 view. Direct .numpy()
            # rejects bf16 ("Got unsupported ScalarType BFloat16") and
            # other torch-only dtypes — view-as-uint8 reinterprets the
            # storage as bytes and works for every fixed-width dtype.
            # ``flatten()`` first because ``view(torch.uint8)`` rejects
            # 0-dim tensors when the target element size differs (Adam's
            # ``step`` field is a scalar 0-dim tensor).
            if t.numel() > 0:
                h.update(t.flatten().view(torch.uint8).numpy().tobytes())
        else:
            # Scalar: int, float, bool, str, None, etc. repr() is
            # stable across processes.
            h.update(repr(obj).encode("utf-8"))

    _emit(sd)
    return h.digest()


def _hash_inner_state_dicts(optim: Any) -> str:
    """SHA-256 over the rank's inner optimizer state dicts.

    Used by the optional Mode-B cross-rank verify path (§2.4 of the
    Phase 2 design). Walks the same inner adapters the save path
    serializes (``_gpu_optim._optim`` and every entry in
    ``_cpu_optim._optims``) and folds each state_dict's structural
    bytes into the hash via :func:`_hash_state_dict`.
    """
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
    """Cross-rank state-equality check for Mode-B (opt-in, single shot).

    Each rank computes a SHA-256 over its inner state, all_gather_object
    the hashes, and raises if any rank disagrees with rank-0. Cheap
    insurance against the corner case where DDP determinism fails
    (numerical drift, manual override, etc.) so neither save nor load
    silently propagates a rank-0-only view of optimizer state.

    Deadlock guard: ``_hash_inner_state_dicts`` walks live optimizer
    state and can raise (e.g. shape mismatch in a saved-state restore on
    rank-0 only). If we proceeded straight to ``all_gather_object`` after
    a local exception, the failed rank would unwind out and the peers
    would block forever on the collective. We therefore (1) catch any
    local hashing exception, (2) fold a 0/1 status flag through
    ``_allreduce_status_or_raise`` so the cluster fails in lockstep, and
    (3) only invoke ``all_gather_object`` once every rank confirms it
    has a valid local hash.
    """
    if world_size <= 1 or not _dist_is_active():
        return
    local_hash = ""
    local_exc: BaseException | None = None
    try:
        local_hash = _hash_inner_state_dicts(optim)
    except BaseException as exc:  # noqa: BLE001 - re-raised after collective
        local_exc = exc
    # Surface any rank's hashing failure cluster-wide BEFORE the
    # all_gather_object so a rank-0-only exception cannot wedge the
    # peers (they would otherwise block forever on the collective).
    _allreduce_status_or_raise(
        1 if local_exc is not None else 0,
        op="verify-replicated-state (local hash)",
    )
    if local_exc is not None:
        # Cluster sum was non-zero AND this rank is one of the failing
        # ranks: re-raise the original exception so the actionable
        # traceback is preserved (peers raise the generic cluster-wide
        # error from ``_allreduce_status_or_raise``).
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
    """Write the protrain_optim/ subdirectory. Returns True iff written.

    Mode-B (DDP-replicated): only rank-0 writes; other ranks return True
    so the caller knows the save was performed cluster-wide via rank-0.

    Mode-C (ZeRO-3 sharded): rank-0 writes metadata + replicated
    persistent (GPU) state; every rank writes its own per-rank shard
    files for non-persistent chunks (``chunk_<N>_rank_<R>.pt``). The
    metadata records ``regions_per_chunk`` describing every chunk's
    dtype-region layout so the load side can validate alignment/dtype-
    mix invariants before torch's ``load_state_dict`` would otherwise
    crash with a shape error.

    Returns False (with a WARN) when the size estimate exceeds
    ``save_max_bytes``. The user opts in to large saves by raising
    that threshold via ``protrain_optim_save_max_bytes``. The HF-side
    optimizer.pt is independent — the plugin's ``save_only_model``
    knob controls that.

    ``rank`` and ``world_size`` are the HF Trainer's view (typically
    ``args.process_index`` / ``args.world_size``). ``world_size=None``
    falls back to ``_current_world_size`` for backward compatibility
    with Phase-1 callers.
    """
    chunk_manager = optim._chunk_manager
    if world_size is None:
        world_size = _current_world_size()
    zero3_shard = bool(getattr(chunk_manager, "zero3_shard", False))

    # Drain any in-flight async CPU Adam futures BEFORE estimating size
    # so the size-gate sees the post-step state. Otherwise a queue full
    # of pending futures could leave inner _optims state dicts empty/
    # smaller than reality, producing a stale estimate that bypasses
    # ``protrain_optim_save_max_bytes`` and proceeds to a write that
    # would have been gated. Every rank drains its own queue.
    chunk_manager.wait_cpu_optim_all()

    estimate = _estimate_optim_state_bytes(optim)
    # The callback already runs a rank-0-broadcast size-gate before
    # calling here (see ProTrainOptimizerCheckpointCallback.on_save),
    # so re-running it here per-rank would let a non-rank-0 local trip
    # diverge from rank-0's cluster-wide decision — in Mode-C that would
    # leave a partial checkpoint where rank-0's metadata says "saved"
    # but rank-N's per-rank shards are missing. Skip the redundant gate
    # in that path; the legacy direct caller (Phase-1 single-rank) keeps
    # the gate by leaving _skip_size_gate at its default False.
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

    if zero3_shard:
        # ---------- Mode-C sharded save ----------
        # Rank-0 owns metadata + replicated GPU state; every rank writes
        # its own per-rank chunk shard files. We barrier between the
        # rank-0 writes and the chunk-shard writes so non-zero ranks
        # don't race ahead of the directory creation. A trailing barrier
        # in the caller (the callback) ensures the cluster sees a fully
        # complete dir before downstream code touches it.
        #
        # Failure protocol (Finding 1): rank-0's writes can raise mid-
        # call (ENOSPC, perm denied, json serialization, ...). Without
        # the broadcast below, non-rank-0 ranks would block forever on
        # the next ``_barrier_or_noop()``. Wrap rank-0's writes in
        # try/except, broadcast a 0/1 status flag from rank-0 to every
        # rank in a ``finally`` so it executes even on the rank-0
        # exception path, then ranks raise in lockstep.
        rank0_status = 0
        try:
            if rank == 0:
                # Reset the dir before reusing it: a partial save or a
                # replayed ``checkpoint-<step>`` could otherwise leave
                # stale ``gpu_optim.pt`` / ``cpu_optim/*.pt`` files
                # behind, and the load side treats those extras as hard
                # mismatches (so a retry could leave an otherwise-good
                # save unloadable).
                shutil.rmtree(target, ignore_errors=True)
                os.makedirs(target, exist_ok=False)

                _fp = _build_layout_fingerprint(chunk_manager, world_size, zero3_shard)
                metadata = {
                    "format_version": SCHEMA_FORMAT_VERSION,
                    "protrain_layout_signature": _layout_signature_from_fingerprint(
                        _fp
                    ),
                    # Raw fingerprint persisted so the offline cross-world-
                    # size reshard tool can recompute the signature for a
                    # new world_size without re-deriving the model layout.
                    # Mode-C only: Mode-B doesn't need it (replicated
                    # state is rank-independent and the load path
                    # tolerates world_size drift natively).
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
                with open(os.path.join(target, METADATA_FILENAME), "w") as f:
                    json.dump(metadata, f, indent=2, sort_keys=True)

                if optim._gpu_optim is not None:
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
            # Broadcast rank-0's status to every rank BEFORE the barrier
            # so a mid-write rank-0 failure does not deadlock the cluster.
            # Non-rank-0 ranks raise a synthetic RuntimeError; rank-0
            # re-raises its original exception via the bare ``raise``
            # above.
            _broadcast_status_or_raise(
                rank0_status, src=0, op="save (rank-0 metadata/gpu_optim)"
            )

        # Barrier so non-rank-0 ranks see metadata + cpu_optim/ before
        # writing into the dir.
        _barrier_or_noop()

        # Every rank writes its own per-rank shard files. Rank-0 also
        # writes its shards here (no separate path).
        #
        # Failure protocol (Finding 1, per-rank phase): if any rank's
        # ``torch.save`` raises (ENOSPC on a NFS rank, perm denied on a
        # rank-local tmp, ...), surviving ranks would block on the
        # callback's trailing barrier. All-reduce a SUM of per-rank
        # statuses; if any rank failed, every rank raises so the cluster
        # fails in lockstep.
        per_rank_status = 0
        try:
            if optim._cpu_optim is not None and optim._cpu_optim._optims:
                cpu_dir = os.path.join(target, CPU_OPTIM_DIRNAME)
                # Require the rank-0 checkpoint tree to be visible on every
                # rank before writing shards. If ``target`` is missing on a
                # non-zero rank, ``output_dir`` is not actually a shared
                # filesystem and an implicit ``makedirs`` would manufacture a
                # local shard dir whose chunk_<N>_rank_<R>.pt files would be
                # invisible to rank 0 -- the save would look successful but
                # be unresumable. Fail loudly instead.
                if not os.path.isdir(target):
                    raise RuntimeError(
                        f"ProTrain optimizer save: checkpoint directory "
                        f"{target!r} is not visible on rank {rank}. Mode-C "
                        "saves require a shared filesystem across all ranks."
                    )
                # Defensive mkdir on every rank in case dist isn't actually
                # initialized (single-rank zero3_shard "test mode" run that
                # falls back to replicated behaviour but still wants the
                # Mode-C disk shape).
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
                "world_size=%d, save_mode=%s)",
                target,
                estimate,
                len(_effective_persistent_ids(chunk_manager)),
                len(optim._cpu_optim._optims) if optim._cpu_optim is not None else 0,
                step,
                world_size,
                SAVE_MODE_SHARDED,
            )
        return True

    # ---------- Mode-B replicated save (rank-0-only write) ----------
    # Failure protocol: only rank-0 writes here, while every rank
    # participates in the callback's trailing barrier. Any exception
    # during rank-0's write block would leave the other ranks blocked on
    # that barrier forever. Wrap the rank-0 write in try/except/finally
    # and broadcast a 0/1 status flag from rank-0 BEFORE rank-0 re-raises
    # its original exception, so non-rank-0 ranks raise a synthetic
    # RuntimeError and the cluster fails in lockstep.
    persistent_ids = _effective_persistent_ids(chunk_manager)
    rank0_status = 0
    try:
        if rank == 0:
            # Reset the dir before reusing it: a partial save or a
            # replayed ``checkpoint-<step>`` could otherwise leave
            # stale ``gpu_optim.pt`` / ``cpu_optim/*.pt`` files behind,
            # and the load side treats those extras as hard mismatches
            # (so a retry could leave an otherwise-good save unloadable).
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
            with open(os.path.join(target, METADATA_FILENAME), "w") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)

            if optim._gpu_optim is not None:
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

    if rank == 0:
        LOG.info(
            "ProTrain optimizer save: wrote %s (estimate=%d bytes, "
            "persistent=%d chunks, cpu_chunks=%d, step=%d, "
            "world_size=%d, save_mode=%s)",
            target,
            estimate,
            len(persistent_ids),
            len(optim._cpu_optim._optims) if optim._cpu_optim is not None else 0,
            step,
            world_size,
            SAVE_MODE_REPLICATED,
        )
    return True


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def _perform_online_reshard(
    original_target: str,
    saved_world: int,
    current_world: int,
) -> str:
    """Run the online Mode-C reshard against a sibling temp dir.

    Rank-0 invokes :func:`reshard_mode_c_shards` on
    ``original_target`` writing to ``original_target/.reshard_to_N<W>/``.
    Every rank then participates in the lockstep failure protocol via
    :func:`_broadcast_status_or_raise` (mirrors the Mode-C save side's
    rank-0-writes-only sections), and a trailing barrier ensures
    non-zero ranks see the temp dir's files before they read them.

    Returns the temp-dir path on success. Raises ``RuntimeError`` on
    any rank if the rank-0 reshard failed. The temp dir is left on
    disk for post-mortem inspection on failure — the caller is
    responsible for cleanup on the success path (after every rank
    has finished reading).
    """
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

    # Lockstep failure protocol (mirrors Mode-C save's rank-0 sections,
    # e.g. metadata.json / gpu_optim.pt): rank-0 attempts the reshard
    # inside try/except, broadcasts a 0/1 status via
    # ``_broadcast_status_or_raise``. Non-zero status raises a
    # synthesised RuntimeError on every non-source rank so the cluster
    # fails together rather than wedging the surviving ranks at the
    # trailing barrier.
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
            # Pre-clean stale temp dir from a previous interrupted run
            # so we never read mixed bytes.
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

    # Barrier so non-rank-0 ranks see the temp dir's files before they
    # try to read them. The reshard writes
    # cpu_optim/chunk_*_rank_*.pt and metadata.json under ``temp_dir``;
    # without this barrier, a fast rank-1 could enter the per-rank
    # read block before rank-0 finishes the last torch.save().
    _barrier_or_noop()

    return temp_dir


def _load_protrain_optim_dir(
    optim: Any,
    checkpoint_dir: str,
    *,
    allow_online_reshard: bool = False,
) -> bool:
    """Load a previously saved protrain_optim/ subdirectory in-place.

    Returns True iff the directory existed and was loaded (or False if
    the checkpoint dir simply has no ProTrain shard, which is the
    normal "first run / opt-out" case).

    Raises RuntimeError on any mismatch the saved metadata flags
    against the current run (zero3_shard, save_mode, layout signature,
    persistent_ids set, missing per-chunk file).

    World-size mismatch policy (CHECKPOINT_DESIGN_PHASE2.md §4.1
    Option B + opt-in C): Mode-B replicated saves are tolerated across
    world_size changes — the on-disk state is rank-independent. Mode-C
    sharded saves default to a hard error on world_size mismatch (the
    shard arithmetic depends on world_size). When the caller passes
    ``allow_online_reshard=True``, the load path instead invokes the
    same reshard logic as the offline tool
    (:func:`axolotl.integrations.protrain.api.reshard.reshard_mode_c_shards`)
    on rank-0 against a temp dir, barriers all ranks, then loads from
    the temp dir as if it had been natively saved at the current
    world_size. The temp dir is cleaned up on successful load (rank-0
    only); failures leave it behind for post-mortem.

    Mode-C also enforces the per-chunk dtype-region layout: the saved
    ``regions_per_chunk`` descriptors must match the current run's
    region layout exactly (chunk_offset, region_bytes,
    region_bytes_padded, shard_bytes, dtype). Any mismatch implies the
    saved per-rank shard tensors won't fit the rebuilt ``shard_param``
    — fail loud with a useful message instead of letting torch's
    ``load_state_dict`` crash deep with a shape error.

    Forward compatibility: ``format_version=1`` saves are read as
    Mode-B replicated with ``saving_rank=0`` and ``world_size=1``
    (CHECKPOINT_DESIGN_PHASE2.md §5).

    All torch.load calls use map_location='cpu'. Inner load_state_dict
    handles device placement per-tensor (GPU adam → GPU, CPU adam →
    CPU), which is correct because the inner state_dicts already hold
    the right device tags.
    """
    original_target = os.path.join(checkpoint_dir, PROTRAIN_OPTIM_DIRNAME)
    target = original_target

    # Cross-rank visibility consensus on ``target`` and ``meta_path``.
    # The Mode-C save path already enforces ``os.path.isdir(target)``
    # per-rank before writing shards (see ``_save_protrain_optim_dir``),
    # but the load side previously gated on rank-local stat() calls. If
    # one rank misses the directory while others see it -- e.g.
    # ``output_dir`` is a node-local filesystem masquerading as shared,
    # or rank-0 wrote shards visible only to itself -- the rank-local
    # check would silently let some ranks skip ProTrain restore while
    # others tried to load, leaving the cluster with a mixed optimizer
    # state. Mirror the per-rank-shard-read sync up-front: every rank
    # skips, every rank loads, or every rank fails.
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
        # Forward compat: v1 saves predate the save_mode / saving_rank
        # fields. They're known to be single-rank non-ZeRO replicated
        # by Phase 1's hard guard.
        metadata.setdefault("protrain_save_mode", SAVE_MODE_REPLICATED)
        metadata.setdefault("saving_rank", 0)
        metadata.setdefault("protrain_world_size", 1)
        metadata.setdefault("protrain_zero3_shard", False)
    elif fmt == SCHEMA_FORMAT_VERSION:
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
    else:
        raise RuntimeError(
            f"ProTrain optimizer load: unknown format_version={fmt} "
            f"(this build expects {SCHEMA_FORMAT_VERSION}). Refusing to load."
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
        # ---------- Mode-C sharded load ----------
        # We've already validated saved_mode == SAVE_MODE_SHARDED above
        # via the save-mode mismatch check; this is the genuine Mode-C
        # resume path.

        # World-size policy (§4.1): Mode-C is hard-error on world_size
        # mismatch by default. Sharded shard arithmetic
        # (region_bytes_padded / world_size = shard_bytes) depends on
        # world_size, so cross-world-size resume requires a re-shard
        # step. Two routes exist:
        #
        # * Default (``allow_online_reshard=False``): hard error,
        #   point the user at the offline tool. The offline path is
        #   the conservative default — explicit user action means the
        #   user knows world_size changed and accepts the cost.
        # * Opt-in (``allow_online_reshard=True``): rank-0 invokes the
        #   shared reshard logic against a temp dir under
        #   ``original_target/.reshard_to_N<W>/``, all ranks barrier on
        #   the result via ``_broadcast_status_or_raise`` (mirroring
        #   the Mode-C save's lockstep failure protocol), then the
        #   load proceeds against the temp dir as if it were a
        #   natively-N=W save. Cleanup on successful load.
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

            # Online reshard: rank-0 writes a sibling temp dir whose
            # name encodes the new world size for forensic clarity;
            # ``_perform_online_reshard`` runs the lockstep failure
            # protocol and the trailing barrier so non-zero ranks see
            # the resharded files before they read them. The temp dir
            # is intentionally left on disk if the helper raises so a
            # developer can inspect the failure; on success the caller
            # cleans it up after every rank has finished reading.
            online_reshard_temp_dir = _perform_online_reshard(
                original_target,
                saved_world=saved_world,
                current_world=current_world,
            )

            # Re-point the load at the resharded dir and reload
            # metadata. ``saved_world`` is now == ``current_world``
            # by construction so the rest of the Mode-C body becomes
            # the standard same-world load path.
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

        # Region-layout match (§3.5). Every region descriptor must
        # match exactly — any drift in chunk_offset, region_bytes,
        # region_bytes_padded, shard_bytes, or dtype implies the saved
        # bytes won't fit the rebuilt shard_param.
        saved_regions = metadata.get("regions_per_chunk")
        if saved_regions is None:
            raise RuntimeError(
                "ProTrain optimizer load: sharded metadata missing "
                "required field 'regions_per_chunk'. The save predates "
                "Mode-C support or the file is corrupt."
            )
        current_regions = _build_regions_per_chunk(chunk_manager)
        _validate_regions_match(saved_regions, current_regions)

        # Layout signature embeds world_size + zero3_shard; recompute
        # against the saved values for the comparison since saved_world
        # == current_world here.
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

        # Resolve this rank's ordinal. The load path is fired from the
        # monkey-patched ``_load_optimizer_and_scheduler`` and doesn't
        # have ready access to the HF TrainingArguments, so fall back
        # to torch.distributed.get_rank() when dist is initialised; on
        # single-rank runs (zero3_shard degraded to no-op) rank=0.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            current_rank = int(torch.distributed.get_rank())
        else:
            current_rank = 0

        # Per-rank chunk shard load. Walk the current set of non-
        # persistent chunks and require every rank-suffixed file to
        # exist. Missing file / unexpected file / corrupt file = hard
        # error.
        #
        # Failure protocol (Finding 2): each rank reads its own shard. A
        # missing or corrupt file on any rank would raise locally; the
        # surviving ranks would then block on the load hook's trailing
        # barrier. Wrap the whole per-rank load in try/except and
        # all-reduce a SUM of statuses; if any rank failed, every rank
        # raises so the cluster fails in lockstep.
        #
        # The replicated gpu_optim.pt read also lives inside this
        # synchronized block: although the file itself is identical
        # across ranks, a missing/corrupt file or a torch.load failure
        # on any single rank would otherwise raise locally and leave
        # peers blocked on the trailing barrier (CR finding 3191143358).
        # Folding the read into the same try/except + allreduce ensures
        # rank-local failures abort uniformly.
        #
        # Stray-file rejection (Finding 3): Mode-B explicitly rejects
        # unknown files in cpu_optim/ via CHUNK_FILE_RE. Mode-C's old
        # behaviour silently tolerated extras (e.g. ``chunk_X_rank_8.pt``
        # left over from a higher-world_size save). Mirror Mode-B's
        # pattern: enumerate cpu_optim/ and reject anything that
        # (a) doesn't match CHUNK_SHARD_FILE_RE,
        # (b) carries a rank ordinal outside ``[0, current_world)`` —
        #     these match the filename grammar but are leftovers from a
        #     larger-world_size save and would silently slip past a
        #     pure regex check, or
        # (c) carries a chunk ID that isn't in the current set of
        #     non-persistent chunk IDs — a syntactically valid filename
        #     for a chunk that the current run does not own (e.g.
        #     leftover from a different partition / persistent_ids
        #     override). Mode-B catches the equivalent case via the
        #     ``saved_cpu_ids != current_cpu_ids`` set comparison; the
        #     Mode-C per-rank loop only opens the files it expects, so
        #     stray chunk IDs would otherwise sit unread on disk and
        #     mask a real partition mismatch.
        # Done up-front (inside the try/except so the cross-rank failure
        # protocol applies) before any torch.load runs.
        cpu_dir = os.path.join(target, CPU_OPTIM_DIRNAME)
        expected_cpu_ids = (
            set(int(cid) for cid in optim._cpu_optim._optims)
            if optim._cpu_optim is not None
            else set()
        )
        load_status = 0
        try:
            # Persistent (GPU) state is replicated across ranks; every
            # rank loads from the same gpu_optim.pt. map_location='cpu'
            # defeats HF Trainer's hostile map_location=device default.
            # Folded into the synchronized block so a rank-local failure
            # (missing file, corrupt file, load error) participates in
            # the lockstep abort instead of deadlocking peers at the
            # trailing barrier.
            gpu_path = os.path.join(target, GPU_OPTIM_FILENAME)
            if os.path.isfile(gpu_path):
                if optim._gpu_optim is None:
                    raise RuntimeError(
                        "ProTrain optimizer load: gpu_optim.pt present on "
                        "disk but current optimizer has no persistent (GPU) "
                        "inner — partition mismatch slipped past the layout-"
                        "signature check."
                    )
                loaded = torch.load(gpu_path, map_location="cpu", weights_only=True)
                optim._gpu_optim._optim.load_state_dict(loaded)
            elif optim._gpu_optim is not None:
                raise RuntimeError(
                    "ProTrain optimizer load: current optimizer has a "
                    "persistent (GPU) inner but gpu_optim.pt is absent on "
                    "disk."
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
                    # Defensive: torch.optim.Optimizer.load_state_dict
                    # auto-casts state tensors to the device of the matching
                    # param. Post-materialize_offload, the user-facing
                    # shard_param holds an empty placeholder on the manager's
                    # device — torch silently moves the loaded exp_avg /
                    # exp_avg_sq there. The DeepSpeedCPUAdam C++ kernel then
                    # segfaults on the next step trying to write through
                    # that pointer. Force CPU after load_state_dict.
                    for state in inner.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                                state[k] = v.cpu()
        except Exception:
            load_status = 1
            raise
        finally:
            _allreduce_status_or_raise(load_status, op="load (per-rank shard read)")

        # Hyperparam drift: warn but accept. ``zip`` runs without
        # ``strict=True`` because the count-mismatch case is handled by
        # the explicit warning above (R8): aborting here with a
        # ValueError would contradict the documented "warn and accept"
        # contract.
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

        # Cleanup: if we used the online reshard path, rank-0 deletes
        # the temp dir now that every rank has finished reading from
        # it. We barrier first so rank-0 can't unlink shard files
        # mid-read. On exception above, the function exits without
        # hitting this block — the temp dir is intentionally left for
        # post-mortem inspection.
        if online_reshard_temp_dir is not None:
            _barrier_or_noop()
            if current_rank == 0 and os.path.isdir(online_reshard_temp_dir):
                try:
                    shutil.rmtree(online_reshard_temp_dir)
                except OSError as cleanup_exc:
                    # Cleanup failure is non-fatal — the load already
                    # succeeded. Log and continue; user can manually
                    # rm -rf the temp dir later.
                    LOG.warning(
                        "ProTrain optimizer load: failed to clean up "
                        "online reshard temp dir %s: %s",
                        online_reshard_temp_dir,
                        cleanup_exc,
                    )
        return True

    # Mode-B replicated load (current scope). World-size differences
    # are tolerated per Option B — replicated state is shape-
    # independent of world_size.
    if saved_world != current_world:
        LOG.info(
            "ProTrain optimizer load: replicated checkpoint saved with "
            "world_size=%d loading into world_size=%d. Replicated state "
            "is rank-independent, so this is supported.",
            saved_world,
            current_world,
        )

    # Layout signature embeds world_size, so a world_size delta would
    # naively trip the signature check. Recompute the saved signature's
    # would-be value at the CURRENT world_size for the comparison —
    # the only legitimately load-bearing layout fields here are chunk
    # geometry + persistent_ids + zero3_shard.
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

    # Failure protocol (Mode-B replicated load): every rank reads the
    # same shared files (gpu_optim.pt + cpu_optim/chunk_<N>.pt). A
    # ``torch.load`` or ``load_state_dict`` failure on ANY rank would
    # cause that rank to raise and bypass the install_load_hook trailing
    # barrier — surviving ranks would then deadlock. All-reduce a SUM of
    # per-rank statuses across the whole read block; if any rank failed,
    # every rank raises so the cluster fails in lockstep. Mirrors the
    # Mode-C per-rank shard load pattern.
    load_status = 0
    captured_exc: Exception | None = None
    try:
        # GPU optim: load if both saved file and current optim slot exist.
        gpu_path = os.path.join(target, GPU_OPTIM_FILENAME)
        if os.path.isfile(gpu_path):
            if optim._gpu_optim is None:
                raise RuntimeError(
                    "ProTrain optimizer load: gpu_optim.pt present on disk but "
                    "current optimizer has no persistent (GPU) inner — partition "
                    "mismatch slipped past the layout-signature check."
                )
            loaded = torch.load(gpu_path, map_location="cpu", weights_only=True)
            optim._gpu_optim._optim.load_state_dict(loaded)
        elif optim._gpu_optim is not None:
            raise RuntimeError(
                "ProTrain optimizer load: current optimizer has a persistent "
                "(GPU) inner but gpu_optim.pt is absent on disk."
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
                # ``torch.optim.Optimizer.load_state_dict`` auto-casts every
                # state tensor to the device of the matching param. After
                # ``ChunkManager.materialize_offload`` runs, the user-facing
                # params held by the inner CPU adam have empty GPU
                # placeholders for ``.data`` — so torch silently moves the
                # loaded ``exp_avg`` / ``exp_avg_sq`` tensors to CUDA. The
                # DeepSpeedCPUAdam C++ kernel then segfaults on the next
                # step trying to write through a GPU pointer. Force the
                # inner CPU adam state back to CPU after the cast.
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
        # When dist is inactive and our local status is non-zero, the
        # helper synthesizes a generic RuntimeError. Prefer the caller's
        # original exception (captured below) over the helper's
        # synthesized one — it carries the actual error context (e.g.
        # "CPU chunk set mismatch", "weights_only=True rejected ...").
        # When dist IS active and our local status is non-zero, the
        # helper short-circuits and returns silently so we never reach
        # this branch on the local-failure path. The branch fires on
        # remote-rank failures (helper raises a synthetic RuntimeError),
        # which is the right exception to surface.
        if captured_exc is None:
            raise
    if captured_exc is not None:
        raise captured_exc

    # Cross-rank state-equality check: a successful Mode-B load proves
    # nothing about whether each rank restored the SAME bytes. If
    # ``output_dir`` exists on every node but with different local files,
    # the run can silently resume with divergent Adam state across DDP
    # ranks. Re-use the save-side helper (short-circuits on world_size<=1
    # / dist inactive) to fingerprint inner state and raise everywhere on
    # disagreement.
    _verify_replicated_state_across_ranks(optim, world_size=current_world)

    # Hyperparam drift: warn but accept. JSON serialization turns
    # ``betas`` tuples into lists; normalize before comparing so
    # round-tripped data doesn't trigger a spurious warning. ``zip``
    # runs without ``strict=True`` because the count-mismatch case is
    # handled by the explicit warning above (R8): aborting here with a
    # ValueError would contradict the documented "warn and accept"
    # contract.
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
    """Lazy-imported callback class — keeps ``transformers`` out of the
    module-import path so unit tests that don't need HF can stay light."""
    from transformers.trainer_callback import TrainerCallback

    class ProTrainOptimizerCheckpointCallback(TrainerCallback):
        """``on_save``: write protrain_optim/ beside HF's checkpoint dir.

        Reads the optimizer off ``kwargs['optimizer']`` (HF passes it in
        on every callback). Routes the save through
        ``_save_protrain_optim_dir``, which enforces the gating + scope
        checks and dispatches between Mode-B (replicated, rank-0-only
        write) and Mode-C (sharded, per-rank shard write). Failures are
        loud (raise) — silently producing an unloadable checkpoint is
        worse than crashing on save.

        HF's ``on_save`` fires on every rank
        (``_maybe_log_save_evaluate`` calls ``callback_handler.on_save``
        unconditionally). The callback orchestrates the cross-rank
        coordination needed by both modes:

        * Every rank drains ``wait_cpu_optim_all`` (CPU adam must be
          quiescent before any rank snapshots).
        * Rank-0 computes the size-gate decision; the decision is
          broadcast so all ranks act consistently (no partial saves).
        * Optional opt-in (Mode-B only): on the FIRST save of each run,
          every rank hashes its inner state and ``all_gather_object``-s
          the hashes to verify Mode-B's replication invariant. Skipped
          on subsequent saves to keep per-save overhead low.
        * Mode-B: rank-0 writes; other ranks no-op.
        * Mode-C: rank-0 writes metadata + replicated GPU state; every
          rank writes its own per-rank chunk shard files.
        * ``dist.barrier()`` at exit so callers see a complete dir.
        """

        def __init__(
            self,
            *,
            save_max_bytes: int,
            verify_replicated: bool = False,
        ) -> None:
            """Store save policy and one-shot replication-verify flag."""
            self._save_max_bytes = save_max_bytes
            self._verify_replicated = bool(verify_replicated)
            # Track whether the cross-rank verify already fired for
            # this run; we only do it on the first save (cheap insurance
            # at run start, but per-save would be expensive).
            self._verify_replicated_done = False

        def on_save(
            self,
            args: "TrainingArguments",
            state: "TrainerState",
            control: "TrainerControl",
            **kwargs: Any,
        ) -> "TrainerControl":
            """Persist the ProTrain optimizer state alongside the HF checkpoint dir."""
            # Trainer.optimizer is wrapped by AcceleratedOptimizer after
            # prepare runs; the callback receives the wrapped form. Unwrap
            # before the duck-type guard.
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
            # Only rank-0 sees the HF-created checkpoint dir on multi-
            # rank runs (`should_save` gates HF's mkdir). The other
            # ranks must still drain their CPU adam and participate in
            # the broadcast / barrier so the cross-rank protocol stays
            # in sync — but if rank-0 itself doesn't see the dir, that's
            # the legitimate "skip" case. Capture the missing-dir
            # decision here without early-returning: an early-return on
            # rank-0 would skip the lockstep preamble (drain + status
            # all-reduce) that non-zero ranks still execute, leaving
            # peers wedged in `_allreduce_status_or_raise` while rank-0
            # waits in broadcast/barrier. Instead we feed
            # `checkpoint_dir_missing` into the existing skip flow so
            # every rank reaches the same collectives in the same order.
            checkpoint_dir_missing = rank == 0 and not os.path.isdir(checkpoint_dir)

            # ---------- 1-3. Pre-save preamble under lockstep protocol ----------
            # Failure protocol: ``wait_cpu_optim_all()``, rank-0's
            # ``_estimate_optim_state_bytes`` size estimate, and the
            # one-shot ``_verify_replicated_state_across_ranks`` all run
            # before the first synchronized status exchange. If any of
            # those raises on only a subset of ranks, surviving ranks
            # would wedge in ``_broadcast_object_list_or_noop``,
            # ``all_gather_object``, or the trailing ``_barrier_or_noop``.
            # All-reduce a SUM of per-rank statuses around the whole
            # preamble; any rank's failure propagates to every rank so
            # the cluster fails in lockstep. ``skip_decision`` and
            # ``self._verify_replicated_done`` are only committed after
            # the synchronized status check confirms every rank
            # succeeded.
            preamble_status = 0
            skip = False
            verify_fired = False
            estimate = 0
            try:
                # ---------- 1. Drain CPU adam on every rank ----------
                chunk_manager.wait_cpu_optim_all()

                # ---------- 2. Estimate-gate (rank-0 decides) ----------
                if rank == 0:
                    if checkpoint_dir_missing:
                        # Missing-dir takes precedence: skip without
                        # estimating (the dir we'd write to isn't
                        # there). Log here so the warning still fires
                        # exactly once on rank-0, matching the prior
                        # early-return behavior.
                        skip = True
                        LOG.warning(
                            "ProTrainOptimizerCheckpointCallback.on_save: "
                            "expected checkpoint dir %s does not exist on "
                            "rank-0; skipping ProTrain shard.",
                            checkpoint_dir,
                        )
                    else:
                        estimate = _estimate_optim_state_bytes(raw)
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

                # ---------- 3. Cross-rank verify (opt-in, once per run) ----------
                # Mode-B only: in Mode-C every rank's inner state
                # intentionally differs (per-rank shard), so cross-rank
                # hashing would falsely raise. The schema documents "Has
                # no effect on single-rank or ZeRO-3 sharded runs" —
                # ``world_size > 1`` covers single-rank; ``not
                # zero3_shard`` covers Mode-C.
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

            # Commit one-shot verify state only after the synchronized
            # status check confirmed every rank's preamble succeeded.
            if verify_fired:
                self._verify_replicated_done = True

            # ---------- 2b. Broadcast skip decision ----------
            # Rank-0's gate decision goes out to every rank; non-rank-0
            # writes a placeholder that the broadcast overwrites.
            skip_decision = [skip]
            _broadcast_object_list_or_noop(skip_decision, src=0)
            if skip_decision[0]:
                _barrier_or_noop()
                return control

            # ---------- 4. Write per-mode ----------
            # Mode-B: rank-0 writes everything; non-zero ranks return
            # without writing. Mode-C: rank-0 writes metadata + GPU
            # state; every rank writes its own per-rank shards. The
            # dispatcher inside _save_protrain_optim_dir routes both
            # cases — the callback just hands off and barriers.
            _save_protrain_optim_dir(
                raw,
                checkpoint_dir,
                step=int(state.global_step),
                save_max_bytes=self._save_max_bytes,
                rank=rank,
                world_size=world_size,
                # Callback already broadcast rank-0's gate decision; the
                # inner per-rank gate must NOT re-trip independently.
                _skip_size_gate=True,
            )

            # ---------- 5. Barrier so downstream code sees the dir ----------
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
    """Wrap ``trainer._load_optimizer_and_scheduler`` to also load ProTrain.

    HF's TrainerCallback API has no ``on_load_checkpoint``;
    ``on_train_begin`` fires AFTER the load slot. This patch is the
    only correct lifecycle position. Symmetric with the existing
    optim.state_dict / optim.load_state_dict monkey-patches in
    plugin.py: the no-op patches stay (they coexist with Accelerate's
    prepare round-trip), and this load hook handles real resume via a
    completely separate path.

    The closed-over ``optim`` is captured at install time (in
    ``post_trainer_create``, BEFORE Accelerate.prepare wraps the
    optimizer), so it's already raw. We unwrap defensively in case
    the caller hands in a wrapper.

    The ``allow_online_reshard`` flag plumbs through to
    :func:`_load_protrain_optim_dir`. Default False keeps the Mode-C
    cross-world-size load path a hard error; setting True opts the
    user into the online reshard surface (rank-0 reshards into a temp
    dir, all ranks barrier and load). See CHECKPOINT_DESIGN_PHASE2.md
    §4.1.
    """
    raw = _unwrap_protrain_optim(optim)
    if raw is None:
        # Caller passed something that isn't a ProTrain optimizer —
        # silently no-op rather than installing a hook that would
        # never fire.
        return

    original = trainer._load_optimizer_and_scheduler

    def _patched(checkpoint: str | None) -> None:
        # Failure protocol: ``original(checkpoint)`` (the native HF
        # optimizer/scheduler load) is outside any cluster-wide status
        # handling, but the patched method still executes a distributed
        # barrier on the success path. If the native HF load fails on
        # one rank only, surviving ranks would otherwise wedge on the
        # trailing barrier. Wrap ``original`` in try/except, capture
        # ``sys.exc_info()`` so the original traceback is preserved,
        # only run ``_load_protrain_optim_dir`` on the success path,
        # always run the lockstep barrier, then re-raise the captured
        # exception after the barrier so the cluster fails in lockstep.
        original_exc_info: Any = None
        hf_load_status = 0
        peer_hf_failure: Exception | None = None
        try:
            original(checkpoint)
        except Exception:
            hf_load_status = 1
            original_exc_info = sys.exc_info()

        # Synchronize the native-HF load result across ranks BEFORE any rank
        # enters ``_load_protrain_optim_dir`` (which runs its own collectives).
        # Otherwise, a one-rank HF failure would leave that rank waiting at
        # the trailing barrier while surviving ranks dive into the ProTrain
        # load path's collectives → cluster wedge. ``_allreduce_status_or_raise``
        # makes every rank raise in lockstep when any rank reports failure.
        try:
            _allreduce_status_or_raise(
                hf_load_status, op="load (HF optimizer/scheduler)"
            )
        except Exception as exc:
            # Local-failure ranks already have ``original_exc_info`` set and
            # _allreduce_status_or_raise returns without raising for them.
            # Surviving ranks land here: capture the peer-failure marker so
            # we still skip the ProTrain load path and hit the same trailing
            # barrier as the failed ranks.
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
                # Run the lockstep barrier before re-raising so a
                # ProTrain-load failure on one rank doesn't wedge the
                # cluster on the next collective.
                _barrier_or_noop()
                raise
        # Defensive barrier: every rank loaded its own copy of the
        # files; the barrier just ensures the cluster moves past the
        # load slot in lockstep before training resumes. Cheap on
        # single-rank (no-op).
        _barrier_or_noop()
        if original_exc_info is not None:
            # Re-raise the original HF load failure with its original
            # traceback intact, AFTER the barrier so surviving ranks
            # don't wedge.
            raise original_exc_info[1].with_traceback(original_exc_info[2])
        if peer_hf_failure is not None:
            # Surviving rank: a peer's HF load failed. Raise after the
            # trailing barrier so the cluster fails in lockstep.
            raise peer_hf_failure

    trainer._load_optimizer_and_scheduler = _patched  # type: ignore[method-assign]


__all__ = [
    "CHUNK_SHARD_FILE_RE",
    "DEFAULT_SAVE_MAX_BYTES",
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
