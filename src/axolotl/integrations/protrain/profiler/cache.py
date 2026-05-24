"""On-disk JSON cache for ProfilerTrace, keyed by (arch_hash, bs, seq, sku, world)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from axolotl.integrations.protrain.types import (
    BlockId,
    OpId,
    OpRecord,
    ProfilerTrace,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_CACHE_SUBDIR = Path("protrain") / "profiler"

# Bump on any schema change to ProfilerTrace; old payloads are then ignored.
TRACE_VERSION = 23


@dataclass(frozen=True)
class ProfilerCacheKey:
    """Identity of a cached trace (re-profile trigger)."""

    arch_hash: str
    bs: int
    seq: int
    sku: str
    world: int

    def fingerprint(self) -> str:
        """sha256 hex digest; TRACE_VERSION prefix invalidates old entries."""
        raw = f"v{TRACE_VERSION}|{self.arch_hash}|{self.bs}|{self.seq}|{self.sku}|{self.world}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_root(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the profiler cache root (explicit dir wins, else XDG_CACHE_HOME or ~/.cache)."""
    if cache_dir is not None:
        return Path(cache_dir) / _CACHE_SUBDIR
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / _CACHE_SUBDIR


def _path_for(
    key: ProfilerCacheKey,
    cache_dir: str | os.PathLike[str] | None = None,
) -> Path:
    return _cache_root(cache_dir) / f"{key.fingerprint()}.json"


# JSON (de)serialization: tuples→list, BlockId/OpId dict keys str→int, BlockMode str→enum.


def _op_record_to_dict(op: OpRecord) -> dict[str, Any]:
    return {
        "op_id": int(op.op_id),
        "module_path": op.module_path,
        "qualified_name": op.qualified_name,
        # tuple[tuple[int, ...], ...] → list[list[int]]
        "shape_signature": [list(s) for s in op.shape_signature],
        "block_id": None if op.block_id is None else int(op.block_id),
        "is_forward": bool(op.is_forward),
    }


def _op_record_from_dict(d: dict[str, Any]) -> OpRecord:
    return OpRecord(
        op_id=OpId(int(d["op_id"])),
        module_path=str(d["module_path"]),
        qualified_name=str(d["qualified_name"]),
        # list[list[int]] → tuple[tuple[int, ...], ...]
        shape_signature=tuple(tuple(int(x) for x in s) for s in d["shape_signature"]),
        block_id=None if d["block_id"] is None else BlockId(int(d["block_id"])),
        is_forward=bool(d["is_forward"]),
    )


def _trace_to_dict(trace: ProfilerTrace) -> dict[str, Any]:
    """Convert ``ProfilerTrace`` to a JSON-friendly dict.

    Note we don't use ``dataclasses.asdict`` for the top-level conversion
    because it would recurse into ``OpRecord`` (fine) but also leave us to
    re-handle every dict-keyed-by-NewType field anyway — explicit is faster
    to read and type-check.
    """
    payload: dict[str, Any] = {
        "trace_version": TRACE_VERSION,
        "op_order": [_op_record_to_dict(op) for op in trace.op_order],
        # dict[OpId, int|float] — JSON requires string keys.
        "intra_op_delta": {
            str(int(k)): int(v) for k, v in trace.intra_op_delta.items()
        },
        "inter_op_delta": {
            str(int(k)): int(v) for k, v in trace.inter_op_delta.items()
        },
        "activation_sizes": {
            str(int(k)): int(v) for k, v in trace.activation_sizes.items()
        },
        "model_state_bytes": int(trace.model_state_bytes),
        "pcie_h2d_bps": float(trace.pcie_h2d_bps),
        "pcie_d2h_bps": float(trace.pcie_d2h_bps),
        # nccl tables: dict[int, float], JSON requires string keys.
        "nccl_gather_s": {
            str(int(k)): float(v) for k, v in trace.nccl_gather_s.items()
        },
        "nccl_reduce_s": {
            str(int(k)): float(v) for k, v in trace.nccl_reduce_s.items()
        },
        "arch_hash": str(trace.arch_hash),
        "bs": int(trace.bs),
        "seq": int(trace.seq),
        "sku": str(trace.sku),
        "world": int(trace.world),
        "op_latencies": {str(int(k)): float(v) for k, v in trace.op_latencies.items()},
        "cpu_adam_bytes_per_sec": float(trace.cpu_adam_bytes_per_sec),
        "gpu_adam_bytes_per_sec": float(trace.gpu_adam_bytes_per_sec),
        "hooked_fwd_wall_s": float(trace.hooked_fwd_wall_s),
        "steady_fwd_wall_s": float(trace.steady_fwd_wall_s),
        "steady_bwd_wall_s": float(trace.steady_bwd_wall_s),
        "steady_fwd_peak_bytes": int(trace.steady_fwd_peak_bytes),
        "steady_fwd_block_peak_bytes": {
            str(int(k)): int(v) for k, v in trace.steady_fwd_block_peak_bytes.items()
        },
        # getattr-defensive against builds that haven't yet exposed newer fields.
        "steady_bwd_peak_bytes": int(getattr(trace, "steady_bwd_peak_bytes", 0)),
        "steady_bwd_block_peak_bytes": {
            str(int(k)): int(v)
            for k, v in getattr(trace, "steady_bwd_block_peak_bytes", {}).items()
        },
        "compute_rate_tflops": float(trace.compute_rate_tflops),
        "trainable_param_fraction": float(trace.trainable_param_fraction),
        "steady_fwd_chunked_wall_s": float(trace.steady_fwd_chunked_wall_s),
        "steady_bwd_chunked_wall_s": float(trace.steady_bwd_chunked_wall_s),
        "steady_step_overlap_s": float(trace.steady_step_overlap_s),
        "steady_phase2_peak_bytes": int(trace.steady_phase2_peak_bytes),
        "phase2_n_persist": int(trace.phase2_n_persist),
        "phase2_n_buffer": int(trace.phase2_n_buffer),
        "phase2_n_checkpoint": int(trace.phase2_n_checkpoint),
        "phase2_n_offload": int(getattr(trace, "phase2_n_offload", 0)),
        "phase2_per_block_recompute_s": float(trace.phase2_per_block_recompute_s),
        "phase2_iter_s": float(getattr(trace, "phase2_iter_s", 0.0)),
        "phase2_analytical_iter_s": float(
            getattr(trace, "phase2_analytical_iter_s", 0.0)
        ),
        "phase2_analytical_peak_bytes": int(
            getattr(trace, "phase2_analytical_peak_bytes", 0)
        ),
        "phase2_fwd_s": float(getattr(trace, "phase2_fwd_s", 0.0)),
        "phase2_bwd_s": float(getattr(trace, "phase2_bwd_s", 0.0)),
        "phase2_step_s": float(getattr(trace, "phase2_step_s", 0.0)),
        "phase2_analytical_fwd_s": float(
            getattr(trace, "phase2_analytical_fwd_s", 0.0)
        ),
        "phase2_analytical_bwd_s": float(
            getattr(trace, "phase2_analytical_bwd_s", 0.0)
        ),
        "phase2_analytical_step_s": float(
            getattr(trace, "phase2_analytical_step_s", 0.0)
        ),
        "phase2_per_comp_pred_iter_s": float(
            getattr(trace, "phase2_per_comp_pred_iter_s", 0.0)
        ),
        "block_tree_index": {
            str(int(k)): int(v) for k, v in trace.block_tree_index.items()
        },
        "hidden_size": int(getattr(trace, "hidden_size", 0)),
        "num_attention_heads": int(getattr(trace, "num_attention_heads", 0)),
        "intermediate_size": int(getattr(trace, "intermediate_size", 0)),
    }
    return payload


def _trace_from_dict(data: dict[str, Any]) -> ProfilerTrace:
    """Reconstruct ProfilerTrace from JSON-decoded dict; raises on shape corruption."""
    # Field-presence guard for optional fields the live dataclass may lack.
    _trace_field_names = {f.name for f in fields(ProfilerTrace)}
    extra: dict[str, Any] = {}
    if "phase2_n_offload" in _trace_field_names:
        extra["phase2_n_offload"] = int(data.get("phase2_n_offload", 0))
    if "steady_bwd_peak_bytes" in _trace_field_names:
        extra["steady_bwd_peak_bytes"] = int(data.get("steady_bwd_peak_bytes", 0))
    if "steady_bwd_block_peak_bytes" in _trace_field_names:
        extra["steady_bwd_block_peak_bytes"] = {
            BlockId(int(k)): int(v)
            for k, v in data.get("steady_bwd_block_peak_bytes", {}).items()
        }
    if "phase2_iter_s" in _trace_field_names:
        extra["phase2_iter_s"] = float(data.get("phase2_iter_s", 0.0))
    if "phase2_analytical_iter_s" in _trace_field_names:
        extra["phase2_analytical_iter_s"] = float(
            data.get("phase2_analytical_iter_s", 0.0)
        )
    if "phase2_analytical_peak_bytes" in _trace_field_names:
        extra["phase2_analytical_peak_bytes"] = int(
            data.get("phase2_analytical_peak_bytes", 0)
        )
    for _fname in (
        "phase2_fwd_s",
        "phase2_bwd_s",
        "phase2_step_s",
        "phase2_analytical_fwd_s",
        "phase2_analytical_bwd_s",
        "phase2_analytical_step_s",
    ):
        if _fname in _trace_field_names:
            extra[_fname] = float(data.get(_fname, 0.0))
    if "phase2_per_comp_pred_iter_s" in _trace_field_names:
        extra["phase2_per_comp_pred_iter_s"] = float(
            data.get("phase2_per_comp_pred_iter_s", 0.0)
        )
    for _arch_fname in ("hidden_size", "num_attention_heads", "intermediate_size"):
        if _arch_fname in _trace_field_names:
            extra[_arch_fname] = int(data.get(_arch_fname, 0))
    return ProfilerTrace(
        op_order=tuple(_op_record_from_dict(d) for d in data["op_order"]),
        intra_op_delta={
            OpId(int(k)): int(v) for k, v in data["intra_op_delta"].items()
        },
        inter_op_delta={
            OpId(int(k)): int(v) for k, v in data["inter_op_delta"].items()
        },
        activation_sizes={
            BlockId(int(k)): int(v) for k, v in data["activation_sizes"].items()
        },
        model_state_bytes=int(data["model_state_bytes"]),
        pcie_h2d_bps=float(data["pcie_h2d_bps"]),
        pcie_d2h_bps=float(data["pcie_d2h_bps"]),
        nccl_gather_s={int(k): float(v) for k, v in data["nccl_gather_s"].items()},
        nccl_reduce_s={int(k): float(v) for k, v in data["nccl_reduce_s"].items()},
        arch_hash=str(data["arch_hash"]),
        bs=int(data["bs"]),
        seq=int(data["seq"]),
        sku=str(data["sku"]),
        world=int(data["world"]),
        op_latencies={
            OpId(int(k)): float(v) for k, v in data.get("op_latencies", {}).items()
        },
        cpu_adam_bytes_per_sec=float(data.get("cpu_adam_bytes_per_sec", 0.0)),
        gpu_adam_bytes_per_sec=float(data.get("gpu_adam_bytes_per_sec", 0.0)),
        hooked_fwd_wall_s=float(data.get("hooked_fwd_wall_s", 0.0)),
        steady_fwd_wall_s=float(data.get("steady_fwd_wall_s", 0.0)),
        steady_bwd_wall_s=float(data.get("steady_bwd_wall_s", 0.0)),
        steady_fwd_peak_bytes=int(data.get("steady_fwd_peak_bytes", 0)),
        steady_fwd_block_peak_bytes={
            BlockId(int(k)): int(v)
            for k, v in data.get("steady_fwd_block_peak_bytes", {}).items()
        },
        compute_rate_tflops=float(data.get("compute_rate_tflops", 0.0)),
        trainable_param_fraction=float(data.get("trainable_param_fraction", 0.0)),
        steady_bwd_chunked_wall_s=float(data.get("steady_bwd_chunked_wall_s", 0.0)),
        steady_step_overlap_s=float(data.get("steady_step_overlap_s", 0.0)),
        steady_phase2_peak_bytes=int(data.get("steady_phase2_peak_bytes", 0)),
        phase2_n_persist=int(data.get("phase2_n_persist", 0)),
        phase2_n_buffer=int(data.get("phase2_n_buffer", 0)),
        phase2_n_checkpoint=int(data.get("phase2_n_checkpoint", 0)),
        phase2_per_block_recompute_s=float(
            data.get("phase2_per_block_recompute_s", 0.0)
        ),
        steady_fwd_chunked_wall_s=float(data.get("steady_fwd_chunked_wall_s", 0.0)),
        block_tree_index={
            BlockId(int(k)): int(v) for k, v in data.get("block_tree_index", {}).items()
        },
        **extra,
    )


def load_cached_trace(
    key: ProfilerCacheKey,
    cache_dir: str | os.PathLike[str] | None = None,
) -> ProfilerTrace | None:
    """Load a previously-saved trace, or None on cache miss."""
    path = _path_for(key, cache_dir)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        LOG.warning("profiler cache miss due to read error at %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        LOG.warning(
            "profiler cache at %s is not a dict (got %s); treating as miss.",
            path,
            type(data).__name__,
        )
        return None
    if data.get("trace_version") != TRACE_VERSION:
        LOG.info(
            "profiler cache at %s has trace_version=%s, current=%s; treating as miss.",
            path,
            data.get("trace_version"),
            TRACE_VERSION,
        )
        return None
    # Defense-in-depth identity check against stale-scheme files / hash collisions.
    payload_identity = (
        str(data.get("arch_hash", "")),
        int(data.get("bs", -1)) if isinstance(data.get("bs"), (int, float)) else -1,
        int(data.get("seq", -1)) if isinstance(data.get("seq"), (int, float)) else -1,
        str(data.get("sku", "")),
        int(data.get("world", -1))
        if isinstance(data.get("world"), (int, float))
        else -1,
    )
    expected_identity = (key.arch_hash, key.bs, key.seq, key.sku, key.world)
    if payload_identity != expected_identity:
        LOG.warning(
            "profiler cache at %s identifies %s but expected %s; treating as miss.",
            path,
            payload_identity,
            expected_identity,
        )
        return None
    try:
        return _trace_from_dict(data)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        # AttributeError covers nested shape corruption (e.g. dict→list).
        LOG.warning(
            "profiler cache at %s failed deserialization (%s); treating as miss.",
            path,
            exc,
        )
        return None


def save_cached_trace(
    key: ProfilerCacheKey,
    trace: ProfilerTrace,
    cache_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Persist ``trace`` under ``key``; returns the on-disk path."""
    root = _cache_root(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = _path_for(key, cache_dir)
    data = _trace_to_dict(trace)
    # Per-rank unique temp so concurrent writes don't clobber; os.replace is atomic.
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f"{path.stem}.",
        suffix=".tmp",
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, separators=(",", ":"))
        os.replace(tmp, path)
    finally:
        # Removes the partial JSON on failure; no-op after successful os.replace.
        tmp.unlink(missing_ok=True)
    LOG.debug("saved profiler trace to %s", path)
    return path


__all__ = [
    "ProfilerCacheKey",
    "load_cached_trace",
    "save_cached_trace",
]
