"""On-disk cache for ProfilerTrace, keyed by (arch_hash, bs, seq, sku, world).

JSON serialization (not pickle) — pickle.load() is a remote-code-execution
sink if any attacker can drop a file under ``$XDG_CACHE_HOME/protrain/profiler``,
and the trace is pure data anyway. JSON has cheap, verifiable round-trip
semantics here; the only fixups required on load are re-tupling sequence
fields, re-typing ``BlockId`` keys (JSON dict keys are always strings), and
reconstructing the ``BlockMode`` str-enum.
"""

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

# Bump when the ProfilerTrace schema changes in a way that invalidates existing
# cached traces. Version 2 adds per-op wall-clock latencies (``op_latencies``);
# version 3 adds measured Adam throughputs (``cpu_adam_bytes_per_sec`` /
# ``gpu_adam_bytes_per_sec``) — traces from v2 have 0.0 for those fields, so
# the runtime cost model would fall back to the hardcoded prior. Bumping the
# version forces a re-profile rather than silently degrading accuracy.
# Version 4 adds hook-dispatch calibration fields (``hooked_fwd_wall_s`` /
# ``steady_fwd_wall_s`` / ``steady_bwd_wall_s``) that the cost model consumes
# to scale the hooked per-op latencies down to a steady-state prior. v3
# traces default those fields to 0.0 which would make the cost model fall
# back to identity scale and regress 7B runtime error to its pre-calibration
# level; bumping forces a fresh trace.
# Version 5 adds an aggregate ``steady_fwd_peak_bytes`` cap used by the
# memory cost model when the searcher picks all-NONE.
# Version 6 adds per-block peaks (``steady_fwd_block_peak_bytes``) captured
# during the hook-less steady forward via lightweight block-level hooks.
# Unlike the v5 aggregate — which only applies when n_checkpoint=0 &&
# n_swap=0 — the per-block max bounds the forward peak for any fractional-
# NONE config, tightening over-prediction across the search space. v5
# traces default the per-block dict to empty, so the cost model falls back
# to the aggregate-only cap (identical v5 behavior); bumping forces a fresh
# trace so the cap takes effect.
# Version 7 changes the steady-state measurement methodology from a single
# iteration to a 4-iter hot loop (2 warmup + 2 measured, median of measured)
# and adds a best-effort steady_bwd_wall_s in the same loop. The recorded
# fields are unchanged but the *values* shift (single-iter carried allocator-
# settle cost the multi-iter median eliminates), so the cost model's measured
# bwd/fwd ratio path requires a fresh trace under the new methodology.
# Version 8 makes ``world`` and the NCCL collective tables real for
# world_size > 1: ``measure_nccl(world_size>1)`` now actually runs
# all_gather_into_tensor / reduce_scatter_tensor sweeps over a payload-size
# grid instead of raising NotImplementedError, and ``run_trace`` plumbs
# ``cfg.world_size`` (or auto-detects from the live process group) into
# both the trace's ``world`` field and the per-payload tables. Single-rank
# traces are unaffected (collective tables stay empty); multi-rank traces
# captured under v7 had ``world=1`` hard-coded and must be re-run.
# Version 9 folds ``requires_grad`` into the arch_hash so that toggling
# freeze-layer config invalidates the cache. Previously a v8 trace
# captured under one freezing pattern would replay against a different
# freezing pattern with the same arch, returning stale
# ``trainable_param_fraction`` / ``model_state_bytes`` and steering the
# cost model into the wrong bwd/fwd-ratio fallback. v8 traces remain on
# disk but never look up under v9 keys.
# Version 10 adds phase-2 chunked-runtime backward fields:
# ``steady_bwd_chunked_wall_s``, ``steady_step_overlap_s``,
# ``phase2_n_checkpoint``, ``phase2_per_block_recompute_s``. These are
# populated by the bootstrap-then-measure loop in
# ``protrain_model_wrapper`` and consumed by ``cost/runtime.py`` to
# translate a measured chunked backward to any candidate ``block_map``
# the search evaluates. v9 traces lack these fields and would steer
# the cost model into the v8 fallback path; bumping invalidates them
# so the next run captures a real chunked backward measurement.
# Version 11 adds the phase-2 chunked-runtime FORWARD field:
# ``steady_fwd_chunked_wall_s``. Same plumbing as v10 — the
# bootstrap-then-measure loop in ``protrain_model_wrapper`` now also
# times the forward window, and ``cost/runtime._fwd_compute_time_from_trace``
# uses the measurement directly as the forward total when populated
# (overrides the per-op-latency-sum + hook-scale + roofline cap path).
# Closes the forward half of the residual over-prediction left after
# v10 backward calibration; on 7B-LoRA + 3090 this drops same-SKU
# runtime error into the high-20% range before the matching backward
# chunked-wall bypass. v10 traces have ``steady_fwd_chunked_wall_s`` at
# 0.0 which would silently force the cost model back to the v10 forward
# path; bumping forces a fresh trace so the new measurement is captured
# and consumed.
# Version 12 invalidates v11 traces after checkpoint recompute was wired
# to re-gather block chunks before replay. v11 phase-2 backward timings
# were captured without that replay-time gather cost, so they
# under-predict all-CKPT offload configs once the runtime is actually
# correct.
# Version 13 changes the phase-2 bootstrap from the initial search's
# often-high ``n_persist`` pick to a conservative low-persistence
# all-CKPT config. v12 traces under-count replay gathers for the
# low-persistence configs selected after calibration.
# Version 14 records ``steady_phase2_peak_bytes`` plus the phase-2
# bootstrap cfg tuple, allowing the wrapper to calibrate peak from the
# same measured chunked run when the final config matches.
# Version 15 stores the EFFECTIVE phase-2 cfg after runtime construction
# (including non-block chunk pins), not the raw bootstrap search tuple.
# Version 16 adds the persisted ``block_tree_index`` field — captured at
# trace-construction from ``discover_blocks(model)`` so the cost model
# no longer has to parse ``OpRecord.module_path`` prefixes (``encoder.``
# / ``decoder.``) to recover tree membership. The string-prefix path
# stays as a fallback for degenerate test traces but cached profiles
# carry the authoritative map.
# Version 17 switches the on-disk format from pickle to JSON. Pickle
# is a remote-code-execution sink (``pickle.load`` calls arbitrary
# constructors during deserialization) and the cache directory is a
# local-attacker writable target; JSON has none of those semantics.
# v16 ``.pkl`` files remain on disk but are never looked up under the
# v17 ``.json`` extension — the cache is local-only and a re-profile
# is cheap, so the migration policy is "ignore + retrace".
# Version 18 adds ``phase2_n_offload`` to the persisted phase-2 bootstrap
# cfg tuple. Option B's search space includes the n_offload axis (see
# ``exhaustive.py`` / ``block/layout_rules.py``) and the bootstrap
# captures it in ``boot_result.cfg.n_offload``, but v17 cached only
# (persist, buffer, checkpoint). Two configs that differ only in the
# offload axis would therefore share a cached measurement and the
# wrapper's ``phase2_matches_cfg`` predicate would mis-calibrate the
# cost model (in particular ``steady_phase2_peak_bytes`` and the
# chunked-bwd base term). Bumping forces a fresh trace so the offload
# count is recorded under the matching cfg. ``ProfilerTrace`` may not
# yet carry the field; the (de)serializers fall back to 0 via getattr
# / fields-introspection so a v18 payload round-trips cleanly either
# way and the bump alone invalidates v17 entries that lacked the axis.
# Version 19 engages backward profiling end-to-end (paper §3.2 / App A.2).
# The wrapper now passes ``include_backward=True`` to ``run_trace`` and
# the steady-state hot loop captures the BACKWARD peak alongside the
# forward peak: ``steady_bwd_peak_bytes`` is the cumulative
# ``max_memory_allocated`` across the hook-less backward pass, and
# ``steady_bwd_block_peak_bytes`` carries the per-block bwd peaks via
# ``register_full_backward_hook`` (mirroring the per-block fwd capture).
# v18 traces have these at 0 / empty so the cost model would silently
# regress to its analytical bwd estimate — bumping the schema forces a
# re-profile so the measurement is captured and consumed.
TRACE_VERSION = 19


@dataclass(frozen=True)
class ProfilerCacheKey:
    """Identity of a cached trace (§7 re-profile trigger).

    Not defined in ``types.py`` by design — cache keys are an implementation
    detail of this subpackage and shouldn't leak into the public plugin API.
    """

    arch_hash: str
    bs: int
    seq: int
    sku: str
    world: int

    def fingerprint(self) -> str:
        """Deterministic 64-char sha256 hex digest used as the on-disk filename.

        The ``TRACE_VERSION`` prefix ensures a schema bump invalidates all prior
        cache entries — old files stay on disk but are never looked up.
        """
        raw = f"v{TRACE_VERSION}|{self.arch_hash}|{self.bs}|{self.seq}|{self.sku}|{self.world}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_root() -> Path:
    """Resolve ``$XDG_CACHE_HOME/protrain/profiler`` or ``~/.cache/protrain/profiler``."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / _CACHE_SUBDIR


def _path_for(key: ProfilerCacheKey) -> Path:
    return _cache_root() / f"{key.fingerprint()}.json"


# ---------------------------------------------------------------------------
# JSON (de)serialization — ProfilerTrace is pure data so this is a small
# fixup pass over ``dataclasses.asdict`` output. The contract:
#   * tuple fields → list on write, retuple on load
#   * dict[BlockId, ...] → str-keyed dict on write (JSON), int-keyed
#     ``BlockId`` dict on load
#   * dict[OpId, ...] → same treatment as BlockId
#   * BlockMode enum → string ``.value`` on write, ``BlockMode(s)`` on load
#   * trace_version is embedded in the payload so loaders can reject
#     mismatched versions (the filename hashes the version too, but a
#     payload-level check is a defense-in-depth tripwire if the hash
#     scheme ever changes).
# ---------------------------------------------------------------------------


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
        # ``steady_bwd_peak_bytes`` / ``steady_bwd_block_peak_bytes``
        # (TRACE_VERSION 19). ``getattr`` keeps this defensive against
        # ``ProfilerTrace`` builds that haven't yet exposed the field —
        # the bump still invalidates v18 traces lacking the measurement.
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
        # ``phase2_n_offload`` (TRACE_VERSION 18) joins the persisted phase-2
        # cfg tuple. ``getattr`` keeps this defensive against ``ProfilerTrace``
        # builds that haven't yet exposed the field — the bump still
        # invalidates v17 traces lacking the offload axis.
        "phase2_n_offload": int(getattr(trace, "phase2_n_offload", 0)),
        "phase2_per_block_recompute_s": float(trace.phase2_per_block_recompute_s),
        "block_tree_index": {
            str(int(k)): int(v) for k, v in trace.block_tree_index.items()
        },
    }
    return payload


def _trace_from_dict(data: dict[str, Any]) -> ProfilerTrace:
    """Reconstruct a ``ProfilerTrace`` from its JSON-decoded dict.

    Raises ``AttributeError`` / ``KeyError`` / ``ValueError`` / ``TypeError``
    if required fields are missing or malformed (including nested payload
    shape corruption such as ``"intra_op_delta": []`` where ``.items()`` is
    called on a non-mapping); callers treat that as a cache miss.
    """
    # ``phase2_n_offload`` (TRACE_VERSION 18) joined the phase-2 cfg tuple.
    # ``steady_bwd_peak_bytes`` / ``steady_bwd_block_peak_bytes``
    # (TRACE_VERSION 19) joined the bwd-aware peak measurements. Pass
    # them as kwargs only when the live ``ProfilerTrace`` dataclass
    # actually exposes the fields — older builds in the same tree (e.g.
    # test fixtures pinned to a prior schema) would otherwise raise
    # TypeError on the unexpected kwarg and turn every fresh-version
    # hit into a cache miss.
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


def load_cached_trace(key: ProfilerCacheKey) -> ProfilerTrace | None:
    """Load a previously-saved trace, or ``None`` if the key misses."""
    path = _path_for(key)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
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
    try:
        return _trace_from_dict(data)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        # ``AttributeError`` covers nested payload shape corruption — e.g. a
        # malformed ``"intra_op_delta": []`` makes ``_trace_from_dict`` call
        # ``.items()`` on a list, which would otherwise escape and abort
        # startup instead of degrading to a clean cache miss.
        LOG.warning(
            "profiler cache at %s failed deserialization (%s); treating as miss.",
            path,
            exc,
        )
        return None


def save_cached_trace(key: ProfilerCacheKey, trace: ProfilerTrace) -> Path:
    """Persist ``trace`` under ``key``. Returns the on-disk path."""
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    path = _path_for(key)
    data = _trace_to_dict(trace)
    # Per-rank unique temp via mkstemp(dir=path.parent) so two ranks racing
    # on the same key can't clobber each other's in-flight writes; os.replace
    # then promotes whichever finished last to the final filename atomically.
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f"{path.stem}.",
        suffix=".tmp",
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            # Compact separators keep the file size close to the pickle
            # output; trace files are O(MB) on real models so the savings
            # over the default ", " / ": " are non-trivial.
            json.dump(data, fh, separators=(",", ":"))
        os.replace(tmp, path)
    finally:
        # Cleanup is a no-op on the success path (replace already moved tmp);
        # on failure it removes the partial JSON. ``missing_ok=True``
        # covers both cases.
        tmp.unlink(missing_ok=True)
    LOG.debug("saved profiler trace to %s", path)
    return path


__all__ = [
    "ProfilerCacheKey",
    "load_cached_trace",
    "save_cached_trace",
]
