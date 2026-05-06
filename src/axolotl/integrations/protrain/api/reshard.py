"""Core reshard logic for ProTrain Mode-C optimizer state.

Pure-Python tensor algebra over a saved ``protrain_optim/`` directory:
takes the per-rank shard files written at ``world_size=src_world`` and
emits a fresh directory at ``world_size=target_world``. No GPUs, no
``torch.distributed`` — only ``torch.load`` / ``torch.save`` /
``torch.cat`` / contiguous slicing on CPU.

This module is the single source of truth for the reshard arithmetic.
Two callers consume it:

* The offline CLI ``scripts/protrain/reshard_optim.py`` — a thin
  argparse wrapper around :func:`reshard_mode_c_shards`. The CLI loads
  this module via file-path-based ``importlib`` so it can run on a
  host that doesn't have the full axolotl import chain (transformers,
  etc.) — useful for "reshard a checkpoint on a CPU box, then move it
  to the training node" workflows.
* The online load path
  (:func:`axolotl.integrations.protrain.api.checkpoint._load_protrain_optim_dir`)
  when the user opts in via ``protrain_allow_online_reshard=True``.
  Rank-0 calls :func:`reshard_mode_c_shards` into a temp dir, all
  ranks barrier, and the load proceeds against the temp dir as if it
  were a natively-saved-at-N2 checkpoint.

Per-region resharding maths (paper's ZeRO-3 sharding rule):

* Each region holds ``region_bytes`` of valid state plus padding to
  ``region_bytes_padded = ceil(region_bytes / lcm(elem_size, W)) *
  lcm(elem_size, W)`` so ``shard_bytes = region_bytes_padded / W`` is
  a clean element-aligned slice. The valid prefix length
  ``region_bytes / element_size`` is independent of W.
* For each region, concatenate the N1 saved per-rank ``exp_avg`` (and
  ``exp_avg_sq``) tensors → flat tensor of length
  ``region_bytes_padded_old / elem_size``.
* The first ``region_bytes / elem_size`` elements are valid. Trailing
  bytes are padding; on a clean save they are zero (the materialize
  pad-zero plus zero gradient on padding bytes means Adam never
  updates those positions).
* Build a fresh tensor of length ``region_bytes_padded_new /
  elem_size``, copy the valid prefix, zero-pad the rest, and split
  into N2 contiguous slices of length ``shard_bytes_new / elem_size``
  each. Slice ``r2`` becomes the new rank ``r2``'s state for that
  region.
* The Adam ``step`` scalar is rank-replicated; we copy it as-is.

Constraints mirrored from ``api/checkpoint.py``: file-naming regex,
schema constants, dtype-name lookup. Any drift between this module's
constants and the checkpoint module's would silently break round-trip
loads — the loader recomputes the layout signature against the new
``world_size`` using the api module's
:func:`_layout_signature_from_fingerprint`, so the formula here must
stay byte-compatible with the api version. Tested via the offline +
online reshard round-trip tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from typing import Any

import torch

# ---- Constants mirrored from api/checkpoint.py ----------------------------
# We deliberately avoid importing the api module so the offline CLI's
# importlib loader can pull this file in without dragging in the heavy
# axolotl import chain (transformers, etc.). Drift between these
# constants and the api module's would silently break round-trip loads —
# guarded by the offline + online reshard round-trip tests.

METADATA_FILENAME = "metadata.json"
GPU_OPTIM_FILENAME = "gpu_optim.pt"
CPU_OPTIM_DIRNAME = "cpu_optim"
SCHEMA_FORMAT_VERSION = 2
SAVE_MODE_SHARDED = "sharded"
CHUNK_SHARD_FILE_RE = re.compile(r"^chunk_(\d+)_rank_(\d+)\.pt$")

_DTYPE_NAME_TO_TORCH: dict[str, torch.dtype] = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float": torch.float32,
    "torch.half": torch.float16,
    "torch.double": torch.float64,
}


# ---- Layout signature ------------------------------------------------------


def _layout_signature_from_fingerprint(fingerprint: dict[str, Any]) -> str:
    """SHA-256 over a layout fingerprint dict.

    Mirrors :func:`api.checkpoint._layout_signature_from_fingerprint`.
    Re-implemented here so this module does not pull in the heavyweight
    api module's transitive imports. The two implementations must stay
    byte-compatible — the loader recomputes the expected signature using
    the api version, so any drift would trip the layout-signature check.
    """
    payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---- Per-region reshard ----------------------------------------------------


def _padded_region_bytes(region_bytes: int, elem_size: int, world_size: int) -> int:
    """Pad ``region_bytes`` so each rank owns a whole number of elements.

    Computes the element count first (``ceil(region_bytes / elem_size)``),
    rounds it up to a multiple of ``world_size`` so per-rank shards are
    element-aligned, then converts back to bytes. The previous formulation
    padded directly in bytes via ``lcm(elem_size, world_size)``, which can
    produce sizes that don't align to whole elements per rank when
    ``gcd(elem_size, world_size) > 1`` (e.g. elem_size=4, world_size=8:
    pad_unit=8 bytes splits into 1-byte shards, half an element each).

    Mirrors the formula in ``ChunkManager.materialize_offload`` (chunk/
    manager.py around the ``region_plans`` block). Must stay
    byte-compatible — the loader's region-layout match step compares
    against the runtime's ``region_bytes_padded`` and any drift would
    trip the regions_per_chunk validation.
    """
    elem_count = (region_bytes + elem_size - 1) // elem_size
    padded_elems = ((elem_count + world_size - 1) // world_size) * world_size
    return padded_elems * elem_size


def _reshard_region_state(
    per_rank_tensors: list[torch.Tensor],
    *,
    region_bytes: int,
    elem_size: int,
    src_world: int,
    dst_world: int,
    region_bytes_padded_old: int | None = None,
    region_bytes_padded_new: int | None = None,
) -> list[torch.Tensor]:
    """Reshard one region's per-rank state tensor (e.g. ``exp_avg``) from
    ``src_world`` ranks to ``dst_world`` ranks.

    Inputs
    ------
    per_rank_tensors:
        List of length ``src_world`` of 1-D tensors, all with the same
        dtype and length ``shard_bytes_old / elem_size``.
    region_bytes:
        Un-padded valid bytes of the region (constant across world
        sizes).
    elem_size:
        ``dtype.itemsize`` for the region.
    region_bytes_padded_old / region_bytes_padded_new:
        If supplied (typically from the saved metadata), use these
        directly instead of recomputing — guards against any drift
        between the script's pad formula and the runtime's.

    Output
    ------
    List of length ``dst_world`` of 1-D tensors, all with the same dtype
    as the inputs and length ``shard_bytes_new / elem_size``.
    """
    if len(per_rank_tensors) != src_world:
        raise RuntimeError(
            f"reshard: expected {src_world} per-rank tensors, got "
            f"{len(per_rank_tensors)}"
        )
    dtype = per_rank_tensors[0].dtype
    for t in per_rank_tensors:
        if t.dtype != dtype:
            raise RuntimeError(
                f"reshard: per-rank tensors have inconsistent dtypes "
                f"({dtype} vs {t.dtype}) — refusing to mix"
            )

    if region_bytes_padded_old is None:
        region_bytes_padded_old = _padded_region_bytes(
            region_bytes, elem_size, src_world
        )
    if region_bytes_padded_new is None:
        region_bytes_padded_new = _padded_region_bytes(
            region_bytes, elem_size, dst_world
        )

    # All byte→numel conversions below use integer division. If any of
    # these aren't exact multiples we'd silently truncate elements,
    # which corrupts shards in subtle ways. Validate up-front and raise
    # with the offending values so the failure is actionable.
    if region_bytes % elem_size != 0:
        raise RuntimeError(
            f"reshard: region_bytes={region_bytes} is not divisible by "
            f"elem_size={elem_size}"
        )
    if region_bytes_padded_old % elem_size != 0:
        raise RuntimeError(
            f"reshard: region_bytes_padded_old={region_bytes_padded_old} "
            f"is not divisible by elem_size={elem_size}"
        )
    if region_bytes_padded_new % elem_size != 0:
        raise RuntimeError(
            f"reshard: region_bytes_padded_new={region_bytes_padded_new} "
            f"is not divisible by elem_size={elem_size}"
        )
    if region_bytes_padded_old % src_world != 0:
        raise RuntimeError(
            f"reshard: region_bytes_padded_old={region_bytes_padded_old} "
            f"is not divisible by src_world={src_world}; cannot split "
            f"into equal per-rank shards"
        )
    if region_bytes_padded_new % dst_world != 0:
        raise RuntimeError(
            f"reshard: region_bytes_padded_new={region_bytes_padded_new} "
            f"is not divisible by dst_world={dst_world}; cannot split "
            f"into equal per-rank shards"
        )

    expected_old_shard_numel = (region_bytes_padded_old // src_world) // elem_size
    for r, t in enumerate(per_rank_tensors):
        if t.numel() != expected_old_shard_numel:
            raise RuntimeError(
                f"reshard: per-rank tensor {r} has numel={t.numel()}, "
                f"expected {expected_old_shard_numel} "
                f"(region_bytes_padded={region_bytes_padded_old}, "
                f"elem_size={elem_size}, src_world={src_world})"
            )

    # Concatenate to the full padded region tensor (length
    # region_bytes_padded_old / elem_size), then carry only the valid
    # prefix forward — Adam never reads/writes padding bytes for a clean
    # run (chunk/manager.py:802 zero-inits cpu_region_grad; materialize
    # zero-pads region_scratch). Freeing full_old before allocating
    # full_new halves peak working RAM per region.
    full_old = torch.cat(per_rank_tensors, dim=0).contiguous()
    valid_numel = region_bytes // elem_size
    valid_prefix = full_old[:valid_numel].clone()
    del full_old
    new_padded_numel = region_bytes_padded_new // elem_size
    full_new = torch.zeros(new_padded_numel, dtype=dtype)
    full_new[:valid_numel] = valid_prefix
    del valid_prefix

    new_shard_numel = (region_bytes_padded_new // dst_world) // elem_size
    out: list[torch.Tensor] = []
    for r in range(dst_world):
        start = r * new_shard_numel
        end = start + new_shard_numel
        # Clone so each output slice owns its own storage (defensive —
        # the slices end up serialized via torch.save which deep-copies,
        # but consumer code may inspect intermediates in tests).
        out.append(full_new[start:end].clone())
    return out


# ---- Driver ---------------------------------------------------------------


def _read_metadata(src_dir: str) -> dict[str, Any]:
    meta_path = os.path.join(src_dir, METADATA_FILENAME)
    if not os.path.isfile(meta_path):
        raise RuntimeError(f"reshard: missing metadata at {meta_path!r}")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _validate_src_metadata(meta: dict[str, Any]) -> None:
    fmt = int(meta.get("format_version", 0))
    if fmt != SCHEMA_FORMAT_VERSION:
        raise RuntimeError(
            f"reshard: source format_version={fmt}, expected "
            f"{SCHEMA_FORMAT_VERSION}. Only Phase-2 v2 saves are supported."
        )
    save_mode = meta.get("protrain_save_mode")
    if save_mode != SAVE_MODE_SHARDED:
        raise RuntimeError(
            f"reshard: source save_mode={save_mode!r}, expected "
            f"{SAVE_MODE_SHARDED!r}. Mode-B replicated saves do not need "
            "resharding (the load path tolerates world_size drift "
            "natively — see CHECKPOINT_DESIGN_PHASE2.md §4.1 Option B)."
        )
    if "regions_per_chunk" not in meta:
        raise RuntimeError(
            "reshard: source metadata missing 'regions_per_chunk'. The "
            "save predates Mode-C support or the file is corrupt."
        )
    if "layout_fingerprint" not in meta:
        raise RuntimeError(
            "reshard: source metadata missing 'layout_fingerprint'. The "
            "save predates the offline reshard support — re-save under a "
            "newer ProTrain build to capture the raw layout fields."
        )
    try:
        src_world = int(meta["protrain_world_size"])
    except Exception as exc:
        raise RuntimeError(
            "reshard: source metadata missing valid 'protrain_world_size'."
        ) from exc
    if src_world < 1:
        raise RuntimeError(
            f"reshard: invalid protrain_world_size={src_world}; expected >= 1."
        )


def _scan_src_chunks(src_dir: str, src_world: int) -> dict[int, list[str]]:
    """Return ``{chunk_id: [path_for_rank0, path_for_rank1, ...]}``."""
    cpu_dir = os.path.join(src_dir, CPU_OPTIM_DIRNAME)
    if not os.path.isdir(cpu_dir):
        return {}
    by_chunk: dict[int, dict[int, str]] = {}
    for name in sorted(os.listdir(cpu_dir)):
        m = CHUNK_SHARD_FILE_RE.match(name)
        if m is None:
            raise RuntimeError(
                f"reshard: unexpected file {name!r} in {cpu_dir!r} — "
                "Mode-C cpu_optim/ must contain only chunk_<N>_rank_<R>.pt"
            )
        cid = int(m.group(1))
        rank = int(m.group(2))
        if rank < 0 or rank >= src_world:
            raise RuntimeError(
                f"reshard: file {name!r} rank ordinal {rank} outside "
                f"[0, {src_world}) — corrupt source dir."
            )
        by_chunk.setdefault(cid, {})[rank] = os.path.join(cpu_dir, name)

    out: dict[int, list[str]] = {}
    for cid, by_rank in by_chunk.items():
        if set(by_rank.keys()) != set(range(src_world)):
            missing = set(range(src_world)) - set(by_rank.keys())
            raise RuntimeError(
                f"reshard: chunk {cid} missing per-rank shards for "
                f"ranks {sorted(missing)}"
            )
        out[cid] = [by_rank[r] for r in range(src_world)]
    return out


def reshard_mode_c_shards(
    src_dir: str,
    dst_dir: str,
    target_world_size: int,
    *,
    log_fn=None,
) -> None:
    """Top-level driver. Reads ``src_dir``, writes ``dst_dir`` at
    ``target_world_size`` ranks.

    Writes a fresh output tree at ``dst_dir``. The function refuses to
    run when ``dst_dir`` already exists and is non-empty, so callers
    must provide an empty or nonexistent destination directory.

    Parameters
    ----------
    src_dir, dst_dir:
        Filesystem paths. ``src_dir`` must contain a Mode-C save
        (``protrain_save_mode == "sharded"`` plus
        ``layout_fingerprint`` in metadata.json).
    target_world_size:
        Target world_size N2; must be >= 1.
    log_fn:
        Optional ``Callable[[str], None]`` used for the two
        informational log lines (default: print to stderr). The online
        load path passes a logger-bound logger so the messages thread
        through axolotl's logging setup.
    """
    if target_world_size < 1:
        raise ValueError(f"target_world_size must be >= 1 (got {target_world_size})")

    if log_fn is None:
        log_fn = lambda msg: print(msg, file=sys.stderr)  # noqa: E731

    meta = _read_metadata(src_dir)
    _validate_src_metadata(meta)

    src_world = int(meta["protrain_world_size"])
    if src_world == target_world_size:
        # Nothing to do; just copy. We still emit a fresh dst_dir for
        # consistency with the "always produce a complete dir" contract.
        log_fn(
            f"reshard: src_world == target_world == {src_world}; "
            "copying source directory verbatim"
        )
        if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
            raise RuntimeError("reshard: dst_dir must differ from src_dir")
        if os.path.isdir(dst_dir) and os.listdir(dst_dir):
            raise RuntimeError(
                f"reshard: refusing to overwrite non-empty dst_dir {dst_dir!r}"
            )
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        log_fn(f"reshard: copied {src_dir!r} to {dst_dir!r} (no reshape needed)")
        return

    log_fn(
        f"reshard: src={src_dir!r} dst={dst_dir!r} "
        f"src_world={src_world} target_world={target_world_size}"
    )

    if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
        raise RuntimeError("reshard: dst_dir must differ from src_dir")
    if os.path.isdir(dst_dir) and os.listdir(dst_dir):
        raise RuntimeError(
            f"reshard: refusing to overwrite non-empty dst_dir {dst_dir!r}"
        )
    os.makedirs(dst_dir, exist_ok=True)
    cpu_dst_dir = os.path.join(dst_dir, CPU_OPTIM_DIRNAME)

    # Replicated artifacts: gpu_optim.pt is rank-independent (same on
    # every rank in Mode-C), so just copy it.
    src_gpu = os.path.join(src_dir, GPU_OPTIM_FILENAME)
    if os.path.isfile(src_gpu):
        shutil.copyfile(src_gpu, os.path.join(dst_dir, GPU_OPTIM_FILENAME))

    saved_regions: dict[str, list[dict[str, Any]]] = meta["regions_per_chunk"]

    # Build fresh regions_per_chunk for the target world_size — only
    # region_bytes_padded and shard_bytes change with world_size.
    new_regions: dict[str, list[dict[str, Any]]] = {}
    for cid_str, regs in saved_regions.items():
        new_list: list[dict[str, Any]] = []
        for r in regs:
            elem_size_int = _DTYPE_NAME_TO_TORCH[r["dtype"]].itemsize
            region_bytes = int(r["region_bytes"])
            new_padded = _padded_region_bytes(
                region_bytes, elem_size_int, target_world_size
            )
            new_shard_bytes = new_padded // target_world_size
            new_list.append(
                {
                    "chunk_offset": int(r["chunk_offset"]),
                    "region_bytes": region_bytes,
                    "region_bytes_padded": int(new_padded),
                    "shard_bytes": int(new_shard_bytes),
                    "dtype": r["dtype"],
                }
            )
        new_regions[cid_str] = new_list

    # Reshard each chunk's per-rank state files.
    chunk_paths = _scan_src_chunks(src_dir, src_world)
    if chunk_paths:
        os.makedirs(cpu_dst_dir, exist_ok=True)

    # Cross-check chunk ids in metadata and on disk.
    saved_cids = set(int(c) for c in saved_regions.keys())
    disk_cids = set(chunk_paths.keys())
    if saved_cids != disk_cids:
        raise RuntimeError(
            "reshard: regions_per_chunk chunk-ids "
            f"{sorted(saved_cids)} disagree with on-disk shard chunk-ids "
            f"{sorted(disk_cids)}"
        )

    for cid in sorted(chunk_paths.keys()):
        per_rank_paths = chunk_paths[cid]
        per_rank_state_dicts = [
            torch.load(p, map_location="cpu", weights_only=True) for p in per_rank_paths
        ]
        regs = saved_regions[str(cid)]

        # Validate state shape consistency: every per-rank state_dict
        # must have one ``state[i]`` entry per region, in order.
        for r_idx, sd in enumerate(per_rank_state_dicts):
            if "state" not in sd or "param_groups" not in sd:
                raise RuntimeError(
                    f"reshard: chunk {cid} rank {r_idx} state_dict missing "
                    "'state' or 'param_groups' key"
                )
            if set(sd["state"].keys()) != set(range(len(regs))):
                raise RuntimeError(
                    f"reshard: chunk {cid} rank {r_idx} state has keys "
                    f"{sorted(sd['state'].keys())}, expected "
                    f"{list(range(len(regs)))} (one per region)"
                )

        # Build new per-rank state_dicts. Reuse rank-0's param_groups
        # (it's rank-independent — defaults + the [0..N-1] params list).
        # ``step`` is also rank-replicated; copy from rank-0.
        new_per_rank_states: list[dict[int, dict[str, Any]]] = [
            {} for _ in range(target_world_size)
        ]
        for region_idx, region_meta in enumerate(regs):
            region_bytes = int(region_meta["region_bytes"])
            elem_size_int = _DTYPE_NAME_TO_TORCH[region_meta["dtype"]].itemsize
            saved_padded_old = int(region_meta["region_bytes_padded"])
            new_padded = new_regions[str(cid)][region_idx]["region_bytes_padded"]

            for state_key in ("exp_avg", "exp_avg_sq"):
                per_rank_inputs = [
                    sd["state"][region_idx][state_key] for sd in per_rank_state_dicts
                ]
                # Defensive: ensure all are 1-D (they should be — the
                # shard_param's flat storage view).
                per_rank_inputs = [t.flatten() for t in per_rank_inputs]
                new_slices = _reshard_region_state(
                    per_rank_inputs,
                    region_bytes=region_bytes,
                    elem_size=elem_size_int,
                    src_world=src_world,
                    dst_world=target_world_size,
                    region_bytes_padded_old=saved_padded_old,
                    region_bytes_padded_new=int(new_padded),
                )
                for r2, slice_ in enumerate(new_slices):
                    new_per_rank_states[r2].setdefault(region_idx, {})[state_key] = (
                        slice_
                    )

            # Replicate ``step`` and any other per-region scalars from
            # rank-0 (they're guaranteed identical across saving ranks
            # since DeepSpeedCPUAdam steps in lockstep within a chunk).
            for k, v in per_rank_state_dicts[0]["state"][region_idx].items():
                if k in ("exp_avg", "exp_avg_sq"):
                    continue
                for r2 in range(target_world_size):
                    # Clone tensors per-rank so mutations don't propagate.
                    val = v.clone() if isinstance(v, torch.Tensor) else v
                    new_per_rank_states[r2].setdefault(region_idx, {})[k] = val

        param_groups = per_rank_state_dicts[0]["param_groups"]

        # Write new per-rank shard files.
        for r2 in range(target_world_size):
            new_sd = {
                "state": new_per_rank_states[r2],
                "param_groups": param_groups,
            }
            out_path = os.path.join(cpu_dst_dir, f"chunk_{cid}_rank_{r2}.pt")
            torch.save(new_sd, out_path)

    # Recompute layout_fingerprint with the new world_size and the
    # corresponding signature.
    fp = dict(meta["layout_fingerprint"])
    fp["world_size"] = int(target_world_size)
    new_signature = _layout_signature_from_fingerprint(fp)

    new_meta = dict(meta)
    new_meta["protrain_world_size"] = int(target_world_size)
    new_meta["layout_fingerprint"] = fp
    new_meta["protrain_layout_signature"] = new_signature
    new_meta["regions_per_chunk"] = new_regions
    # Mark the source world for forensic-friendliness; the loader
    # ignores unknown keys.
    new_meta["resharded_from_world_size"] = int(src_world)
    # ``saving_rank`` is only meaningful for the original save; preserve it.

    with open(os.path.join(dst_dir, METADATA_FILENAME), "w", encoding="utf-8") as f:
        json.dump(new_meta, f, indent=2, sort_keys=True)

    log_fn(
        f"reshard: wrote {dst_dir!r} "
        f"(chunks={len(chunk_paths)}, target_world={target_world_size})"
    )


__all__ = [
    "CHUNK_SHARD_FILE_RE",
    "CPU_OPTIM_DIRNAME",
    "GPU_OPTIM_FILENAME",
    "METADATA_FILENAME",
    "SAVE_MODE_SHARDED",
    "SCHEMA_FORMAT_VERSION",
    "_layout_signature_from_fingerprint",
    "_padded_region_bytes",
    "_reshard_region_state",
    "reshard_mode_c_shards",
]
