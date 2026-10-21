# Copyright 2026 Axolotl AI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Sharded FSDP2 adapter checkpoints: every rank writes the adapter slices it holds, and rank 0
consolidates them into ``adapter_model.safetensors`` on a background thread.

Shard files use the layout of ``torch.distributed.checkpoint``'s Hugging Face storage
(``DCP_SHARDING_INFO`` with per-tensor ``saved_offsets``). An interleaved expert LoRA slice is
one box only in the view ``_box`` reshapes it into, so shards and offsets live in that view
and the consolidated header records the real PEFT shape over the same row-major bytes.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import torch
import torch.distributed as dist
from accelerate.utils.modeling import is_peft_model

from axolotl.utils.logging import get_logger

from .fsdp2_checkpoint import (
    _adapter_model_state,
    _boxes,
    _check_errors,
    _dtype,
    _gather_records,
    _gather_tensor,
    _gather_value,
    _local,
    _staging,
    _transfer,
    expert_ownership,
)

LOG = get_logger(__name__)

SHARD_DIRNAME = ".adapter_shards"
EXPORT_DIRNAME = "export"
MANIFEST = "manifest.json"
ADAPTER_FILE = "adapter_model.safetensors"
_ADAPTER_FILES = (ADAPTER_FILE, "adapter_model.bin")
_VERSION = 1
_WRITE_BYTES = 64 * 1024 * 1024
CONSOLIDATE_THREADS = 4


def shard_name(source: int) -> str:
    return f"shard-{source:05d}-model-00001-of-00001.safetensors"


def safetensors_order(dtypes: dict[str, torch.dtype]) -> list[tuple[str, str]]:
    """``(name, safetensors dtype)`` in the order ``safetensors`` lays the tensors out."""
    from safetensors.torch import save

    raw = save({name: torch.empty(0, dtype=dtype) for name, dtype in dtypes.items()})
    (length,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + length])
    header.pop("__metadata__", None)
    return [(name, entry["dtype"]) for name, entry in header.items()]


def _header(entries, metadata):
    """Length-prefixed header (padded like ``safetensors``) and each entry's data offset."""
    header = {"__metadata__": metadata} if metadata else {}
    offsets, offset = [], 0
    for name, dtype, shape, nbytes in entries:
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offsets.append(offset)
        offset += nbytes
    text = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    text += b" " * (-len(text) % 8)
    return struct.pack("<Q", len(text)) + text, offsets, offset


def _bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def _write_tensor(stream, tensor):
    data = _bytes(tensor)
    if data.device.type == "cpu":
        stream.write(memoryview(data.numpy()))
        return
    for start in range(0, data.numel(), _WRITE_BYTES):
        piece = data[start : start + _WRITE_BYTES]
        staged = _staging(torch.uint8, piece.numel())
        staged.copy_(piece, non_blocking=True)
        torch.cuda.current_stream(piece.device).synchronize()
        stream.write(memoryview(staged.numpy()))


class _ShardWriter:
    """Stream tensors into a safetensors file whose header is known up front."""

    def __init__(self, path, entries, metadata):
        prefix, _, _ = _header(entries, metadata)
        self.expected = [(name, nbytes) for name, _, _, nbytes in entries]
        self.position = 0
        self.stream = open(path, "wb")  # noqa: SIM115
        self.stream.write(prefix)

    def write(self, name, tensor):
        expected, nbytes = self.expected[self.position]
        if name != expected or tensor.numel() * tensor.element_size() != nbytes:
            raise ValueError(f"Adapter shard received {name} out of order or resized")
        self.position += 1
        if nbytes:
            _write_tensor(self.stream, tensor)

    def close(self):
        try:
            if self.position != len(self.expected):
                raise ValueError("Adapter shard is missing tensors")
            self.stream.flush()
            os.fsync(self.stream.fileno())
        finally:
            self.stream.close()


@dataclass
class _Piece:
    source: int
    starts: list
    lengths: list
    local_shape: list


@dataclass
class _Plan:
    kind: str  # "box": per-rank slices, "gather": rank 0 assembles, "scalar"
    dtype: torch.dtype
    shape: list
    view: list
    pieces: list


def _chunks(records, visible):
    """``_gather_tensor``'s unique chunks, preferring a source whose files rank 0 can read."""
    chosen: dict = {}
    for source, info in enumerate(records):
        identity = repr(
            (info["owner"], info["offset"], info["local_shape"], info.get("axes"))
        )
        if identity not in chosen or (
            visible[source] and not visible[chosen[identity][0]]
        ):
            chosen[identity] = (source, info)
    return list(chosen.values())


def _plan(records, visible):
    shape = records[0]["shape"]
    dtype = _dtype(records[0]["dtype"])
    if any(info["shape"] != shape for info in records):
        raise ValueError(
            "Full checkpoint failed: Checkpoint shards disagree on the global tensor shape"
        )
    if any(info["quantized"] for info in records):
        return None
    full = _Plan(
        "gather", dtype, shape, shape, [_Piece(0, [0] * len(shape), shape, shape)]
    )
    # every chunk must already hold the export dtype for its bytes to be copied verbatim
    if any(_dtype(info["dtype"]) != dtype for info in records):
        return full
    chunks = _chunks(records, visible)
    boxes = _boxes(chunks)
    if boxes is None:
        return full
    view, placed = boxes
    pieces = [
        _Piece(source, starts, lengths, info["local_shape"])
        for (source, info), (starts, lengths) in zip(chunks, placed, strict=True)
        if math.prod(lengths)
    ]
    return _Plan("box", dtype, shape, view, pieces)


@dataclass
class AdapterShards:
    directory: Path
    target: str
    write_seconds: float
    nbytes: int


def _probe(shard_dir):
    """Which ranks see the files rank 0 writes (``False`` on another node's local disk)."""
    token = [uuid.uuid4().hex if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(token, src=0)
    probe = shard_dir / f"probe-{token[0]}"
    error = None
    if dist.get_rank() == 0:
        try:
            shutil.rmtree(shard_dir, ignore_errors=True)
            shard_dir.mkdir(parents=True)
            probe.touch()
        except OSError as exc:
            error = f"Cannot create {shard_dir}: {exc}"
    _check_errors(error)
    visible = [None] * dist.get_world_size()
    dist.all_gather_object(visible, probe.is_file())
    if dist.get_rank() == 0:
        probe.unlink(missing_ok=True)
    return visible


def write_adapter_shards(model, output_dir) -> AdapterShards | None:
    """Collective: write this rank's adapter slices under ``output_dir/.adapter_shards``.

    Ranks whose files rank 0 cannot read send their slices to rank 0, which writes them in
    their place. Returns ``None`` on every rank when the adapter needs the gather path.
    """
    if not is_peft_model(model) or len(model.peft_config) != 1:
        return None
    (adapter,) = model.peft_config
    exports: dict = {}
    selected = _adapter_model_state(model, exports)
    names = {key: names[0] for key, names in exports[adapter].items()}
    values = {key: selected[name] for key, name in names.items()}
    owners = expert_ownership(model)
    tensors = [key for key, value in values.items() if value.ndim > 0]
    records = dict(
        zip(
            tensors,
            _gather_records([(values[key], owners.get(names[key])) for key in tensors]),
            strict=True,
        )
    )
    root = Path(output_dir)
    shard_dir = root / SHARD_DIRNAME
    visible = _probe(shard_dir)
    plans = {}
    for key, value in values.items():
        if key in records:
            plan = _plan(records[key], visible)
            if plan is None:
                LOG.warning(
                    f"{key} is quantized; saving the adapter through the rank 0 gather"
                )
                if dist.get_rank() == 0:
                    shutil.rmtree(shard_dir, ignore_errors=True)
                return None
        else:
            plan = _Plan("scalar", _local(value).dtype, [], [], [_Piece(0, [], [], [])])
        plans[key] = plan
    order = safetensors_order({key: plan.dtype for key, plan in plans.items()})
    rank = dist.get_rank()

    def writer(source):
        return source if visible[source] else 0

    files: dict[int, list] = {}
    sharding: dict[int, dict] = {}
    for key, st_dtype in order:
        plan = plans[key]
        size = torch.empty((), dtype=plan.dtype).element_size()
        for piece in plan.pieces:
            files.setdefault(piece.source, []).append(
                (key, st_dtype, piece.lengths, math.prod(piece.lengths) * size)
            )
            sharding.setdefault(piece.source, {})[key] = {"saved_offsets": piece.starts}
    began = time.perf_counter()
    error, written, writers = None, 0, {}
    try:
        for source, entries in files.items():
            if writer(source) == rank:
                metadata = {
                    "format": "pt",
                    "DCP_VERSION": "1.0",
                    "DCP_SHARDING_INFO": json.dumps(sharding[source]),
                }
                writers[source] = _ShardWriter(
                    shard_dir / shard_name(source), entries, metadata
                )
    except OSError as exc:
        error = f"rank {rank}: {exc}"

    def write(source, key, tensor):
        nonlocal error, written
        if error is not None:
            return
        try:
            writers[source].write(key, tensor)
            written += tensor.numel() * tensor.element_size()
        except (OSError, ValueError, RuntimeError) as exc:
            error = f"rank {rank}: {exc}"

    # every rank walks the same plan so the collectives below line up
    for key, _ in order:
        plan, value = plans[key], values[key]
        owner = owners.get(names[key])
        if plan.kind == "box":
            local = _local(value)
            for piece in plan.pieces:
                if visible[piece.source]:
                    if rank == piece.source:
                        write(piece.source, key, local)
                    continue
                received = _transfer(
                    local if rank == piece.source else None,
                    piece.local_shape,
                    plan.dtype,
                    piece.source,
                )
                if rank == 0:
                    write(piece.source, key, received)
        elif plan.kind == "gather":
            full = _gather_tensor(value, owner, records[key])
            if rank == 0:
                write(0, key, full)
        else:
            scalar = _gather_value(value, owner)
            if rank == 0:
                if torch.is_tensor(scalar):
                    write(0, key, scalar)
                elif error is None:
                    error = f"{key}: replicas hold different scalar values"
    for handle in writers.values():
        try:
            handle.close()
        except (OSError, ValueError) as exc:
            error = error or f"rank {rank}: {exc}"
    seconds = time.perf_counter() - began
    results = [None] * dist.get_world_size()
    dist.all_gather_object(results, (error, seconds, written))
    errors = [item[0] for item in results if item[0]]
    error = "; ".join(errors) if errors else None
    target = ADAPTER_FILE if adapter == "default" else f"{adapter}/{ADAPTER_FILE}"
    if error is None and rank == 0:
        manifest = {
            "version": _VERSION,
            "target": target,
            "shards": [shard_name(source) for source in sorted(files)],
            "tensors": [
                {
                    "name": key,
                    "dtype": str(plans[key].dtype),
                    "st_dtype": st_dtype,
                    "shape": plans[key].shape,
                    "view": plans[key].view,
                }
                for key, st_dtype in order
            ],
        }
        try:
            _write_json(shard_dir / MANIFEST, manifest)
            # a stale export must not pass for this save while it consolidates
            for name in _ADAPTER_FILES:
                (root / target).with_name(name).unlink(missing_ok=True)
        except OSError as exc:
            error = f"rank 0: {exc}"
    final = [error]
    dist.broadcast_object_list(final, src=0)
    if final[0]:
        if rank == 0:
            shutil.rmtree(shard_dir, ignore_errors=True)
        raise ValueError(f"Sharded adapter save failed: {final[0]}")
    shards = AdapterShards(
        shard_dir,
        target,
        max(item[1] for item in results),
        sum(item[2] for item in results),
    )
    if rank == 0:
        LOG.info(
            f"Wrote {shards.nbytes / 2**30:.2f} GiB of adapter shards in "
            f"{shards.write_seconds:.1f}s (slowest rank, {len(files)} files) to {shard_dir}"
        )
    return shards


def _write_json(path, data):
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def move_export(export_dir, output_dir):
    """Move everything ``save_pretrained`` wrote except the (empty) adapter weights."""
    export_dir, output_dir = Path(export_dir), Path(output_dir)
    for path in sorted(export_dir.rglob("*")):
        if not path.is_file() or path.name in _ADAPTER_FILES:
            continue
        destination = output_dir / path.relative_to(export_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)
    shutil.rmtree(export_dir, ignore_errors=True)


def _read_into(path, offset, tensor):
    if not tensor.numel():
        return
    buffer = memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
    with open(path, "rb", buffering=0) as stream:
        stream.seek(offset)
        while buffer:
            count = stream.readinto(buffer)
            if not count:
                raise ValueError(f"{path} is truncated")
            buffer = buffer[count:]


def _pwrite(fd, data, offset):
    view = memoryview(data)
    while view:
        count = os.pwrite(fd, view, offset)
        offset += count
        view = view[count:]


def _fsync_directory(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def consolidate(shard_dir, threads: int = CONSOLIDATE_THREADS):
    """Assemble the shards into the PEFT adapter file, move it into place and drop the shards.

    Returns ``(path, seconds, bytes)``.
    """
    began = time.perf_counter()
    shard_dir = Path(shard_dir)
    manifest = json.loads((shard_dir / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("version") != _VERSION:
        raise ValueError(f"Unsupported adapter shard manifest in {shard_dir}")
    target_parts = PurePosixPath(manifest["target"]).parts
    if (
        target_parts[-1:] != (ADAPTER_FILE,)
        or len(target_parts) > 2
        or {"/", ".", ".."} & set(target_parts)
        or any(Path(name).name != name for name in manifest["shards"])
    ):
        raise ValueError(f"Adapter shard manifest in {shard_dir} points outside it")
    pieces: dict[str, list] = {entry["name"]: [] for entry in manifest["tensors"]}
    for name in manifest["shards"]:
        path = shard_dir / name
        with open(path, "rb") as stream:
            (length,) = struct.unpack("<Q", stream.read(8))
            header = json.loads(stream.read(length))
        offsets = json.loads(header.pop("__metadata__")["DCP_SHARDING_INFO"])
        for key, entry in header.items():
            pieces[key].append(
                (
                    path,
                    8 + length + entry["data_offsets"][0],
                    entry["shape"],
                    offsets[key]["saved_offsets"],
                )
            )
    tensors = manifest["tensors"]
    sizes = [
        math.prod(entry["shape"])
        * torch.empty((), dtype=_dtype(entry["dtype"])).element_size()
        for entry in tensors
    ]
    prefix, offsets, total = _header(
        [
            (entry["name"], entry["st_dtype"], entry["shape"], size)
            for entry, size in zip(tensors, sizes, strict=True)
        ],
        {"format": "pt"},
    )

    def assemble(index):
        entry = tensors[index]
        dtype, view = _dtype(entry["dtype"]), entry["view"]
        parts = pieces[entry["name"]]
        if sum(math.prod(shape) for _, _, shape, _ in parts) != math.prod(view):
            raise ValueError(f"Adapter shards do not cover {entry['name']}")
        out = torch.empty(view, dtype=dtype)
        if len(parts) == 1:
            _read_into(parts[0][0], parts[0][1], out)
        else:
            for path, offset, shape, starts in parts:
                chunk = torch.empty(shape, dtype=dtype)
                _read_into(path, offset, chunk)
                destination = out
                for dim, (start, length) in enumerate(zip(starts, shape, strict=True)):
                    destination = destination.narrow(dim, start, length)
                destination.copy_(chunk)
        if out.numel():
            _pwrite(
                fd,
                out.reshape(-1).view(torch.uint8).numpy(),
                len(prefix) + offsets[index],
            )

    tmp = shard_dir / f"{ADAPTER_FILE}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, len(prefix) + total)
        _pwrite(fd, prefix, 0)
        with ThreadPoolExecutor(
            max_workers=max(1, threads), thread_name_prefix="adapter-consolidate-copy"
        ) as pool:
            list(pool.map(assemble, range(len(tensors))))
        os.fsync(fd)
    finally:
        os.close(fd)
    target = shard_dir.parent / manifest["target"]
    # never recreate a checkpoint that rotation deleted mid-consolidation
    if target.parent != shard_dir.parent:
        target.parent.mkdir(exist_ok=True)
    os.replace(tmp, target)
    _fsync_directory(target.parent)
    # a crash inside rmtree must not leave a manifest that resume would consolidate again
    (shard_dir / MANIFEST).unlink()
    shutil.rmtree(shard_dir, ignore_errors=True)
    return target, time.perf_counter() - began, len(prefix) + total


def _key(path) -> str:
    return os.path.abspath(path)


class ConsolidationQueue:
    """Rank 0's background consolidations: one runs at a time, and at most ``limit`` are
    outstanding before a new save waits for the oldest."""

    def __init__(self, limit: int = 2):
        self.limit = limit
        self.executor: ThreadPoolExecutor | None = None
        self.futures: dict[str, Future] = {}
        self.lock = threading.Lock()

    def submit(self, shard_dir) -> Future:
        checkpoint = _key(Path(shard_dir).parent)
        with self.lock:
            if self.executor is None:
                # a non-daemon worker: interpreter exit waits for a running consolidation
                self.executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="adapter-consolidate"
                )
            future = self.executor.submit(self._run, Path(shard_dir))
            self.futures[checkpoint] = future
        return future

    @staticmethod
    def _run(shard_dir):
        try:
            target, seconds, nbytes = consolidate(shard_dir)
        except Exception:
            LOG.exception(
                f"Consolidating {shard_dir} failed; the shards are kept and the next "
                "resume from this checkpoint retries"
            )
            raise
        LOG.info(
            f"Consolidated {target} in the background: {nbytes / 2**30:.2f} GiB in "
            f"{seconds:.1f}s ({nbytes / 2**30 / max(seconds, 1e-9):.2f} GiB/s)"
        )
        return target

    def pending(self) -> list[str]:
        with self.lock:
            return [key for key, future in self.futures.items() if not future.done()]

    def status(self, checkpoint) -> bool | None:
        """``None`` while consolidating, then whether it succeeded (``True`` if unknown)."""
        with self.lock:
            future = self.futures.get(_key(checkpoint))
        if future is None:
            return True
        if not future.done():
            return None
        return future.exception() is None

    def wait(self, checkpoint=None):
        with self.lock:
            futures = (
                list(self.futures.values())
                if checkpoint is None
                else [self.futures[_key(checkpoint)]]
                if _key(checkpoint) in self.futures
                else []
            )
        for future in futures:
            try:
                future.result()
            except Exception:  # pylint: disable=broad-except
                pass

    def wait_for_capacity(self):
        while True:
            with self.lock:
                outstanding = [f for f in self.futures.values() if not f.done()]
            if len(outstanding) < self.limit:
                return
            LOG.info(
                "Waiting for an earlier adapter consolidation before the next save"
            )
            try:
                outstanding[0].result()
            except Exception:  # pylint: disable=broad-except
                pass


CONSOLIDATIONS = ConsolidationQueue()
_SHARDED_EXPORTS: set[str] = set()


def mark_sharded_export(checkpoint):
    _SHARDED_EXPORTS.add(_key(checkpoint))


def is_sharded_export(checkpoint) -> bool:
    """Whether this run wrote ``checkpoint``'s adapter through the sharded path."""
    return _key(checkpoint) in _SHARDED_EXPORTS


def complete_interrupted_consolidation(checkpoint) -> bool:
    """Finish a consolidation that a stopped run left behind; ``True`` if one was needed."""
    checkpoint = Path(checkpoint)
    CONSOLIDATIONS.wait(checkpoint)
    shard_dir = checkpoint / SHARD_DIRNAME
    if not (shard_dir / MANIFEST).is_file():
        return False
    LOG.info(f"Consolidating the adapter shards left in {shard_dir}")
    consolidate(shard_dir)
    return True
