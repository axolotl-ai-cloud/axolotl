"""Streaming full/delta checkpoint chains against an exact, committed baseline.

The committed baseline is the set of bytes the destination last accepted, stored
on disk by tensor name. A publish streams ``(name, tensor)`` pairs into a
session; the session writes the artifact (full shards, or per-tensor XOR/zstd
frames with Adler-32 shard metadata) and stages the next baseline beside the
committed one. Only ``ChainWriter.commit`` promotes it, so a failed publish
leaves the baseline untouched.
"""

import errno
import json
import os
import shutil
import tempfile
import zlib
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file as save_bytes
from safetensors.torch import save_file

from axolotl.utils.delta_checkpoint import encode_xor_delta, validate_identity
from axolotl.utils.schemas.checkpoint_publish import MAX_CHAIN_DELTAS

from .encoder import LARGE_TENSOR_BYTES, encode_xor_delta_chunked
from .fingerprint import fingerprint_digest

INDEX_NAME = "model.safetensors.index.json"
MANIFEST_NAME = "baseline.json"
COMMITTED_NAME = "committed.json"


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _bytes(tensor: torch.Tensor) -> np.ndarray:
    return tensor.reshape(-1).view(torch.uint8).numpy()


def _link(source: Path, target: Path):
    try:
        os.link(source, target)
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EMLINK, errno.ENOTSUP}:
            raise
        shutil.copy2(source, target)


def _write_json(path: Path, value: Any):
    staged = path.with_name(f".{path.name}.tmp")
    staged.write_text(json.dumps(value), encoding="utf-8")
    os.replace(staged, path)


class Baseline(Mapping[str, torch.Tensor]):
    """Read-only view of a committed (or staged) baseline directory, keyed by name."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        manifest = json.loads((self.path / MANIFEST_NAME).read_text(encoding="utf-8"))
        self.identity: str = manifest["identity"]
        self.delta_count: int = manifest["delta_count"]
        self.fingerprint: dict | None = manifest["fingerprint"]
        self.layout: dict[str, tuple[str, list[int]]] = {
            name: (dtype, list(shape))
            for name, (dtype, shape) in manifest["layout"].items()
        }
        self.files: dict[str, str] = manifest["files"]
        self.checksums: dict[str, str] = manifest["checksums"]

    @property
    def fingerprint_digest(self) -> str | None:
        return fingerprint_digest(self.fingerprint)

    def exists(self) -> bool:
        return self.path.is_dir()

    def __eq__(self, other):
        return isinstance(other, Baseline) and other.path == self.path

    def __hash__(self):
        return hash(self.path)

    def __getitem__(self, name: str) -> torch.Tensor:
        with safe_open(self.path / self.files[name], framework="pt") as shard:
            return shard.get_tensor(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self.layout)

    def __len__(self) -> int:
        return len(self.layout)

    def verify(self) -> None:
        """Re-check every tensor against the recorded Adler-32 of its bytes."""
        for name in self.layout:
            if f"{zlib.adler32(_bytes(self[name])):08x}" != self.checksums[name]:
                raise ValueError(f"Baseline tensor {name} does not match its checksum")


class StagedCheckpoint(NamedTuple):
    """A written artifact and its staged baseline, awaiting commit or discard."""

    identity: str
    baseline: Baseline
    full: bool
    path: Path
    base_identity: str | None
    weight_map: dict[str, str]
    checksums: dict[str, str]
    total_size: int
    artifact_size: int
    reason: str
    built_on: str | None
    encoded_sizes: dict[str, int] | None = None

    @property
    def empty(self) -> bool:
        return not self.full and not self.weight_map


class ChainWriter:
    """Own one chain's committed baseline, the delta limit, and the fingerprint rule.

    ``rank`` names this writer's shard files so several writers (one per rank) can
    publish disjoint tensors into one checkpoint; rank 0 joins their indexes with
    ``combine_indexes``.
    """

    def __init__(
        self,
        workdir: Path | str,
        *,
        max_deltas: int = MAX_CHAIN_DELTAS,
        delta: bool = True,
        max_shard_size: int = 1 << 30,
        rank: int | None = None,
        workers: int | None = None,
        max_inflight: int | None = None,
        compression_level: int = 1,
        large_tensor_bytes: int = LARGE_TENSOR_BYTES,
        zstd_threads: int | None = None,
        retain_artifacts: bool = True,
        resume: bool = False,
    ):
        if not 0 <= max_deltas <= MAX_CHAIN_DELTAS:
            raise ValueError(f"max_deltas must be between 0 and {MAX_CHAIN_DELTAS}")
        if max_shard_size <= 0:
            raise ValueError("max_shard_size must be positive")
        self.workdir = Path(workdir)
        self.max_deltas, self.delta = max_deltas, delta
        self.max_shard_size = max_shard_size
        self.rank = rank
        self.workers = workers or min(8, os.cpu_count() or 1)
        self.max_inflight = max_inflight or 2 * self.workers
        self.compression_level = compression_level
        self.large_tensor_bytes = large_tensor_bytes
        self.zstd_threads = zstd_threads
        self.retain_artifacts = retain_artifacts
        self.baseline: Baseline | None = None
        self._pool: ThreadPoolExecutor | None = None
        if resume:
            self._resume()

    @property
    def identity(self) -> str | None:
        return None if self.baseline is None else self.baseline.identity

    @property
    def delta_count(self) -> int:
        return 0 if self.baseline is None else self.baseline.delta_count

    @property
    def pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self.workers, thread_name_prefix="chain-encode"
            )
        return self._pool

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def shard_name(self, number: int) -> str:
        if self.rank is None:
            return f"model-{number:05d}.safetensors"
        return f"model-rank{self.rank:03d}-{number:05d}.safetensors"

    def full_reason(
        self,
        fingerprint: dict | None = None,
        layout: Mapping[str, tuple[str, list[int]]] | None = None,
    ) -> str | None:
        """Why the next publish must be a full checkpoint, or ``None`` for a delta."""
        if self.baseline is None:
            return "no committed baseline"
        if not self.delta:
            return "deltas disabled"
        if self.delta_count >= self.max_deltas:
            return f"chain reached {self.max_deltas} deltas"
        if fingerprint_digest(fingerprint) != self.baseline.fingerprint_digest:
            return "merge numerics changed"
        if (
            layout is not None
            and {name: (dtype, list(shape)) for name, (dtype, shape) in layout.items()}
            != self.baseline.layout
        ):
            return "tensor layout changed"
        return None

    def begin(
        self,
        identity: str,
        *,
        fingerprint: dict | None = None,
        layout: Mapping[str, tuple[str, list[int]]] | None = None,
        force_full: bool = False,
    ) -> "ChainSession":
        validate_identity(identity)
        if identity == self.identity:
            raise ValueError("A new checkpoint needs a new identity")
        reason = "forced" if force_full else self.full_reason(fingerprint, layout)
        return ChainSession(self, identity, reason, fingerprint)

    def write(
        self,
        tensors: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
        identity: str,
        *,
        config: dict | None = None,
        fingerprint: dict | None = None,
        layout: Mapping[str, tuple[str, list[int]]] | None = None,
        force_full: bool = False,
        copy: bool = True,
    ) -> StagedCheckpoint:
        items = tensors.items() if isinstance(tensors, Mapping) else tensors
        with self.begin(
            identity, fingerprint=fingerprint, layout=layout, force_full=force_full
        ) as session:
            for name, tensor in items:
                session.add(name, tensor, copy=copy)
            return session.finish(config)

    def commit(self, staged: StagedCheckpoint):
        """Promote a published checkpoint's bytes to the committed baseline."""
        if staged.built_on != self.identity:
            raise RuntimeError(
                f"{staged.identity} was staged while {staged.built_on} was committed, "
                f"but the committed baseline is now {self.identity}"
            )
        if not staged.baseline.exists():
            raise RuntimeError(f"Staged baseline for {staged.identity} is missing")
        _write_json(self.workdir / COMMITTED_NAME, {"identity": staged.identity})
        previous, self.baseline = self.baseline, staged.baseline
        if previous is not None:
            shutil.rmtree(previous.path, ignore_errors=True)
        if not self.retain_artifacts:
            shutil.rmtree(staged.path, ignore_errors=True)

    def adopt(self, staged: StagedCheckpoint, *, delta_count: int):
        """Commit a rebuilt full checkpoint as the baseline of an earlier publish.

        The bytes were published before (as a full checkpoint or at the end of a
        delta chain), so only the baseline is kept and it continues that chain.
        """
        if not staged.full:
            raise ValueError("Only a full rebuild can be adopted as a baseline")
        if not 0 <= delta_count <= self.max_deltas:
            raise ValueError(f"Cannot adopt a chain with {delta_count} deltas")
        manifest_path = staged.baseline.path / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["delta_count"] = delta_count
        _write_json(manifest_path, manifest)
        self.commit(staged._replace(baseline=Baseline(staged.baseline.path)))
        shutil.rmtree(staged.path, ignore_errors=True)

    def discard(self, staged: StagedCheckpoint):
        """Drop a checkpoint that will not be committed; the baseline is unchanged."""
        if self.baseline is not None and staged.baseline.path == self.baseline.path:
            raise RuntimeError("Cannot discard the committed baseline")
        shutil.rmtree(staged.baseline.path, ignore_errors=True)
        shutil.rmtree(staged.path, ignore_errors=True)

    def _resume(self):
        pointer = self.workdir / COMMITTED_NAME
        if not pointer.is_file():
            return
        identity = json.loads(pointer.read_text(encoding="utf-8"))["identity"]
        baseline = Baseline(self.workdir / f".baseline-{validate_identity(identity)}")
        baseline.verify()
        self.baseline = baseline
        for stale in self.workdir.glob(".baseline-*"):
            if stale != baseline.path:
                shutil.rmtree(stale, ignore_errors=True)
        for stale in self.workdir.glob(".staging-*"):
            shutil.rmtree(stale, ignore_errors=True)


def combine_indexes(indexes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Join per-rank indexes of one checkpoint into the index rank 0 uploads last."""
    indexes = list(indexes)
    if not indexes:
        raise ValueError("No indexes to combine")
    metadata = dict(indexes[0]["metadata"])
    full = "delta_encoding" not in metadata
    weight_map: dict[str, str] = {}
    total = 0
    for index in indexes:
        other = dict(index["metadata"])
        if full:
            total += other.pop("total_size")
            metadata.pop("total_size", None)
        if other != {k: v for k, v in metadata.items() if k != "total_size"}:
            raise ValueError("Per-rank indexes disagree on checkpoint metadata")
        if weight_map.keys() & index["weight_map"].keys():
            raise ValueError("A tensor was written by more than one rank")
        if set(weight_map.values()) & set(index["weight_map"].values()):
            raise ValueError("Per-rank shard file names collide")
        weight_map.update(index["weight_map"])
    if full:
        metadata["total_size"] = total
    return {"metadata": metadata, "weight_map": weight_map}


class _Pending(NamedTuple):
    name: str
    tensor: torch.Tensor
    owned: bool
    future: Future


class ChainSession:
    """Stream one checkpoint's tensors; ``finish`` stages it, errors discard it."""

    def __init__(
        self,
        writer: ChainWriter,
        identity: str,
        reason: str | None,
        fingerprint: dict | None,
    ):
        self.writer, self.identity = writer, identity
        self.built_on = writer.identity
        self.base = None if reason is not None else writer.baseline
        self.full = self.base is None
        self.reason = reason or "delta"
        self.fingerprint = fingerprint
        destination = writer.workdir / identity
        if destination.exists():
            raise FileExistsError(destination)
        writer.workdir.mkdir(parents=True, exist_ok=True)
        self.staging = Path(
            tempfile.mkdtemp(prefix=f".staging-{identity}-", dir=writer.workdir)
        )
        self.artifact = self.staging / "artifact"
        self.next_baseline = self.staging / "baseline"
        self.artifact.mkdir()
        self.next_baseline.mkdir()
        self.layout: dict[str, tuple[str, list[int]]] = {}
        self.weight_map: dict[str, str] = {}
        self.checksums: dict[str, str] = {}
        self.baseline_files: dict[str, str] = {}
        self.encoded_sizes: dict[str, int] = {}
        self.total_size = 0
        self.artifact_size = 0
        self._inflight: deque[_Pending] = deque()
        self._shard: dict[str, Any] = {}
        self._shard_checksums: dict[str, str] = {}
        self._shard_size = 0
        self._shard_number = 0
        self._overlay: dict[str, torch.Tensor] = {}
        self._overlay_size = 0
        self._overlay_number = 0
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None or not self._closed:
            self.abort()
        return False

    def add(self, name: str, tensor: torch.Tensor, *, copy: bool = True):
        """Queue one tensor. With ``copy=False`` the caller must not mutate it."""
        if self._closed:
            raise RuntimeError("Session is closed")
        if name in self.layout:
            raise ValueError(f"Duplicate tensor {name}")
        tensor = tensor.detach().to("cpu")
        tensor = tensor.clone() if copy else tensor.contiguous()
        entry = (_dtype_name(tensor.dtype), list(tensor.shape))
        if self.base is not None and self.base.layout.get(name) != entry:
            raise ValueError(
                f"Tensor layout changed for {name}; publish a full checkpoint"
            )
        self.layout[name] = entry
        self.total_size += tensor.numel() * tensor.element_size()
        job = (
            self.writer.pool.submit(self._checksum, tensor)
            if self.full
            else self.writer.pool.submit(self._encode, name, tensor)
        )
        self._inflight.append(_Pending(name, tensor, copy, job))
        while len(self._inflight) > self.writer.max_inflight:
            self._drain_one()

    def keep(self, name: str):
        """Carry a baseline tensor known to be unchanged into a delta without reading it."""
        if self._closed:
            raise RuntimeError("Session is closed")
        if self.base is None:
            raise ValueError("A full checkpoint must add every tensor")
        if name in self.layout:
            raise ValueError(f"Duplicate tensor {name}")
        if name not in self.base.layout:
            raise KeyError(f"{name} is not in the baseline")
        self.layout[name] = self.base.layout[name]

    def _checksum(self, tensor):
        return f"{zlib.adler32(_bytes(tensor)):08x}"

    def _encode(self, name, tensor):
        assert self.base is not None
        current, previous = _bytes(tensor), _bytes(self.base[name].contiguous())
        if current.nbytes >= self.writer.large_tensor_bytes:
            return encode_xor_delta_chunked(
                current,
                previous,
                level=self.writer.compression_level,
                threads=self.writer.zstd_threads,
            )
        return encode_xor_delta(current, previous, level=self.writer.compression_level)

    def _drain_one(self):
        name, tensor, owned, future = self._inflight.popleft()
        result = future.result()
        # safetensors refuses tensors that share storage, e.g. tied weights
        if not owned:
            tensor = tensor.clone()
        if self.full:
            self.checksums[name] = result
            self.encoded_sizes[name] = tensor.numel() * tensor.element_size()
            self._append(name, tensor, self.encoded_sizes[name])
            return
        if result is None:
            return
        encoded, digest = result
        self.checksums[name] = digest
        self.encoded_sizes[name] = encoded.nbytes
        self._append(name, encoded, encoded.nbytes)
        self._shard_checksums[name] = digest
        nbytes = tensor.numel() * tensor.element_size()
        if self._overlay and self._overlay_size + nbytes > self.writer.max_shard_size:
            self._flush_overlay()
        self._overlay[name] = tensor
        self._overlay_size += nbytes

    def _append(self, name, value, nbytes):
        if self._shard and self._shard_size + nbytes > self.writer.max_shard_size:
            self._flush_shard()
        self._shard[name] = value
        self._shard_size += nbytes

    def _flush_shard(self):
        if not self._shard:
            return
        filename = self.writer.shard_name(self._shard_number)
        path = self.artifact / filename
        if self.full:
            save_file(self._shard, path, metadata={"format": "pt"})
        else:
            save_bytes(self._shard, path, metadata=dict(self._shard_checksums))
        self.artifact_size += path.stat().st_size
        self.weight_map.update(dict.fromkeys(self._shard, filename))
        self._shard.clear()
        self._shard_checksums.clear()
        self._shard_size = 0
        self._shard_number += 1

    def _flush_overlay(self):
        if not self._overlay:
            return
        filename = f"{self.identity}-{self._overlay_number:05d}.safetensors"
        save_file(self._overlay, self.next_baseline / filename)
        self.baseline_files.update(dict.fromkeys(self._overlay, filename))
        self._overlay.clear()
        self._overlay_size = 0
        self._overlay_number += 1

    def index(self) -> dict[str, Any]:
        if self.full:
            metadata: dict[str, Any] = {"total_size": self.total_size}
        else:
            assert self.base is not None
            metadata = {
                "version": self.identity,
                "base_version": self.base.identity,
                "delta_encoding": "xor",
                "compression_format": "zstd",
                "checksum_format": "adler32",
            }
        if (digest := fingerprint_digest(self.fingerprint)) is not None:
            metadata["merge_fingerprint"] = digest
        return {"metadata": metadata, "weight_map": dict(self.weight_map)}

    def finish(self, config: dict | None = None) -> StagedCheckpoint:
        """Write the index and staged baseline; the artifact is then ready to upload."""
        if self._closed:
            raise RuntimeError("Session is closed")
        while self._inflight:
            self._drain_one()
        if not self.layout:
            raise ValueError("Cannot publish an empty checkpoint")
        if self.full and config is None:
            raise ValueError("A full checkpoint requires the model config")
        if self.base is not None and (
            missing := self.base.layout.keys() - self.layout.keys()
        ):
            raise ValueError(
                f"Checkpoint is missing {len(missing)} baseline tensors, "
                f"e.g. {sorted(missing)[0]}"
            )
        self._flush_shard()
        self._flush_overlay()
        if self.full:
            (self.artifact / "config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            for filename in set(self.weight_map.values()):
                linked = f"{self.identity}-{filename}"
                _link(self.artifact / filename, self.next_baseline / linked)
            files = {
                name: f"{self.identity}-{f}" for name, f in self.weight_map.items()
            }
            checksums = dict(self.checksums)
        else:
            assert self.base is not None
            files = dict(self.base.files)
            files.update(self.baseline_files)
            checksums = {**self.base.checksums, **self.checksums}
            for filename in set(files.values()) - set(self.baseline_files.values()):
                _link(self.base.path / filename, self.next_baseline / filename)
        (self.artifact / INDEX_NAME).write_text(
            json.dumps(self.index()), encoding="utf-8"
        )
        _write_json(
            self.next_baseline / MANIFEST_NAME,
            {
                "identity": self.identity,
                "delta_count": 0 if self.base is None else self.base.delta_count + 1,
                "fingerprint": self.fingerprint,
                "layout": self.layout,
                "files": files,
                "checksums": checksums,
            },
        )
        path = self.writer.workdir / self.identity
        baseline_path = self.writer.workdir / f".baseline-{self.identity}"
        if baseline_path.exists():
            raise FileExistsError(baseline_path)
        self.artifact.rename(path)
        try:
            self.next_baseline.rename(baseline_path)
        except BaseException:
            shutil.rmtree(path, ignore_errors=True)
            raise
        self._closed = True
        shutil.rmtree(self.staging, ignore_errors=True)
        return StagedCheckpoint(
            identity=self.identity,
            baseline=Baseline(baseline_path),
            full=self.full,
            path=path,
            base_identity=None if self.base is None else self.base.identity,
            weight_map=dict(self.weight_map),
            checksums=dict(self.checksums),
            total_size=self.total_size,
            artifact_size=self.artifact_size,
            reason=self.reason,
            built_on=self.built_on,
            encoded_sizes=dict(self.encoded_sizes),
        )

    def abort(self):
        for pending in self._inflight:
            pending.future.cancel()
        for pending in self._inflight:
            if not pending.future.cancelled():
                pending.future.exception()
        self._inflight.clear()
        self._closed = True
        shutil.rmtree(self.staging, ignore_errors=True)
