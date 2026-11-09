"""Crash-recovery adapter checkpoints written from the staged LoRA snapshot.

Each checkpoint is a directory ``<dir>/step-<N>/`` in which every rank writes
only its own files, so ``dir`` can be node-local:

- ``rank-<r>.safetensors``: the rank's LoRA slices, keyed by adapter key;
- ``optimizer-rank-<r>.pt``: the rank's optimizer state, on a slower cadence;
- ``rank-<r>.json``: the manifest (pieces, step, world size). It is written only
  after every rank's files are on disk, so it marks the rank's part of a
  complete checkpoint.

The writes run on a background thread. Consolidation into a PEFT adapter
directory assembles the full adapter on rank 0 over a gloo group, key by key,
and runs only for the checkpoints that are kept (``consolidate_every`` and the
final one); the rest rotate away as shards.
"""

import json
import os
import shutil
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors import safe_open

from axolotl.utils.logging import get_logger

from .lora_shards import Need, Piece, assemble, full_box, global_shapes, plan_transfers
from .staging import LoRAStager

LOG = get_logger(__name__)

MANIFEST_FORMAT = 1
ADAPTER_NAME = "adapter_model.safetensors"
ADAPTER_CONFIG_NAME = "adapter_config.json"
_SAFETENSORS_DTYPES = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e5m2: "F8_E5M2",
}


def step_dir(directory: Path, step: int) -> Path:
    return directory / f"step-{step:08d}"


def shard_name(rank: int) -> str:
    return f"rank-{rank:05d}.safetensors"


def optimizer_name(rank: int) -> str:
    return f"optimizer-rank-{rank:05d}.pt"


def manifest_name(rank: int) -> str:
    return f"rank-{rank:05d}.json"


def _piece_json(piece: Piece) -> dict:
    return json.loads(json.dumps(asdict(piece)))


def write_safetensors(
    path: Path,
    names: Sequence[str],
    specs: Mapping[str, tuple[torch.dtype, Sequence[int]]],
    tensors,
    metadata: Mapping[str, str] | None = None,
) -> int:
    """Write a safetensors file tensor by tensor; ``tensors`` yields them in ``names`` order.

    The header is laid out from ``specs`` up front, so no tensor has to be held
    or serialized in full before writing. Writes ``path`` atomically and fsyncs.
    """
    header: dict[str, Any] = {"__metadata__": dict(metadata or {"format": "pt"})}
    offset = 0
    for name in names:
        dtype, shape = specs[name]
        size = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        header[name] = {
            "dtype": _SAFETENSORS_DTYPES[dtype],
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    staged = path.with_name(f".{path.name}.tmp")
    with open(staged, "wb") as handle:
        handle.write(len(encoded).to_bytes(8, "little"))
        handle.write(encoded)
        for name, tensor in zip(names, tensors, strict=True):
            dtype, shape = specs[name]
            if tensor.dtype != dtype or tuple(tensor.shape) != tuple(shape):
                raise ValueError(
                    f"{name}: expected {dtype} {tuple(shape)}, got "
                    f"{tensor.dtype} {tuple(tensor.shape)}"
                )
            data = tensor.detach().to("cpu").contiguous().reshape(-1)
            handle.write(memoryview(data.view(torch.uint8).numpy()))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staged, path)
    return offset


def _flatten_state(value, tensors: list, param: int | None = None):
    if isinstance(value, torch.Tensor):
        is_dtensor = type(value).__name__ == "DTensor"
        tensors.append(value.to_local() if is_dtensor else value)
        return {"__tensor__": len(tensors) - 1, "dtensor": is_dtensor, "param": param}
    if isinstance(value, Mapping):
        return {
            key: _flatten_state(item, tensors, param) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        items = [_flatten_state(item, tensors, param) for item in value]
        return items if isinstance(value, list) else tuple(items)
    return value


def _unflatten_state(value, tensors: Sequence[torch.Tensor], wrap: Callable):
    if isinstance(value, Mapping):
        if "__tensor__" in value:
            return wrap(tensors[value["__tensor__"]], value)
        return {
            key: _unflatten_state(item, tensors, wrap) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        items = [_unflatten_state(item, tensors, wrap) for item in value]
        return items if isinstance(value, list) else tuple(items)
    return value


class OptimizerStager:
    """Host copy of this rank's optimizer state, staged like the LoRA snapshot.

    Host buffers are reused while the state's layout stays the same. DTensor
    state is copied as its local shard and rewrapped on load.
    """

    def __init__(self):
        self._stager: LoRAStager | None = None
        self._layout: list | None = None
        self.skeleton: dict | None = None

    @property
    def nbytes(self) -> int:
        return self._stager.nbytes if self._stager is not None else 0

    def stage(self, optimizer) -> None:
        tensors: list[torch.Tensor] = []
        state = optimizer.state_dict()
        skeleton = {
            "state": {
                key: _flatten_state(value, tensors, key)
                for key, value in state["state"].items()
            },
            "param_groups": _flatten_state(state["param_groups"], tensors),
        }
        layout = [(t.shape, t.dtype, t.device) for t in tensors]
        if self._stager is None or layout != self._layout:
            self._stager, self._layout = LoRAStager(tensors, slots=1), layout
        else:
            self._stager.sources = tensors
        self._stager.stage(0)
        self.skeleton = skeleton

    def wait_training(self) -> None:
        if self._stager is not None:
            self._stager.wait_training()

    def ready(self) -> tuple[dict, list[torch.Tensor]]:
        if self._stager is None or self.skeleton is None:
            raise RuntimeError("No optimizer state was staged")
        return self.skeleton, self._stager.ready(0)


def load_optimizer_state(path: Path | str, optimizer) -> None:
    """Load one rank's saved optimizer state into an optimizer with the same sharding."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    params = [p for group in optimizer.param_groups for p in group["params"]]

    def wrap(tensor, entry):
        if not entry["dtensor"]:
            return tensor
        from torch.distributed.tensor import DTensor

        param = params[entry["param"]]
        return DTensor.from_local(
            tensor.to(param.to_local().device),
            param.device_mesh,
            param.placements,
            run_check=False,
            shape=param.shape,
            stride=param.stride(),
        )

    skeleton = saved["skeleton"]
    state = {
        "state": {
            int(key): _unflatten_state(value, saved["tensors"], wrap)
            for key, value in skeleton["state"].items()
        },
        "param_groups": _unflatten_state(
            skeleton["param_groups"], saved["tensors"], wrap
        ),
    }
    optimizer.load_state_dict(state)


@dataclass(frozen=True)
class RecoverySchedule:
    """Resolved recovery-save settings (see ``RecoveryCheckpointConfig``)."""

    every: int
    directory: Path
    keep: int = 2
    optimizer_every: int = 0
    consolidate_every: int = 0
    consolidate_dir: Path | None = None
    final: bool = True

    @classmethod
    def from_config(
        cls, config, *, output_dir: Path | str, workdir: Path | str | None = None
    ) -> "RecoverySchedule":
        output_dir = Path(output_dir)
        return cls(
            every=config.every,
            directory=Path(config.dir or Path(workdir or output_dir) / "recovery"),
            keep=config.keep,
            optimizer_every=config.optimizer_every,
            consolidate_every=config.consolidate_every,
            consolidate_dir=Path(
                config.consolidate_dir or output_dir / "recovery-adapters"
            ),
            final=config.final,
        )


@dataclass
class RecoveryJob:
    step: int
    lora: Callable[[], Sequence[torch.Tensor]] | None
    release: Callable[[], None]
    optimizer: OptimizerStager | None = None
    consolidate: bool = False


class _ShardData(Sequence):
    """This rank's pieces read back lazily from its shard file."""

    def __init__(self, path: Path, pieces: Sequence[Piece]):
        self.handle = safe_open(str(path), framework="pt")
        self.pieces = pieces

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, number):
        return self.handle.get_tensor(self.pieces[number].key)


class RecoveryWriter:
    """Write this rank's recovery checkpoints on a background thread.

    ``submit`` runs on the training thread and needs ``busy()`` to be false on
    every rank (the caller agrees on that over the training group). Everything
    else, including every ``comm`` collective, runs on the writer thread.
    """

    def __init__(
        self,
        directory: Path | str,
        pieces: Sequence[Piece],
        comm,
        *,
        keep: int = 2,
        consolidate_dir: Path | str | None = None,
        adapter_config: Mapping[str, Any] | None = None,
    ):
        keys = [piece.key for piece in pieces]
        if len(set(keys)) != len(keys):
            raise ValueError("A rank holds more than one slice of an adapter tensor")
        self.directory = Path(directory)
        self.pieces = list(pieces)
        self.comm = comm
        self.keep = keep
        self.consolidate_dir = Path(consolidate_dir) if consolidate_dir else None
        self.adapter_config = dict(adapter_config) if adapter_config else None
        self.last_written: int | None = None
        self.last_consolidated: int | None = None
        self.metrics: deque[dict] = deque()
        self.error: BaseException | None = None
        self._jobs: deque[RecoveryJob | None] = deque()
        self._cond = threading.Condition()
        self._busy = False
        self._all_pieces: list[list[Piece]] | None = None
        self.directory.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(
            target=self._run, name="recovery-checkpoint", daemon=True
        )
        self.thread.start()

    def busy(self) -> bool:
        self.check()
        with self._cond:
            return self._busy

    def check(self):
        if self.error is not None:
            raise RuntimeError("Recovery checkpointing failed") from self.error

    def submit(self, job: RecoveryJob):
        self.check()
        with self._cond:
            if self._busy:
                raise RuntimeError("A recovery checkpoint is still being written")
            self._busy = True
            self._jobs.append(job)
            self._cond.notify_all()

    def wait(self, timeout: float | None = None):
        with self._cond:
            self._cond.wait_for(
                lambda: not self._busy or self.error is not None, timeout
            )
        self.check()

    def drain_metrics(self) -> list[dict]:
        out = []
        while self.metrics:
            out.append(self.metrics.popleft())
        return out

    def close(self, timeout: float | None = None):
        with self._cond:
            self._jobs.append(None)
            self._cond.notify_all()
        self.thread.join(timeout)
        self.check()

    def _run(self):
        try:
            while True:
                with self._cond:
                    self._cond.wait_for(lambda: bool(self._jobs))
                    job = self._jobs.popleft()
                if job is None:
                    return
                try:
                    self._process(job)
                finally:
                    with self._cond:
                        self._busy = False
                        self._cond.notify_all()
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            LOG.exception("Recovery checkpoint thread failed")
            self.error = exc
            with self._cond:
                self._busy = False
                self._cond.notify_all()

    def _process(self, job: RecoveryJob):
        metrics: dict[str, Any] = {"recovery_step": job.step}
        if job.lora is not None:
            started = time.perf_counter()
            error = None
            try:
                metrics["recovery_bytes"] = self._write(job)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                error = f"rank {self.comm.rank}: {exc!r}"
            finally:
                job.release()
            errors = [e for e in self.comm.all_gather_object(error) if e is not None]
            if errors:
                self._remove(job.step)
                LOG.warning(
                    "Recovery checkpoint of step %d was discarded: %s",
                    job.step,
                    "; ".join(errors),
                )
                metrics["recovery_failed"] = 1
                self.metrics.append(metrics)
                return
            self._write_manifest(job)
            self.last_written = job.step
            metrics["recovery_write_seconds"] = time.perf_counter() - started
            metrics["recovery_rotated"] = self._rotate()
        if job.consolidate and self.consolidate_dir is not None:
            started = time.perf_counter()
            error = None
            try:
                self._consolidate(job.step)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                error = f"rank {self.comm.rank}: {exc!r}"
            errors = [e for e in self.comm.all_gather_object(error) if e is not None]
            if errors:
                LOG.warning(
                    "Consolidating recovery checkpoint %d failed: %s",
                    job.step,
                    "; ".join(errors),
                )
                metrics["recovery_consolidate_failed"] = 1
            else:
                self.last_consolidated = job.step
                metrics["recovery_consolidate_seconds"] = time.perf_counter() - started
        self.metrics.append(metrics)

    def _write(self, job: RecoveryJob) -> int:
        assert job.lora is not None
        target = step_dir(self.directory, job.step)
        target.mkdir(parents=True, exist_ok=True)
        (target / manifest_name(self.comm.rank)).unlink(missing_ok=True)
        buffers = job.lora()
        names = [piece.key for piece in self.pieces]
        specs = {
            piece.key: (buffer.dtype, tuple(buffer.shape))
            for piece, buffer in zip(self.pieces, buffers, strict=True)
        }
        written = write_safetensors(
            target / shard_name(self.comm.rank), names, specs, buffers
        )
        optimizer_path = target / optimizer_name(self.comm.rank)
        if job.optimizer is None:
            optimizer_path.unlink(missing_ok=True)
            return written
        skeleton, tensors = job.optimizer.ready()
        staged = optimizer_path.with_name(f".{optimizer_path.name}.tmp")
        with open(staged, "wb") as handle:
            torch.save({"skeleton": skeleton, "tensors": list(tensors)}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, optimizer_path)
        return written + optimizer_path.stat().st_size

    def _write_manifest(self, job: RecoveryJob):
        target = step_dir(self.directory, job.step)
        manifest = {
            "format": MANIFEST_FORMAT,
            "step": job.step,
            "rank": self.comm.rank,
            "world": self.comm.world,
            "optimizer": job.optimizer is not None,
            "pieces": [_piece_json(piece) for piece in self.pieces],
        }
        path = target / manifest_name(self.comm.rank)
        staged = path.with_name(f".{path.name}.tmp")
        staged.write_text(json.dumps(manifest), encoding="utf-8")
        os.replace(staged, path)

    def _remove(self, step: int):
        target = step_dir(self.directory, step)
        for name in (
            manifest_name(self.comm.rank),
            shard_name(self.comm.rank),
            optimizer_name(self.comm.rank),
        ):
            (target / name).unlink(missing_ok=True)
        try:
            target.rmdir()
        except OSError:
            pass

    def _rotate(self) -> int:
        steps = complete_steps(self.directory, self.comm.rank)
        kept = set(steps[-self.keep :])
        with_optimizer = [
            step
            for step in steps
            if (
                step_dir(self.directory, step) / optimizer_name(self.comm.rank)
            ).exists()
        ]
        if with_optimizer:
            kept.add(with_optimizer[-1])
        removed = 0
        for step in steps:
            if step not in kept:
                self._remove(step)
                removed += 1
        return removed

    def _consolidate(self, step: int):
        if self._all_pieces is None:
            self._all_pieces = self.comm.all_gather_object(self.pieces)
        all_pieces = self._all_pieces
        rank = self.comm.rank
        data = _ShardData(
            step_dir(self.directory, step) / shard_name(rank), all_pieces[rank]
        )
        shapes = global_shapes(all_pieces)
        names = sorted(shapes)
        views = {
            piece.key: piece.view for rank_pieces in all_pieces for piece in rank_pieces
        }

        def assembled():
            for name in names:
                need = Need(name, views[name], full_box(views[name]))
                needs = [[need] if r == 0 else [] for r in range(self.comm.world)]
                transfers = plan_transfers(all_pieces, needs)
                out = assemble(
                    self.comm, rank, data, needs[rank], transfers, all_pieces
                )
                yield out[0].reshape(shapes[name][1]) if rank == 0 else None

        if rank != 0:
            for _ in assembled():
                pass
            return
        assert self.consolidate_dir is not None
        final = self.consolidate_dir / f"checkpoint-{step}"
        staged = self.consolidate_dir / f".checkpoint-{step}.tmp"
        shutil.rmtree(staged, ignore_errors=True)
        staged.mkdir(parents=True)
        specs = {
            name: (getattr(torch, dtype), shape)
            for name, (dtype, shape) in shapes.items()
        }
        write_safetensors(staged / ADAPTER_NAME, names, specs, assembled())
        if self.adapter_config is not None:
            (staged / ADAPTER_CONFIG_NAME).write_text(
                json.dumps(self.adapter_config, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        shutil.rmtree(final, ignore_errors=True)
        os.replace(staged, final)


def complete_steps(directory: Path | str, rank: int) -> list[int]:
    """Steps for which ``rank``'s part of a complete checkpoint exists, oldest first."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    steps = []
    for entry in directory.iterdir():
        if entry.name.startswith("step-") and (entry / manifest_name(rank)).exists():
            steps.append(int(entry.name.removeprefix("step-")))
    return sorted(steps)


def latest_common_step(directory: Path | str, rank: int | None = None) -> int | None:
    """The newest step every rank holds complete (agreed over the default group)."""
    distributed = dist.is_available() and dist.is_initialized()
    if rank is None:
        rank = dist.get_rank() if distributed else 0
    steps = set(complete_steps(directory, rank))
    if distributed:
        gathered: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, sorted(steps))
        steps = set.intersection(*(set(s) for s in gathered))
    return max(steps) if steps else None


def load_recovery_checkpoint(
    directory: Path | str,
    model: torch.nn.Module,
    *,
    step: int | None = None,
    optimizer=None,
    adapter: str = "default",
) -> int:
    """Copy this rank's saved LoRA slices (and optimizer state) into the live model.

    The model must be sharded exactly as when the checkpoint was written; the
    pieces recorded in the manifest are checked against the live ones.
    """
    from .lora_shards import collect_model_pieces

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    directory = Path(directory)
    if step is None:
        step = latest_common_step(directory, rank)
        if step is None:
            raise FileNotFoundError(f"No complete recovery checkpoint in {directory}")
    target = step_dir(directory, step)
    manifest = json.loads((target / manifest_name(rank)).read_text(encoding="utf-8"))
    if manifest["world"] != world:
        raise ValueError(
            f"Recovery checkpoint {step} was written by {manifest['world']} ranks, "
            f"not {world}"
        )
    live = collect_model_pieces(model, adapter)
    if [_piece_json(piece) for piece, _ in live] != manifest["pieces"]:
        raise ValueError(
            f"Recovery checkpoint {step} was written with a different adapter sharding"
        )
    with safe_open(str(target / shard_name(rank)), framework="pt") as handle:
        with torch.no_grad():
            for piece, tensor in live:
                tensor.copy_(handle.get_tensor(piece.key))
    if optimizer is not None:
        if not manifest["optimizer"]:
            raise ValueError(f"Recovery checkpoint {step} has no optimizer state")
        load_optimizer_state(target / optimizer_name(rank), optimizer)
    return step
