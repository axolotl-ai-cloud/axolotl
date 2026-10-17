"""Mirror checkpoints from a fast local output_dir to a shared filesystem in the background.

Training saves to local storage and continues while each ``checkpoint-N`` is copied to the
mirror. Copies of one checkpoint from several nodes land in a hidden staging directory and
appear as ``checkpoint-N`` only once every node's share is there, so a resume never sees a
partial checkpoint. A failed copy is logged and never stops training.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from transformers import TrainerCallback

from axolotl.core.checkpoint_publisher import BackgroundCheckpointPublisher
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

PREFIX = "checkpoint"
QUEUE_DIRNAME = ".checkpoint-mirror-queue"
PROBE_PREFIX = ".checkpoint-mirror-probe-"
STAGING_PREFIX = ".staging-"
MARKERS_DIRNAME = ".mirror-nodes"
FINALIZE_LOCK = "finalizing"


def checkpoint_steps(root: Path | str) -> dict[int, Path]:
    """``checkpoint-N`` directories directly under ``root``, keyed by step."""
    root = Path(root)
    steps: dict[int, Path] = {}
    if not root.is_dir():
        return steps
    for path in root.glob(f"{PREFIX}-*"):
        suffix = path.name[len(PREFIX) + 1 :]
        if suffix.isdigit() and path.is_dir():
            steps[int(suffix)] = path
    return steps


def _step_of(name: str | None) -> int | None:
    if not name:
        return None
    suffix = Path(name).name.rpartition("-")[2]
    return int(suffix) if suffix.isdigit() else None


@dataclass(frozen=True)
class MirrorLayout:
    """Which process uploads, and how many node-local shares make one checkpoint."""

    uploader: bool
    node_index: int = 0
    num_nodes: int = 1

    @property
    def node_local(self) -> bool:
        return self.num_nodes > 1


def layout_from(
    hostnames: list[str], rank: int, *, shared_output_dir: bool
) -> MirrorLayout:
    """Layout for ``rank`` given every rank's hostname, indexed by global rank."""
    nodes = list(dict.fromkeys(hostnames))
    if len(nodes) == 1 or shared_output_dir:
        return MirrorLayout(uploader=rank == 0)
    host = hostnames[rank]
    return MirrorLayout(
        uploader=rank == hostnames.index(host),
        node_index=nodes.index(host),
        num_nodes=len(nodes),
    )


_LAYOUTS: dict[str, MirrorLayout] = {}


def detect_layout(output_dir: Path | str) -> MirrorLayout:
    """Find the nodes and whether they share ``output_dir``. Collective: call on every rank."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return MirrorLayout(uploader=True)
    key = str(Path(output_dir).resolve())
    if key in _LAYOUTS:
        return _LAYOUTS[key]
    rank, world = dist.get_rank(), dist.get_world_size()
    hostnames: list = [None] * world
    dist.all_gather_object(hostnames, socket.gethostname())
    shared = len(set(hostnames)) == 1
    if not shared:
        probe = [f"{PROBE_PREFIX}{uuid.uuid4().hex}" if rank == 0 else None]
        dist.broadcast_object_list(probe, src=0)
        probe_path = Path(output_dir) / str(probe[0])
        if rank == 0:
            probe_path.parent.mkdir(parents=True, exist_ok=True)
            probe_path.touch()
        dist.barrier()
        seen: list = [None] * world
        dist.all_gather_object(seen, probe_path.exists())
        dist.barrier()
        if rank == 0:
            probe_path.unlink(missing_ok=True)
        shared = all(seen)
    layout = layout_from(hostnames, rank, shared_output_dir=shared)
    _LAYOUTS[key] = layout
    return layout


def copy_file(source: Path, destination: Path, tag: str) -> int:
    """Copy through a temporary name so readers never see a partial file."""
    tmp = destination.with_name(f".{destination.name}.{tag}.tmp")
    try:
        shutil.copyfile(source, tmp)
        os.replace(tmp, destination)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return source.stat().st_size


def copy_tree(source: Path, destination: Path, *, tag: str, threads: int) -> int:
    """Copy every file under ``source`` into ``destination``; returns bytes copied."""
    files = [path for path in source.rglob("*") if path.is_file()]
    destination.mkdir(parents=True, exist_ok=True)
    for path in files:
        (destination / path.relative_to(source)).parent.mkdir(
            parents=True, exist_ok=True
        )
    with ThreadPoolExecutor(
        max_workers=threads, thread_name_prefix="checkpoint-mirror-copy"
    ) as pool:
        return sum(
            pool.map(
                lambda path: copy_file(
                    path, destination / path.relative_to(source), tag
                ),
                files,
            )
        )


def _trim(checkpoints: dict[int, Path], keep: int, best: int | None) -> list[Path]:
    """HF's retention: keep the newest ``keep`` plus the newest and best checkpoints."""
    ordered = sorted(checkpoints)
    protected = {ordered[-1]} if ordered else set()
    if best in checkpoints:
        protected.add(best)
    removable = [step for step in ordered if step not in protected]
    excess = len(ordered) - max(keep, len(protected))
    return [checkpoints[step] for step in removable[: max(excess, 0)]]


class CheckpointMirror:
    """Copy local ``checkpoint-N`` directories to ``mirror_dir`` in order, off the training thread.

    ``keep`` limits the mirror to the newest complete checkpoints (plus the best one);
    ``None`` keeps them all. Any node may publish or rotate: both only act on state that
    every node agrees on, so concurrent runs are idempotent.
    """

    def __init__(
        self,
        local_dir: Path | str,
        mirror_dir: Path | str,
        *,
        layout: MirrorLayout | None = None,
        keep: int | None = None,
        max_pending: int = 2,
        copy_threads: int = 4,
    ):
        self.local_dir, self.mirror_dir = Path(local_dir), Path(mirror_dir)
        self.layout = layout or MirrorLayout(uploader=True)
        self.keep = keep
        self.copy_threads = copy_threads
        self.best_checkpoint: str | None = None
        self.failed: set[int] = set()
        self.tag = f"node{self.layout.node_index}"
        self.worker = BackgroundCheckpointPublisher(
            self._upload,
            self.local_dir / QUEUE_DIRNAME,
            thread_name_prefix="checkpoint-mirror",
            max_pending=max_pending,
        )

    def submit(self, step: int) -> None:
        source = self.local_dir / f"{PREFIX}-{step}"
        try:
            self.worker.submit(source, step)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.failed.add(step)
            LOG.warning(f"Not mirroring {source}: {exc}")

    def _upload(self, retained: Path, step: int) -> bool:
        name = f"{PREFIX}-{step}"
        staging = self.mirror_dir / f"{STAGING_PREFIX}{name}"
        start = time.monotonic()
        try:
            nbytes = copy_tree(
                retained, staging, tag=self.tag, threads=self.copy_threads
            )
            markers = staging / MARKERS_DIRNAME
            markers.mkdir(parents=True, exist_ok=True)
            marker = markers / f"node-{self.layout.node_index}.json"
            tmp = markers / f".{marker.name}.tmp"
            tmp.write_text(json.dumps({"bytes": nbytes}), encoding="utf-8")
            os.replace(tmp, marker)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.failed.add(step)
            LOG.warning(
                f"Mirroring {name} to {self.mirror_dir} failed; training continues and "
                f"the local checkpoint is unaffected: {exc}"
            )
            return False
        self.failed.discard(step)
        LOG.info(
            f"Copied {name} ({nbytes / 2**30:.2f} GiB) to {self.mirror_dir} in "
            f"{time.monotonic() - start:.1f}s"
        )
        self._settle()
        return True

    def _settle(self) -> None:
        """Publish complete checkpoints, rotate the mirror, and drop dead staging."""
        try:
            for staging in sorted(
                self.mirror_dir.glob(f"{STAGING_PREFIX}{PREFIX}-*"),
                key=lambda path: _step_of(path.name) or -1,
            ):
                self._finalize(staging)
            if self.keep is not None:
                for path in _trim(
                    checkpoint_steps(self.mirror_dir),
                    self.keep,
                    _step_of(self.best_checkpoint),
                ):
                    shutil.rmtree(path, ignore_errors=True)
                    LOG.info(f"Removed {path} from the checkpoint mirror")
            self._drop_dead_staging()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOG.warning(
                f"Checkpoint mirror bookkeeping in {self.mirror_dir} failed: {exc}"
            )

    def _finalize(self, staging: Path) -> bool:
        markers = staging / MARKERS_DIRNAME
        if len(list(markers.glob("node-*.json"))) < self.layout.num_nodes:
            return False
        try:
            # one node wins the exclusive create and publishes; the others leave it be
            os.close(os.open(markers / FINALIZE_LOCK, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            return False
        final = self.mirror_dir / staging.name[len(STAGING_PREFIX) :]
        replaced = None
        if final.exists():
            replaced = self.mirror_dir / f".replaced-{final.name}-{uuid.uuid4().hex}"
            os.rename(final, replaced)
        os.rename(staging, final)
        if replaced is not None:
            shutil.rmtree(replaced, ignore_errors=True)
        LOG.info(f"{final} is complete")
        return True

    def _drop_dead_staging(self) -> None:
        # each node copies in step order, so staging older than the newest complete
        # checkpoint can only belong to a node whose copy failed
        complete = checkpoint_steps(self.mirror_dir)
        if not complete:
            return
        newest = max(complete)
        for staging in self.mirror_dir.glob(f"{STAGING_PREFIX}{PREFIX}-*"):
            step = _step_of(staging.name)
            if step is not None and step < newest:
                shutil.rmtree(staging, ignore_errors=True)

    def rotate_local(self, save_total_limit: int | None) -> None:
        """Apply HF's retention to ``local_dir``; HF only rotates on the global main process."""
        if not save_total_limit or save_total_limit <= 0:
            return
        for path in _trim(
            checkpoint_steps(self.local_dir),
            save_total_limit,
            _step_of(self.best_checkpoint),
        ):
            shutil.rmtree(path, ignore_errors=True)

    def finish(self) -> None:
        """Drain the queue, retry the newest checkpoint if its copy failed, and copy the
        rest of ``output_dir`` (final weights, tokenizer, configs). Blocks."""
        self.worker.wait()
        local = checkpoint_steps(self.local_dir)
        if local and max(local) in self.failed:
            LOG.info(f"Retrying the mirror copy of {local[max(local)]}")
            self.submit(max(local))
            self.worker.wait()
        try:
            for entry in self.local_dir.iterdir():
                if entry.name == QUEUE_DIRNAME or entry.name.startswith(PROBE_PREFIX):
                    continue
                if entry.is_dir():
                    if _step_of(entry.name) is not None and entry.name.startswith(
                        f"{PREFIX}-"
                    ):
                        continue
                    copy_tree(
                        entry,
                        self.mirror_dir / entry.name,
                        tag=self.tag,
                        threads=self.copy_threads,
                    )
                elif entry.is_file():
                    self.mirror_dir.mkdir(parents=True, exist_ok=True)
                    copy_file(entry, self.mirror_dir / entry.name, self.tag)
            LOG.info(
                f"Copied the final outputs in {self.local_dir} to {self.mirror_dir}"
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOG.warning(f"Copying the final outputs to {self.mirror_dir} failed: {exc}")
        self.worker.close()


def _barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


class CheckpointMirrorCallback(TrainerCallback):
    """Mirror each saved checkpoint to a shared directory without blocking training.

    With ``delete="mirror"`` the mirror keeps as many checkpoints as ``save_total_limit``
    keeps locally; with ``delete="keep"`` it keeps everything, or ``save_total_limit``
    checkpoints when given. Call ``finish`` after the final model save.
    """

    def __init__(
        self,
        mirror_dir: str,
        *,
        delete: str = "mirror",
        save_total_limit: int | None = None,
        max_pending: int = 2,
        copy_threads: int = 4,
    ):
        self.mirror_dir = mirror_dir
        self.delete = delete
        self.save_total_limit = save_total_limit
        self.max_pending, self.copy_threads = max_pending, copy_threads
        self.layout: MirrorLayout | None = None
        self.mirror: CheckpointMirror | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        if self.layout is None:
            self.layout = detect_layout(args.output_dir)
        if self.layout.uploader and self.mirror is None:
            keep = (
                args.save_total_limit
                if self.delete == "mirror"
                else self.save_total_limit
            )
            self.mirror = CheckpointMirror(
                args.output_dir,
                self.mirror_dir,
                layout=self.layout,
                keep=keep if keep and keep > 0 else None,
                max_pending=self.max_pending,
                copy_threads=self.copy_threads,
            )
            LOG.info(
                f"Mirroring checkpoints from {args.output_dir} to {self.mirror_dir} "
                f"(node {self.layout.node_index + 1} of {self.layout.num_nodes})"
            )
        return control

    def on_save(self, args, state, control, **kwargs):
        # every local rank writes its own shards and RNG state into the checkpoint
        _barrier()
        if self.mirror is None:
            return control
        self.mirror.best_checkpoint = state.best_model_checkpoint
        if self.layout is not None and self.layout.node_index > 0:
            self.mirror.rotate_local(args.save_total_limit)
        self.mirror.submit(state.global_step)
        return control

    def finish(self) -> None:
        _barrier()
        if self.mirror is not None:
            self.mirror.finish()
            self.mirror = None


def finish_checkpoint_mirror(trainer) -> None:
    """Copy the final outputs and drain pending checkpoint copies. Call on every rank."""
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, CheckpointMirrorCallback):
            callback.finish()


def resume_checkpoint(
    output_dir: Path | str, mirror_dir: Path | str, local_last: str | None
) -> str | None:
    """Newest resumable checkpoint across ``output_dir`` and the mirror. Collective."""
    layout = detect_layout(output_dir)
    remote = checkpoint_steps(mirror_dir)
    remote_last = str(remote[max(remote)]) if remote else None
    if layout.node_local:
        # each node holds only its own ranks' files, so only the mirror is complete
        if local_last and remote_last:
            LOG.info(
                f"output_dir is local to each node; resuming from {remote_last} in the mirror"
            )
        return remote_last
    if remote_last is None or local_last is None:
        return local_last or remote_last
    return local_last if (_step_of(local_last) or -1) >= max(remote) else remote_last
