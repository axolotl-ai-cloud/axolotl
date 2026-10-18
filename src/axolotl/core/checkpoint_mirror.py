"""Mirror checkpoints from a fast local output_dir to a shared filesystem in the background.

Training saves to local storage and continues while each ``checkpoint-N`` is copied to the
mirror. Copies of one checkpoint from several nodes land in a hidden staging directory and
appear as ``checkpoint-N`` only once every node's share is there, so a resume never sees a
partial checkpoint. A failed copy is logged and never stops training.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from transformers import TrainerCallback

from axolotl.core.checkpoint_publisher import (
    BackgroundCheckpointPublisher,
    QueueFull,
    retain_checkpoint,
)
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
    # shared by every rank of one training run; scopes staging so a restarted run
    # never completes a checkpoint with another run's shares
    attempt: str = ""

    @property
    def node_local(self) -> bool:
        return self.num_nodes > 1


def layout_from(
    hostnames: list[str], rank: int, *, shared_output_dir: bool, attempt: str = ""
) -> MirrorLayout:
    """Layout for ``rank`` given every rank's hostname, indexed by global rank."""
    nodes = list(dict.fromkeys(hostnames))
    if len(nodes) == 1 or shared_output_dir:
        return MirrorLayout(uploader=rank == 0, attempt=attempt)
    host = hostnames[rank]
    return MirrorLayout(
        uploader=rank == hostnames.index(host),
        node_index=nodes.index(host),
        num_nodes=len(nodes),
        attempt=attempt,
    )


_LAYOUTS: dict[str, MirrorLayout] = {}


def detect_layout(output_dir: Path | str) -> MirrorLayout:
    """Find the nodes and whether they share ``output_dir``. Collective: call on every rank."""
    import torch.distributed as dist

    key = str(Path(output_dir).resolve())
    if key in _LAYOUTS:
        return _LAYOUTS[key]
    if not (dist.is_available() and dist.is_initialized()):
        _LAYOUTS[key] = MirrorLayout(uploader=True, attempt=uuid.uuid4().hex[:12])
        return _LAYOUTS[key]
    rank, world = dist.get_rank(), dist.get_world_size()
    hostnames: list = [None] * world
    dist.all_gather_object(hostnames, socket.gethostname())
    run = [uuid.uuid4().hex[:12] if rank == 0 else None]
    dist.broadcast_object_list(run, src=0)
    attempt = str(run[0])
    shared = len(set(hostnames)) == 1
    if not shared:
        probe_path = Path(output_dir) / f"{PROBE_PREFIX}{attempt}"
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
    layout = layout_from(hostnames, rank, shared_output_dir=shared, attempt=attempt)
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
    every node agrees on, so concurrent runs are idempotent. One training run writes to
    a mirror directory at a time.
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
        self.staging_prefix = f"{STAGING_PREFIX}{self.layout.attempt or 'local'}-"
        self.worker = BackgroundCheckpointPublisher(
            self._upload,
            self.local_dir / QUEUE_DIRNAME,
            retain=_retain,
            thread_name_prefix="checkpoint-mirror",
            max_pending=max_pending,
        )

    def submit(self, step: int) -> None:
        """Queue ``checkpoint-<step>``; never blocks the training thread."""
        source = self.local_dir / f"{PREFIX}-{step}"
        try:
            # only the uploader ranks get here, so waiting could outlast the other
            # ranks' next collective; skip instead and retry the newest at the end
            self.worker.submit(source, step, block=False)
        except QueueFull as exc:
            self.failed.add(step)
            LOG.warning(
                f"Not mirroring {source.name}: {exc}. The newest checkpoint is "
                "retried at the end of training."
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.failed.add(step)
            LOG.warning(f"Not mirroring {source}: {exc}")

    def clear_other_attempts(self) -> None:
        """Remove staging and set-aside copies left by earlier runs. Call before any copy."""
        for path in self.mirror_dir.glob(f"{STAGING_PREFIX}*"):
            if not path.name.startswith(self.staging_prefix):
                shutil.rmtree(path, ignore_errors=True)
        for path in self.mirror_dir.glob(".replaced-*"):
            shutil.rmtree(path, ignore_errors=True)

    def _upload(self, retained: Path, step: int) -> bool:
        name = f"{PREFIX}-{step}"
        staging = self.mirror_dir / f"{self.staging_prefix}{name}"
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
                self.mirror_dir.glob(f"{self.staging_prefix}{PREFIX}-*"),
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
        lock = markers / FINALIZE_LOCK
        try:
            # one node wins the exclusive create and publishes; the others leave it be
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            return False
        final = self.mirror_dir / staging.name[len(self.staging_prefix) :]
        replaced = None
        try:
            if final.exists():
                replaced = (
                    self.mirror_dir / f".replaced-{final.name}-{uuid.uuid4().hex}"
                )
                os.rename(final, replaced)
            os.rename(staging, final)
        except OSError as exc:
            if replaced is not None and not final.exists():
                with contextlib.suppress(OSError):
                    os.rename(replaced, final)
            # release the lock so the next settle (or the final drain) retries
            with contextlib.suppress(OSError):
                lock.unlink()
            LOG.warning(f"Publishing {final} failed and will be retried: {exc}")
            return False
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
        for staging in self.mirror_dir.glob(f"{STAGING_PREFIX}*"):
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

    def drain(self) -> None:
        """Wait for queued copies, publish what is complete, and retry the newest
        checkpoint once if its copy failed or was skipped. Blocks."""
        self.worker.wait()
        self._settle()
        local = checkpoint_steps(self.local_dir)
        if local and max(local) in self.failed:
            LOG.info(f"Retrying the mirror copy of {local[max(local)]}")
            self.submit(max(local))
            self.worker.wait()

    def finish(self) -> None:
        """Drain, then (node 0) copy the rest of ``output_dir``: final weights,
        tokenizer, configs. Other nodes' copies of those files are not authoritative."""
        self.drain()
        if self.layout.node_index == 0:
            self._copy_final_outputs()
        self.worker.close()

    def _copy_final_outputs(self) -> None:
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


def _retain(source: Path, root: Path) -> Path:
    """Retain a checkpoint; an empty one (a node whose ranks wrote nothing, e.g. with
    save_only_model) still queues, so the node records its share as done."""
    if source.is_dir() and not any(source.iterdir()):
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="queued-", dir=root))
    return retain_checkpoint(source, root)


def _barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


class CheckpointMirrorCallback(TrainerCallback):
    """Mirror each saved checkpoint to a shared directory without blocking training.

    With ``delete="mirror"`` the mirror keeps as many checkpoints as ``save_total_limit``
    keeps locally; with ``delete="keep"`` it keeps everything, or ``save_total_limit``
    checkpoints when given. Call ``drain`` before and ``finish`` after the final save.
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
            if self.layout.node_index == 0:
                self.mirror.clear_other_attempts()
            LOG.info(
                f"Mirroring checkpoints from {args.output_dir} to {self.mirror_dir} "
                f"(node {self.layout.node_index + 1} of {self.layout.num_nodes})"
            )
        # no node copies before node 0 has cleared earlier runs' staging
        _barrier()
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

    def drain(self) -> None:
        _barrier()
        if self.mirror is not None:
            self.mirror.drain()
        # every node's share is copied and published once all ranks get past here
        _barrier()

    def finish(self) -> None:
        _barrier()
        if self.mirror is not None:
            self.mirror.finish()
            self.mirror = None


def _mirror_callbacks(trainer) -> list[CheckpointMirrorCallback]:
    return [
        callback
        for callback in trainer.callback_handler.callbacks
        if isinstance(callback, CheckpointMirrorCallback)
    ]


def drain_checkpoint_mirror(trainer) -> None:
    """Finish pending checkpoint copies before the final save. Call on every rank."""
    for callback in _mirror_callbacks(trainer):
        callback.drain()


def finish_checkpoint_mirror(trainer) -> None:
    """Copy the final outputs and drain pending checkpoint copies. Call on every rank."""
    for callback in _mirror_callbacks(trainer):
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


def merge_source_checkpoint(
    output_dir: Path | str,
    mirror_dir: Path | str,
    local_last: str | None,
    *,
    timeout: float = 60.0,
) -> str | None:
    """Checkpoint the final sharded-weights merge should read.

    With a node-local output_dir across nodes, the local checkpoint holds only this
    node's shards, so read the mirrored copy of the same step once it is visible.
    """
    layout = detect_layout(output_dir)
    step = _step_of(local_last)
    if not layout.node_local or step is None:
        return local_last
    deadline = time.monotonic() + timeout
    while True:
        # listing the directory refreshes a network filesystem's cached view of it
        mirrored = checkpoint_steps(mirror_dir).get(step)
        if mirrored is not None:
            return str(mirrored)
        if time.monotonic() >= deadline:
            break
        time.sleep(2)
    LOG.warning(
        f"{PREFIX}-{step} is not in {mirror_dir}; merging from {local_last}, which "
        "holds only this node's shards"
    )
    return local_last
