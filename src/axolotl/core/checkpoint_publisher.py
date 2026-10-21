"""Ordered background publication of immutable checkpoint directories."""

import errno
import os
import shutil
import tempfile
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path


def retain_checkpoint(
    source: Path | str, root: Path | str, *, files: Iterable[str] | None = None
) -> Path:
    """Pin a checkpoint tree, or selected relative files, against checkpoint rotation.

    Callers must submit completed saves whose contents will not be overwritten.
    Hard links avoid copying weights; unsupported filesystems fall back to copying.
    """
    source, root = Path(source), Path(root)
    names = (
        list(files)
        if files is not None
        else [
            str(path.relative_to(source))
            for path in source.rglob("*")
            if path.is_file()
        ]
    )
    if not source.is_dir() or not names:
        raise ValueError("Checkpoint must be a nonempty directory")
    for name in names:
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not (source / path).is_file()
            or not (source / path).resolve().is_relative_to(source.resolve())
        ):
            raise ValueError(f"Invalid checkpoint file: {name}")
    root.mkdir(parents=True, exist_ok=True)
    retained = Path(tempfile.mkdtemp(prefix="queued-", dir=root))
    try:
        for name in set(names):
            destination = retained / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (source / name).is_symlink():
                shutil.copy2(source / name, destination)
                continue
            try:
                os.link(source / name, destination)
            except OSError as exc:
                if exc.errno not in {
                    errno.EXDEV,
                    errno.EPERM,
                    errno.EOPNOTSUPP,
                    errno.ENOSYS,
                }:
                    raise
                shutil.copy2(source / name, destination)
    except BaseException:
        shutil.rmtree(retained)
        raise
    return retained


class BackgroundCheckpointPublisher:
    """Publish retained directories in order without blocking the training thread.

    ``publish(directory, step)`` runs on one worker and may upload files directly
    or transform them first. ``retain(source, queue_root)`` runs synchronously and
    must return a private directory owned by this queue. Queued directories are
    removed after publication, failure, or cancellation. Call ``check`` at training
    boundaries and ``close`` to drain work, or cancel pending work during teardown.
    Submission and polling belong to a single producer thread.
    """

    def __init__(
        self,
        publish: Callable[[Path, int], object],
        directory: Path | str,
        *,
        retain: Callable[[Path, Path], Path] = retain_checkpoint,
        thread_name_prefix: str = "checkpoint-publish",
    ):
        self.publish, self.retain = publish, retain
        self.directory = Path(directory)
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=thread_name_prefix
        )
        self.futures: deque[Future] = deque()
        self.closed = False

    def check(self):
        while self.futures and self.futures[0].done():
            future = self.futures[0]
            # Keep a failed future at the head so every subsequent check fails.
            future.result()
            self.futures.popleft()

    def submit(self, directory, step):
        if self.closed:
            raise RuntimeError("Checkpoint publisher is closed")
        self.check()
        retained = self.retain(Path(directory), self.directory)
        previous = self.futures[-1] if self.futures else None
        try:
            future = self.executor.submit(self._publish, retained, step, previous)
        except BaseException:
            shutil.rmtree(retained)
            raise
        self.futures.append(future)
        future.add_done_callback(
            lambda completed: shutil.rmtree(retained) if completed.cancelled() else None
        )

    def _publish(self, directory, step, previous):
        try:
            if previous is not None:
                previous.result()
            return self.publish(directory, step)
        finally:
            shutil.rmtree(directory)

    def close(self, *, cancel_pending=False):
        self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=cancel_pending)
        if not cancel_pending:
            self.check()
