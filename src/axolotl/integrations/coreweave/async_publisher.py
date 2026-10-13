"""Ordered background publication of completed on-disk checkpoints."""

import errno
import json
import os
import shutil
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def retain_checkpoint(source, root):
    """Pin saved files so Trainer checkpoint rotation cannot remove queued work."""
    source, root = Path(source), Path(root)
    index = source / "model.safetensors.index.json"
    if (source / "adapter_config.json").is_file():
        names = {"adapter_config.json", "adapter_model.safetensors"}
    elif index.is_file():
        manifest = json.loads(index.read_text(encoding="utf-8"))
        names = set(manifest["weight_map"].values()) | {index.name, "config.json"}
    else:
        names = {"model.safetensors", "config.json"}
    for name in names:
        if Path(name).name != name or not (source / name).is_file():
            raise ValueError(f"Invalid checkpoint file: {name}")
    root.mkdir(parents=True, exist_ok=True)
    retained = Path(tempfile.mkdtemp(prefix="queued-", dir=root))
    try:
        for name in names:
            try:
                os.link(source / name, retained / name)
            except OSError as exc:
                if exc.errno not in {
                    errno.EXDEV,
                    errno.EPERM,
                    errno.EOPNOTSUPP,
                    errno.ENOSYS,
                }:
                    raise
                shutil.copy2(source / name, retained / name)
    except BaseException:
        shutil.rmtree(retained)
        raise
    return retained


class BackgroundCheckpointPublisher:
    """One worker owns the delta chain; the queue holds disk paths, not tensors."""

    def __init__(self, factory, loader, directory):
        self.factory, self.loader = factory, loader
        self.directory = Path(directory)
        self.publisher = None
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="coreweave-publish"
        )
        self.futures = deque()
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
        retained = retain_checkpoint(directory, self.directory)
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
            if self.publisher is None:
                self.publisher = self.factory()
            weights, config = self.loader(directory)
            return self.publisher.publish(weights, config, step)
        finally:
            shutil.rmtree(directory)

    def close(self, *, cancel_pending=False):
        self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=cancel_pending)
        if not cancel_pending:
            self.check()
