"""Publication must not hold up training or lose checkpoints to rotation."""

import errno
import shutil
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from axolotl.integrations.coreweave.async_publisher import (
    CoreWeaveBackgroundPublisher,
    retain_model_checkpoint,
)
from axolotl.integrations.coreweave.plugin import (
    CoreWeaveCheckpointCallback,
    load_weights,
)
from axolotl.utils.delta_checkpoint import write_checkpoint

from .test_checkpoints import config


def checkpoint(root, step):
    return write_checkpoint(
        {"weight": torch.tensor([step])},
        root / f"checkpoint-{step}",
        identity=f"step-{step}",
        config={"model_type": "test"},
    )


def test_save_returns_while_upload_is_blocked_and_survives_rotation(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []
    training_thread = threading.get_ident()

    def publish(weights, model_config, step):
        assert threading.get_ident() != training_thread
        if step == 1:
            entered.set()
            assert release.wait(5)
        calls.append((step, weights["weight"].item()))

    callback = CoreWeaveCheckpointCallback(config())
    callback.publisher = MagicMock(publish=publish)
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=1, is_world_process_zero=True)
    first = checkpoint(tmp_path, 1)
    try:
        callback.on_save(args, state, None)
        assert entered.wait(5)
        assert calls == []
        shutil.rmtree(first)
        second = checkpoint(tmp_path, 2)
        state.global_step = 2
        callback.on_save(args, state, None)
        shutil.rmtree(second)
        callback.on_step_end(args, state, None)
        assert calls == []
        release.set()
        callback.on_train_end(args, state, None)
        assert calls == [(1, 1), (2, 2)]
        assert not list((tmp_path / "coreweave-checkpoints" / ".queue").iterdir())
    finally:
        release.set()
        callback.close()


def test_first_failure_stops_later_publications_and_surfaces_on_step(tmp_path):
    entered, release = threading.Event(), threading.Event()
    publisher = MagicMock()

    def fail(*args):
        entered.set()
        assert release.wait(5)
        raise OSError("upload failed")

    publisher.publish.side_effect = fail
    callback = CoreWeaveCheckpointCallback(config())
    callback.publisher = publisher
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=1, is_world_process_zero=True)
    checkpoint(tmp_path, 1)
    checkpoint(tmp_path, 2)
    try:
        callback.on_save(args, state, None)
        assert entered.wait(5)
        state.global_step = 2
        callback.on_save(args, state, None)
        release.set()
        with pytest.raises(OSError, match="upload failed"):
            callback.worker.futures[-1].result(timeout=5)
        with pytest.raises(RuntimeError, match="upload failed"):
            callback.on_step_end(args, state, None)
        with pytest.raises(RuntimeError, match="upload failed"):
            callback.on_train_end(args, state, None)
        publisher.publish.assert_called_once()
        assert not list((tmp_path / "coreweave-checkpoints" / ".queue").iterdir())
    finally:
        release.set()
        callback.close()


def test_unload_cancels_queued_work_and_waits_for_active_upload(tmp_path):
    entered, release, canceled, closed = (threading.Event() for _ in range(4))
    publisher = MagicMock()

    def publish(*args):
        entered.set()
        assert release.wait(5)

    publisher.publish.side_effect = publish
    worker = CoreWeaveBackgroundPublisher(
        lambda: publisher, load_weights, tmp_path / "queue"
    )
    worker.submit(checkpoint(tmp_path, 1), 1)
    assert entered.wait(5)
    worker.submit(checkpoint(tmp_path, 2), 2)
    worker.futures[-1].add_done_callback(
        lambda future: canceled.set() if future.cancelled() else None
    )

    def close():
        worker.close(cancel_pending=True)
        closed.set()

    thread = threading.Thread(target=close)
    thread.start()
    try:
        assert canceled.wait(5)
        assert not closed.is_set()
    finally:
        release.set()
        thread.join(timeout=5)
    assert closed.is_set()
    publisher.publish.assert_called_once()
    assert not list((tmp_path / "queue").iterdir())


def test_retention_uses_links_without_copying_weights(tmp_path):
    source = checkpoint(tmp_path, 1)
    retained = retain_model_checkpoint(source, tmp_path / "queue")
    for path in retained.iterdir():
        assert path.stat().st_ino == (source / path.name).stat().st_ino


def test_retention_falls_back_to_copy_and_cleans_up_errors(tmp_path, monkeypatch):
    import axolotl.core.checkpoint_publisher as module

    source = checkpoint(tmp_path, 1)
    monkeypatch.setattr(
        module.os, "link", MagicMock(side_effect=OSError(errno.EXDEV, "cross device"))
    )
    retained = retain_model_checkpoint(source, tmp_path / "queue")
    assert load_weights(retained)[0]["weight"].item() == 1
    shutil.rmtree(retained)
    monkeypatch.setattr(
        module.os, "link", MagicMock(side_effect=OSError(errno.ENOSPC, "disk full"))
    )
    with pytest.raises(OSError, match="disk full"):
        retain_model_checkpoint(source, tmp_path / "queue")
    assert not list((tmp_path / "queue").iterdir())
