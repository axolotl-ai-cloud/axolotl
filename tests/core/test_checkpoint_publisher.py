"""The shared queue works with file uploaders without any model dependencies."""

import errno
import shutil
import threading
from unittest.mock import MagicMock

import pytest

from axolotl.core.checkpoint_publisher import (
    BackgroundCheckpointPublisher,
    retain_checkpoint,
)


def checkpoint(root, step):
    source = root / f"checkpoint-{step}"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "weights.bin").write_bytes(bytes([step]))
    (source / "state.json").write_text("{}")
    return source


def test_file_uploader_is_ordered_and_survives_rotation(tmp_path):
    entered, release = threading.Event(), threading.Event()
    uploads = []
    producer = threading.get_ident()

    def upload(directory, step):
        assert threading.get_ident() != producer
        if step == 1:
            entered.set()
            assert release.wait(5)
        uploads.append((step, (directory / "nested" / "weights.bin").read_bytes()))

    worker = BackgroundCheckpointPublisher(upload, tmp_path / "queue")
    try:
        first = checkpoint(tmp_path, 1)
        worker.submit(first, 1)
        assert entered.wait(5)
        second = checkpoint(tmp_path, 2)
        worker.submit(second, 2)
        shutil.rmtree(first)
        shutil.rmtree(second)
        assert uploads == []
        release.set()
        worker.close()
        assert uploads == [(1, b"\x01"), (2, b"\x02")]
        assert not list((tmp_path / "queue").iterdir())
        with pytest.raises(RuntimeError, match="closed"):
            worker.submit(tmp_path, 3)
    finally:
        release.set()
        worker.close(cancel_pending=True)


def test_failure_blocks_following_uploads_and_remains_observable(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def fail(*args):
        entered.set()
        assert release.wait(5)
        raise OSError("upload failed")

    upload = MagicMock(side_effect=fail)
    worker = BackgroundCheckpointPublisher(upload, tmp_path / "queue")
    try:
        worker.submit(checkpoint(tmp_path, 1), 1)
        assert entered.wait(5)
        worker.submit(checkpoint(tmp_path, 2), 2)
        release.set()
        with pytest.raises(OSError, match="upload failed"):
            worker.close()
        with pytest.raises(OSError, match="upload failed"):
            worker.check()
        upload.assert_called_once()
        assert not list((tmp_path / "queue").iterdir())
    finally:
        release.set()
        worker.close(cancel_pending=True)


def test_custom_retention_and_copy_fallback(tmp_path, monkeypatch):
    import axolotl.core.checkpoint_publisher as module

    source = checkpoint(tmp_path, 1)
    linked = retain_checkpoint(source, tmp_path / "links", files=["nested/weights.bin"])
    assert (linked / "nested/weights.bin").stat().st_ino == (
        source / "nested/weights.bin"
    ).stat().st_ino
    monkeypatch.setattr(
        module.os, "link", MagicMock(side_effect=OSError(errno.EXDEV, "cross device"))
    )
    retained = retain_checkpoint(source, tmp_path / "copies")
    shutil.rmtree(source)
    assert (retained / "nested/weights.bin").read_bytes() == b"\x01"
    assert (retained / "state.json").read_text() == "{}"

    def retain(source, root):
        return retain_checkpoint(source, root, files=["nested/weights.bin"])

    seen = []
    worker = BackgroundCheckpointPublisher(
        lambda path, step: seen.append(
            sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file())
        ),
        tmp_path / "queue",
        retain=retain,
    )
    worker.submit(retained, 1)
    worker.close()
    assert seen == [["nested/weights.bin"]]


@pytest.mark.parametrize("filename", ["../escape", "/absolute", "missing"])
def test_retention_rejects_invalid_selection(tmp_path, filename):
    source = checkpoint(tmp_path, 1)
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        retain_checkpoint(source, tmp_path / "queue", files=[filename])
    assert not (tmp_path / "queue").exists()


def test_retention_materializes_internal_symlinks_and_rejects_external_ones(tmp_path):
    source = checkpoint(tmp_path, 1)
    (source / "weights.bin").symlink_to("nested/weights.bin")
    retained = retain_checkpoint(source, tmp_path / "queue", files=["weights.bin"])
    assert not (retained / "weights.bin").is_symlink()
    shutil.rmtree(source)
    assert (retained / "weights.bin").read_bytes() == b"\x01"
    source = checkpoint(tmp_path, 2)
    (source / "outside").symlink_to(retained / "weights.bin")
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        retain_checkpoint(source, tmp_path / "other")
