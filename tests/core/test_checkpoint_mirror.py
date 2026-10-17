"""Background checkpoint mirroring from a local output_dir to a shared directory."""

import logging
import threading
from types import SimpleNamespace

import pytest

from axolotl.core import checkpoint_mirror
from axolotl.core.checkpoint_mirror import (
    QUEUE_DIRNAME,
    CheckpointMirror,
    CheckpointMirrorCallback,
    MirrorLayout,
    checkpoint_steps,
    layout_from,
    resume_checkpoint,
)
from axolotl.utils.dict import DictDefault
from axolotl.utils.train import determine_last_checkpoint


def save(root, step, files=None):
    directory = root / f"checkpoint-{step}"
    for name, data in (files or {"model.safetensors": bytes([step])}).items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(data)
    return directory


def tree(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".mirror-nodes" not in path.parts
    }


@pytest.fixture
def dirs(tmp_path):
    local, remote = tmp_path / "nvme", tmp_path / "nfs"
    local.mkdir()
    return local, remote


def test_copies_a_complete_checkpoint_and_cleans_up(dirs):
    local, remote = dirs
    save(local, 1, {"model.safetensors": b"w", "nested/optimizer.pt": b"o"})
    mirror = CheckpointMirror(local, remote)

    mirror.submit(1)
    mirror.worker.wait()

    assert tree(remote / "checkpoint-1") == tree(local / "checkpoint-1")
    assert not list(remote.glob(".staging-*"))
    assert not [p for p in remote.rglob("*.tmp")]
    assert not list((local / QUEUE_DIRNAME).iterdir())
    mirror.worker.close()


def test_copy_survives_local_rotation(dirs, monkeypatch):
    local, remote = dirs
    entered, release = threading.Event(), threading.Event()
    copy_tree = checkpoint_mirror.copy_tree

    def gated(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return copy_tree(*args, **kwargs)

    monkeypatch.setattr(checkpoint_mirror, "copy_tree", gated)
    source = save(local, 3)
    mirror = CheckpointMirror(local, remote)
    mirror.submit(3)
    assert entered.wait(5)
    for path in source.iterdir():
        path.unlink()
    source.rmdir()
    release.set()
    mirror.worker.wait()

    assert (remote / "checkpoint-3" / "model.safetensors").read_bytes() == bytes([3])
    mirror.worker.close()


@pytest.mark.parametrize("first", [0, 1])
def test_multi_node_checkpoint_appears_only_once_every_node_copied(tmp_path, first):
    remote = tmp_path / "nfs"
    shares = {
        0: {"trainer_state.json": b"{}", "rng_state_0.pth": b"r0"},
        1: {"rng_state_1.pth": b"r1", "shard/__1_0.distcp": b"s1"},
    }
    mirrors = {}
    for node, files in shares.items():
        local = tmp_path / f"node{node}"
        save(local, 5, files)
        mirrors[node] = CheckpointMirror(
            local, remote, layout=MirrorLayout(True, node_index=node, num_nodes=2)
        )

    mirrors[first].submit(5)
    mirrors[first].worker.wait()
    assert checkpoint_steps(remote) == {}

    mirrors[1 - first].submit(5)
    mirrors[1 - first].worker.wait()
    assert tree(remote / "checkpoint-5") == {
        name: data for files in shares.values() for name, data in files.items()
    }
    assert not list(remote.glob(".staging-*"))
    for mirror in mirrors.values():
        mirror.worker.close()


@pytest.mark.parametrize("best, kept", [(None, [2, 3]), ("checkpoint-1", [1, 3])])
def test_mirror_rotation_keeps_newest_and_best(dirs, best, kept):
    local, remote = dirs
    mirror = CheckpointMirror(local, remote, keep=2)
    mirror.best_checkpoint = best and str(local / best)
    for step in (1, 2, 3):
        save(local, step)
        mirror.submit(step)
        mirror.worker.wait()

    assert sorted(checkpoint_steps(remote)) == kept
    mirror.worker.close()


def test_keep_none_keeps_every_checkpoint(dirs):
    local, remote = dirs
    mirror = CheckpointMirror(local, remote, keep=None)
    for step in (1, 2, 3):
        save(local, step)
        mirror.submit(step)
    mirror.worker.wait()

    assert sorted(checkpoint_steps(remote)) == [1, 2, 3]
    mirror.worker.close()


def test_copy_failure_warns_and_later_checkpoints_still_copy(dirs, monkeypatch, caplog):
    local, remote = dirs
    copy_file = checkpoint_mirror.copy_file

    def flaky(source, destination, tag):
        if destination.parent.name.endswith("checkpoint-2"):
            destination.with_name(f".{destination.name}.{tag}.tmp").write_bytes(b"x")
            raise OSError("stale NFS handle")
        return copy_file(source, destination, tag)

    monkeypatch.setattr(checkpoint_mirror, "copy_file", flaky)
    mirror = CheckpointMirror(local, remote, keep=1)
    with caplog.at_level(logging.WARNING):
        for step in (1, 2, 3):
            save(local, step)
            mirror.submit(step)
        mirror.worker.wait()

    assert sorted(checkpoint_steps(remote)) == [3]
    assert not list(remote.glob(".staging-*"))
    assert "stale NFS handle" in caplog.text
    assert mirror.failed == {2}
    mirror.worker.close()


def test_rotation_keeps_the_last_good_copy_when_newer_copies_fail(dirs, monkeypatch):
    local, remote = dirs
    copy_file = checkpoint_mirror.copy_file

    def fail_after_first(source, destination, tag):
        if not destination.parent.name.endswith("checkpoint-1"):
            raise OSError("nfs down")
        return copy_file(source, destination, tag)

    monkeypatch.setattr(checkpoint_mirror, "copy_file", fail_after_first)
    mirror = CheckpointMirror(local, remote, keep=1)
    for step in (1, 2, 3):
        save(local, step)
        mirror.submit(step)
    mirror.worker.wait()

    assert sorted(checkpoint_steps(remote)) == [1]
    mirror.worker.close()


def test_finish_retries_a_failed_newest_checkpoint_and_copies_final_outputs(
    dirs, monkeypatch
):
    local, remote = dirs
    copy_file = checkpoint_mirror.copy_file
    attempts = []

    def fail_once(source, destination, tag):
        if destination.parent.name.endswith("checkpoint-2") and not attempts:
            attempts.append(1)
            raise OSError("transient")
        return copy_file(source, destination, tag)

    monkeypatch.setattr(checkpoint_mirror, "copy_file", fail_once)
    mirror = CheckpointMirror(local, remote)
    for step in (1, 2):
        save(local, step)
        mirror.submit(step)
    (local / "adapter_model.safetensors").write_bytes(b"final")
    (local / "merged").mkdir()
    (local / "merged" / "model.safetensors").write_bytes(b"merged")

    mirror.worker.wait()
    assert mirror.failed == {2}

    mirror.finish()

    assert attempts == [1]
    assert sorted(checkpoint_steps(remote)) == [1, 2]
    assert (remote / "adapter_model.safetensors").read_bytes() == b"final"
    assert (remote / "merged" / "model.safetensors").read_bytes() == b"merged"
    assert not (remote / QUEUE_DIRNAME).exists()
    assert mirror.worker.closed


def test_max_pending_blocks_saves_until_the_oldest_copy_finishes(dirs, monkeypatch):
    local, remote = dirs
    entered, release = threading.Event(), threading.Event()
    copy_tree = checkpoint_mirror.copy_tree

    calls = []

    def gated(source, *args, **kwargs):
        calls.append(source)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        return copy_tree(source, *args, **kwargs)

    monkeypatch.setattr(checkpoint_mirror, "copy_tree", gated)
    mirror = CheckpointMirror(local, remote, max_pending=1)
    save(local, 1)
    save(local, 2)
    mirror.submit(1)
    assert entered.wait(5)
    second = threading.Thread(target=mirror.submit, args=(2,))
    second.start()
    second.join(0.3)
    assert second.is_alive()

    release.set()
    second.join(5)
    assert not second.is_alive()
    mirror.worker.wait()
    assert sorted(checkpoint_steps(remote)) == [1, 2]
    mirror.worker.close()


def test_rotate_local_applies_save_total_limit(dirs):
    local, remote = dirs
    for step in (1, 2, 3, 4):
        save(local, step)
    mirror = CheckpointMirror(local, remote)
    mirror.best_checkpoint = str(local / "checkpoint-1")

    mirror.rotate_local(2)

    assert sorted(checkpoint_steps(local)) == [1, 4]
    mirror.worker.close()


def test_layout_from_hostnames():
    single = ["a", "a", "a", "a"]
    assert layout_from(single, 0, shared_output_dir=False) == MirrorLayout(True)
    assert layout_from(single, 2, shared_output_dir=False) == MirrorLayout(False)

    two = ["a", "a", "b", "b"]
    assert layout_from(two, 2, shared_output_dir=False) == MirrorLayout(True, 1, 2)
    assert layout_from(two, 3, shared_output_dir=False) == MirrorLayout(False, 1, 2)
    assert layout_from(two, 2, shared_output_dir=True) == MirrorLayout(False)
    assert layout_from(two, 0, shared_output_dir=True) == MirrorLayout(True)


@pytest.mark.parametrize(
    "local_steps, remote_steps, node_local, expected",
    [
        ([3], [5], False, "nfs/checkpoint-5"),
        ([5], [5], False, "nvme/checkpoint-5"),
        ([6], [5], False, "nvme/checkpoint-6"),
        ([], [5], False, "nfs/checkpoint-5"),
        ([4], [], False, "nvme/checkpoint-4"),
        ([6], [5], True, "nfs/checkpoint-5"),
        ([6], [], True, None),
    ],
)
def test_resume_prefers_the_newest_complete_checkpoint(
    tmp_path, monkeypatch, local_steps, remote_steps, node_local, expected
):
    local, remote = tmp_path / "nvme", tmp_path / "nfs"
    for step in local_steps:
        save(local, step)
    for step in remote_steps:
        save(remote, step)
    (remote / ".staging-checkpoint-9").mkdir(parents=True)
    layout = MirrorLayout(True, 0, 2 if node_local else 1)
    monkeypatch.setattr(checkpoint_mirror, "detect_layout", lambda _: layout)
    local_last = str(local / f"checkpoint-{max(local_steps)}") if local_steps else None

    chosen = resume_checkpoint(local, remote, local_last)

    assert chosen == (expected and str(tmp_path / expected))


def test_auto_resume_uses_the_mirror(tmp_path):
    local, remote = tmp_path / "nvme", tmp_path / "nfs"
    save(local, 2)
    save(remote, 4)
    cfg = DictDefault(
        output_dir=str(local),
        auto_resume_from_checkpoints=True,
        resume_from_checkpoint=None,
        checkpoint_mirror={"dir": str(remote), "resume": True},
    )

    assert determine_last_checkpoint(cfg) == str(remote / "checkpoint-4")

    cfg.resume_from_checkpoint = None
    cfg.checkpoint_mirror["resume"] = False
    assert determine_last_checkpoint(cfg) == str(local / "checkpoint-2")


@pytest.mark.parametrize(
    "delete, limit, kept", [("mirror", 2, [2, 3]), ("keep", 2, [1, 2, 3])]
)
def test_callback_mirrors_saves_and_final_outputs(dirs, delete, limit, kept):
    local, remote = dirs
    callback = CheckpointMirrorCallback(str(remote), delete=delete)
    args = SimpleNamespace(output_dir=str(local), save_total_limit=limit)
    control = SimpleNamespace()
    callback.on_train_begin(args, SimpleNamespace(), control)
    for step in (1, 2, 3):
        save(local, step)
        state = SimpleNamespace(global_step=step, best_model_checkpoint=None)
        callback.on_save(args, state, control)
    (local / "config.json").write_text("{}")

    callback.finish()

    assert sorted(checkpoint_steps(remote)) == kept
    assert (remote / "config.json").read_text() == "{}"
    assert callback.mirror is None
