"""Retrying checkpoint writes after a full disk, exceeded quota or I/O error."""

import contextlib
import errno
import io
import os
from types import SimpleNamespace

import pytest
import torch
import transformers.modeling_utils as modeling_utils
from transformers import TrainerState

from axolotl.core import checkpoint_retry
from axolotl.core.checkpoint_retry import (
    is_disk_error,
    retry_call,
    retry_disk_writes,
)
from axolotl.core.trainers.mixins.checkpoints import (
    CheckpointWriteRetryMixin,
    with_checkpoint_write_retry,
)


class CheckpointException(BaseException):
    """Stands in for torch.distributed.checkpoint's, matched by class name."""

    def __init__(self, failures):
        super().__init__("checkpoint failed")
        self.failures = failures


def _chained(outer, inner):
    try:
        try:
            raise inner
        except BaseException as exc:
            raise outer from exc
    except BaseException as exc:
        return exc


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("I/O error: Disk quota exceeded (os error 122)"),
        OSError(errno.EDQUOT, "Disk quota exceeded"),
        OSError(errno.ENOSPC, "No space left on device"),
        OSError(errno.EIO, "Input/output error"),
        RuntimeError(
            'Error while serializing: IoError(Os { code: 28, kind: StorageFull, message: "No space left on device" })'
        ),
        RuntimeError(
            "[enforce fail at inline_container.cc:664] . PytorchStreamWriter failed "
            "writing file data/0: file write failed"
        ),
        RuntimeError("[Errno 116] Stale file handle"),
        RuntimeError("basic_ios::clear: iostream error"),
        RuntimeError(
            "[enforce fail at inline_container.cc:672] . unexpected pos 704 vs 598"
        ),
        _chained(RuntimeError("save failed"), OSError(errno.ENOSPC, "full")),
        ExceptionGroup("writers", [ValueError("x"), OSError(errno.EDQUOT, "q")]),
        CheckpointException({1: (OSError(errno.ENOSPC, "full"), None)}),
    ],
)
def test_recognizes_disk_errors(exc):
    assert is_disk_error(exc)


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("bad shape"),
        OSError(errno.EACCES, "Permission denied"),
        KeyboardInterrupt(),
        CheckpointException({0: (ValueError("planner"), None)}),
    ],
)
def test_ignores_other_errors(exc):
    assert not is_disk_error(exc)


@pytest.fixture
def sleeps(monkeypatch):
    waited = []
    monkeypatch.setattr(checkpoint_retry.time, "sleep", waited.append)
    return waited


def _failing(times, exc_factory, then=lambda *a, **k: "written"):
    calls = []

    def writer(*args, **kwargs):
        calls.append(args)
        if len(calls) <= times:
            raise exc_factory()
        return then(*args, **kwargs)

    writer.calls = calls
    return writer


def test_torch_save_is_rewritten_after_a_disk_error(tmp_path, monkeypatch, sleeps):
    real_save = torch.save
    flaky = _failing(1, lambda: RuntimeError("file write failed"), then=real_save)
    monkeypatch.setattr(torch, "save", flaky)
    path = tmp_path / "optimizer.pt"

    with retry_disk_writes(wait_seconds=7):
        torch.save({"step": 3}, path)

    assert torch.load(path) == {"step": 3}
    assert len(flaky.calls) == 2 and sleeps == [7]
    assert torch.save is flaky


def test_raises_after_the_last_retry(tmp_path, monkeypatch, sleeps):
    always = _failing(10, lambda: OSError(errno.EDQUOT, "Disk quota exceeded"))
    monkeypatch.setattr(torch, "save", always)

    with retry_disk_writes(wait_seconds=1, retries=2):
        with pytest.raises(OSError, match="quota"):
            torch.save({}, tmp_path / "x.pt")

    assert len(always.calls) == 3 and sleeps == [1, 1]


def test_other_errors_and_streams_are_not_retried(tmp_path, monkeypatch, sleeps):
    monkeypatch.setattr(torch, "save", _failing(1, lambda: ValueError("bad")))
    with retry_disk_writes(wait_seconds=1):
        with pytest.raises(ValueError):
            torch.save({}, tmp_path / "x.pt")

    monkeypatch.setattr(
        torch, "save", _failing(1, lambda: RuntimeError("file write failed"))
    )
    with retry_disk_writes(wait_seconds=1):
        with pytest.raises(RuntimeError):
            torch.save({}, io.BytesIO())
    assert sleeps == []


def test_no_retry_outside_the_scope(tmp_path, monkeypatch, sleeps):
    monkeypatch.setattr(
        torch, "save", _failing(1, lambda: RuntimeError("file write failed"))
    )
    with pytest.raises(RuntimeError):
        torch.save({}, tmp_path / "x.pt")
    assert sleeps == []


def test_safetensors_and_trainer_state_writers_are_covered(
    tmp_path, monkeypatch, sleeps
):
    quota = lambda: RuntimeError("I/O error: Disk quota exceeded (os error 122)")  # noqa: E731
    safetensors_save = _failing(1, quota)
    monkeypatch.setattr(modeling_utils, "safe_save_file", safetensors_save)
    real_state_save = TrainerState.save_to_json
    state_save = _failing(1, quota, then=real_state_save)
    monkeypatch.setattr(TrainerState, "save_to_json", state_save)

    with retry_disk_writes(wait_seconds=2):
        assert (
            modeling_utils.safe_save_file({}, str(tmp_path / "model.safetensors"))
            == "written"
        )
        TrainerState(global_step=4).save_to_json(str(tmp_path / "trainer_state.json"))

    assert len(safetensors_save.calls) == 2 and len(state_save.calls) == 2
    assert (
        TrainerState.load_from_json(str(tmp_path / "trainer_state.json")).global_step
        == 4
    )
    assert sleeps == [2, 2]


def test_distributed_checkpoint_retries_only_collective_failures(monkeypatch, sleeps):
    import torch.distributed.checkpoint as dcp

    collective = _failing(
        1, lambda: CheckpointException({0: (OSError(errno.ENOSPC, "full"), None)})
    )
    monkeypatch.setattr(dcp, "save", collective)
    with retry_disk_writes(wait_seconds=3):
        assert dcp.save({}, checkpoint_id="/ckpt/pytorch_model_fsdp_0") == "written"
    assert len(collective.calls) == 2 and sleeps == [3]

    # a rank-local error inside a DCP save could leave other ranks in a collective
    local = _failing(1, lambda: OSError(errno.ENOSPC, "full"))
    monkeypatch.setattr(dcp, "save", local)
    with retry_disk_writes(wait_seconds=3):
        with pytest.raises(OSError):
            dcp.save({}, checkpoint_id="/ckpt")
    assert len(local.calls) == 1


def test_nested_scopes_keep_writers_wrapped_until_the_outermost_exits(
    monkeypatch, sleeps
):
    original = torch.save
    with retry_disk_writes(wait_seconds=1):
        wrapped = torch.save
        with retry_disk_writes(wait_seconds=5):
            assert torch.save is wrapped
        assert torch.save is wrapped
    assert torch.save is original


def test_retry_call_uses_the_active_policy(tmp_path, sleeps):
    write = _failing(1, lambda: OSError(errno.ENOSPC, "full"))
    with retry_disk_writes(wait_seconds=4):
        assert retry_call(tmp_path / "tokens.json", write, "data") == "written"
    assert sleeps == [4]

    with pytest.raises(OSError):
        retry_call(
            tmp_path / "tokens.json", _failing(1, lambda: OSError(errno.ENOSPC, "x"))
        )


@pytest.mark.skipif(not os.path.exists("/dev/full"), reason="needs /dev/full (Linux)")
@pytest.mark.parametrize("payload", [{"step": 1}, {"w": torch.ones(1 << 20)}])
def test_real_full_device_is_detected_and_retried(sleeps, payload):
    # safetensors writes a temporary file beside the target, so /dev/full cannot
    # exercise it; its errors are covered by message in test_recognizes_disk_errors
    with retry_disk_writes(wait_seconds=0):
        with pytest.raises(BaseException) as error:
            torch.save(payload, "/dev/full")

    assert is_disk_error(error.value)
    assert sleeps == [0]


def test_mixin_scope_follows_training_arguments():
    class Trainer(CheckpointWriteRetryMixin):
        def __init__(self, args):
            self.args = args

    assert isinstance(
        Trainer(SimpleNamespace()).checkpoint_write_retry(), contextlib.nullcontext
    )
    trainer = Trainer(
        SimpleNamespace(checkpoint_save_retry_wait=9, checkpoint_save_retries=2)
    )
    with trainer.checkpoint_write_retry():
        assert checkpoint_retry._active_policy() == checkpoint_retry.RetryPolicy(9, 2)
    assert checkpoint_retry._active_policy() is None


def test_decorated_override_runs_inside_the_scope():
    seen = []

    class Stub:
        @contextlib.contextmanager
        def checkpoint_write_retry(self):
            seen.append("enter")
            yield
            seen.append("exit")

        @with_checkpoint_write_retry
        def _save_checkpoint(self, model, trial):
            seen.append("save")
            return "saved"

    assert Stub()._save_checkpoint(None, None) == "saved"
    assert seen == ["enter", "save", "exit"]

    class Bare:
        @with_checkpoint_write_retry
        def _save_checkpoint(self):
            return "saved"

    assert Bare()._save_checkpoint() == "saved"


def test_every_axolotl_trainer_retries_checkpoint_writes():
    from axolotl.core.trainers.base import AxolotlTrainer
    from axolotl.core.trainers.dpo.trainer import AxolotlDPOTrainer
    from axolotl.core.trainers.ebft.trl_async_trainer import AxolotlTRLAsyncEBFTTrainer
    from axolotl.core.trainers.grpo.trainer import (
        AxolotlAsyncGRPOTrainer,
        AxolotlGRPOTrainer,
    )
    from axolotl.core.trainers.trl import (
        AxolotlCPOTrainer,
        AxolotlKTOTrainer,
        AxolotlORPOTrainer,
        AxolotlPRMTrainer,
        AxolotlRewardTrainer,
    )

    for trainer in (
        AxolotlTrainer,
        AxolotlDPOTrainer,
        AxolotlGRPOTrainer,
        AxolotlAsyncGRPOTrainer,
        AxolotlTRLAsyncEBFTTrainer,
        AxolotlORPOTrainer,
        AxolotlKTOTrainer,
        AxolotlCPOTrainer,
        AxolotlRewardTrainer,
        AxolotlPRMTrainer,
    ):
        assert issubclass(trainer, CheckpointWriteRetryMixin), trainer


def test_config_and_tokenizer_writers_are_covered(tmp_path, monkeypatch, sleeps):
    from transformers import PretrainedConfig
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    quota = lambda: OSError(errno.EDQUOT, "Disk quota exceeded")  # noqa: E731
    config_save = _failing(1, quota)
    tokenizer_save = _failing(1, quota)
    monkeypatch.setattr(PretrainedConfig, "save_pretrained", config_save)
    monkeypatch.setattr(PreTrainedTokenizerBase, "save_pretrained", tokenizer_save)

    with retry_disk_writes(wait_seconds=1):
        PretrainedConfig().save_pretrained(str(tmp_path))
        PreTrainedTokenizerBase.save_pretrained(object(), str(tmp_path))

    assert len(config_save.calls) == len(tokenizer_save.calls) == 2
    assert sleeps == [1, 1]
    assert PretrainedConfig.save_pretrained is config_save


def test_only_the_outermost_writer_retries(tmp_path, monkeypatch, sleeps):
    from transformers import PretrainedConfig

    inner = _failing(1, lambda: OSError(errno.ENOSPC, "full"))
    monkeypatch.setattr(torch, "save", inner)

    def save_config(self, save_directory):
        torch.save({}, f"{save_directory}/extra.pt")
        return "saved"

    monkeypatch.setattr(PretrainedConfig, "save_pretrained", save_config)
    with retry_disk_writes(wait_seconds=1, retries=3):
        assert PretrainedConfig().save_pretrained(str(tmp_path)) == "saved"

    assert len(inner.calls) == 2 and sleeps == [1]
