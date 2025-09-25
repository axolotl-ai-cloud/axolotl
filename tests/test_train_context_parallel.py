"""Unit tests for choosing the correct context parallel implementation."""

from types import SimpleNamespace

from axolotl.train import execute_training
from axolotl.utils.dict import DictDefault


class DummyTrainer:
    """Minimal trainer stub to exercise execute_training."""

    def __init__(self):
        self.model = object()
        self.ref_model = None
        self.accelerator = SimpleNamespace(torch_device_mesh=None)
        self.train_called = False

    def train(self, resume_from_checkpoint=None):  # pylint: disable=unused-argument
        self.train_called = True


class DummyPluginManager:
    """Minimal plugin manager stub."""

    @staticmethod
    def post_train(cfg, model):  # pylint: disable=unused-argument
        return None


class DummyContext:
    """Test context manager that records entries/exits."""

    def __init__(self, recorder, **kwargs):
        recorder.append({"kwargs": kwargs})
        self.recorder = recorder

    def __enter__(self):
        self.recorder[-1]["entered"] = True
        return self

    def __exit__(self, exc_type, exc, tb):  # pylint: disable=unused-argument
        self.recorder[-1]["exited"] = True
        return False


def _base_cfg(**overrides):
    base = {
        "context_parallel_size": 2,
        "gradient_accumulation_steps": 1,
        "ring_attn_func": None,
        "heads_k_stride": None,
        "rl": None,
        "flash_optimum": False,
    }
    base.update(overrides)
    return DictDefault(base)


def test_execute_training_uses_ring_when_flash(monkeypatch):
    """FlashAttention CP should engage the custom ring context manager."""
    recorder: list[dict] = []

    monkeypatch.setattr(
        "axolotl.train.SequenceParallelContextManager",
        lambda **kwargs: DummyContext(recorder, **kwargs),
    )
    monkeypatch.setattr(
        "axolotl.train.PluginManager.get_instance",
        lambda: DummyPluginManager(),
    )

    cfg = _base_cfg(flash_attention=True, sdp_attention=False)
    trainer = DummyTrainer()

    execute_training(cfg, trainer, resume_from_checkpoint=None)

    assert trainer.train_called
    assert len(recorder) == 1
    assert recorder[0]["kwargs"]["context_parallel_size"] == 2
    assert recorder[0].get("entered") is True
    assert recorder[0].get("exited") is True


def test_execute_training_uses_transformers_cp_for_sdpa(monkeypatch):
    """SDPA CP should bypass the ring context manager."""
    invoked = {"count": 0}

    class NoOpContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # pylint: disable=unused-argument
            return False

    monkeypatch.setattr(
        "axolotl.train.SequenceParallelContextManager",
        lambda **kwargs: invoked.__setitem__("count", invoked["count"] + 1)
        or NoOpContext(),
    )
    monkeypatch.setattr(
        "axolotl.train.PluginManager.get_instance",
        lambda: DummyPluginManager(),
    )

    cfg = _base_cfg(flash_attention=False, sdp_attention=True)
    trainer = DummyTrainer()

    execute_training(cfg, trainer, resume_from_checkpoint=None)

    assert trainer.train_called
    assert invoked["count"] == 0
