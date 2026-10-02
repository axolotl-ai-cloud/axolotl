"""CPU-offload Trainer gradient-norm telemetry coverage."""

from types import SimpleNamespace

import pytest


def test_cpu_offload_telemetry_unscales_before_nonmutating_norm(monkeypatch):
    from torch.distributed.fsdp import CPUOffloadPolicy

    from axolotl.core.trainers.mixins.distributed_parallel import (
        DistributedParallelMixin,
    )
    from axolotl.utils import gradient_clipping

    events = []
    accelerator = SimpleNamespace(
        state=SimpleNamespace(
            fsdp_plugin=SimpleNamespace(cpu_offload=CPUOffloadPolicy())
        ),
        parallelism_config=SimpleNamespace(ep_enabled=False),
        unscale_gradients=lambda: events.append("unscale"),
    )
    trainer = object.__new__(DistributedParallelMixin)
    trainer.accelerator = accelerator
    model = SimpleNamespace(parameters=lambda: [object()])
    monkeypatch.setattr(
        gradient_clipping, "has_cpu_offloaded_dtensor_parameters", lambda _: True
    )

    def norm(parameters):
        assert list(parameters)
        assert events == ["unscale"]
        return 7.0

    monkeypatch.setattr(gradient_clipping, "get_grad_norm_local_shards_", norm)

    assert trainer._get_grad_norm(model) == 7.0
    assert events == ["unscale"]


def test_existing_grad_norm_bypasses_cpu_offload_telemetry():
    from accelerate.utils import DistributedType

    from axolotl.core.trainers.mixins.distributed_parallel import (
        DistributedParallelMixin,
    )

    events = []
    trainer = object.__new__(DistributedParallelMixin)
    trainer.accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        unscale_gradients=lambda: events.append("unscale"),
    )
    assert (
        trainer._get_grad_norm(SimpleNamespace(parameters=lambda: []), grad_norm=3.0)
        == 3.0
    )
    assert events == []


def _ep_trainer(monkeypatch, parallelism, max_grad_norm=1.0):
    from axolotl.core.trainers.mixins.distributed_parallel import (
        DistributedParallelMixin,
    )
    from axolotl.utils import gradient_clipping

    events = []
    trainer = object.__new__(DistributedParallelMixin)
    trainer.accelerator = SimpleNamespace(
        state=SimpleNamespace(fsdp_plugin=SimpleNamespace(cpu_offload=None)),
        parallelism_config=parallelism,
        torch_device_mesh="mesh",
        unscale_gradients=lambda: events.append("unscale"),
    )
    trainer.args = SimpleNamespace(max_grad_norm=max_grad_norm)
    monkeypatch.setattr(gradient_clipping, "ep_local_parameter_ids", lambda _: {1})

    def clip(parameters, max_norm, *, ep_local_parameters, global_mesh):
        events.append(("clip", max_norm, ep_local_parameters, global_mesh))
        return 5.0

    def norm(parameters, *, ep_local_parameters, global_mesh):
        events.append(("norm", ep_local_parameters, global_mesh))
        return 6.0

    monkeypatch.setattr(gradient_clipping, "clip_grad_norm_ep_local_shards_", clip)
    monkeypatch.setattr(gradient_clipping, "get_grad_norm_ep_local_shards_", norm)
    return trainer, events


@pytest.mark.parametrize("via", ["parallelism_config", "env"])
def test_expert_parallel_owns_clipping_without_cpu_offload(monkeypatch, via):
    if via == "env":
        monkeypatch.setenv("PARALLELISM_CONFIG_EP_SIZE", "2")
        parallelism = None
    else:
        monkeypatch.delenv("PARALLELISM_CONFIG_EP_SIZE", raising=False)
        parallelism = SimpleNamespace(ep_enabled=True)
    trainer, events = _ep_trainer(monkeypatch, parallelism)
    model = SimpleNamespace(parameters=lambda: [object()])

    assert trainer._clip_grad_norm(model) == 5.0
    assert events == ["unscale", ("clip", 1.0, {1}, "mesh")]
    events.clear()
    assert trainer._get_grad_norm(model) == 6.0
    assert events == ["unscale", ("norm", {1}, "mesh")]


def test_dense_runs_keep_the_trainer_clipping(monkeypatch):
    from transformers import Trainer

    monkeypatch.delenv("PARALLELISM_CONFIG_EP_SIZE", raising=False)
    trainer, events = _ep_trainer(monkeypatch, SimpleNamespace(ep_enabled=False))
    monkeypatch.setattr(Trainer, "_clip_grad_norm", lambda self, model: "trainer")
    assert trainer._clip_grad_norm(SimpleNamespace(parameters=lambda: [])) == "trainer"
    assert events == []
