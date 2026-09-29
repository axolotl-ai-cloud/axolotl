"""CPU-offload Trainer gradient-norm telemetry coverage."""

from types import SimpleNamespace


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
