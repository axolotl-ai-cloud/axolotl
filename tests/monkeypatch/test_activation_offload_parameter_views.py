"""Parameter-storage tracking for TRL's activation offloader under FSDP2."""

from types import SimpleNamespace

import pytest
import torch

from axolotl.core.trainers.mixins import activation_checkpointing as offload


class LocalShardView(torch.Tensor):
    pass


def test_inaccessible_storage_keeps_existing_fallback():
    class Opaque:
        def untyped_storage(self):
            raise RuntimeError("storage unavailable")

    assert offload._storage_ptr(Opaque()) is None
    assert offload._storage_ptr(SimpleNamespace()) is None


def test_live_fsdp_parameter_storage_is_scoped_to_owner_forward(monkeypatch):
    class Context:
        def __init__(self):
            self.param_storages = {12345}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    context = Context()
    seen = []

    class Owner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 4))
            self.fail = True

        def forward(self, inputs):
            pointer = self.weight.untyped_storage().data_ptr()
            seen.append(
                (
                    pointer in context.param_storages,
                    inputs.untyped_storage().data_ptr() in context.param_storages,
                )
            )
            if self.fail:
                raise RuntimeError("injected forward failure")
            return inputs @ self.weight

    model = Owner()
    monkeypatch.setattr(offload, "DTensor", LocalShardView)
    offload._install_live_fsdp_parameter_storage_hooks(context, model)
    assert context.param_storages == {12345}

    monkeypatch.setattr(offload, "DTensor", torch.nn.Parameter)

    class Harness(offload.ActivationOffloadingMixin):
        pass

    harness = object.__new__(Harness)
    harness.activation_offload_context = context
    harness.model = model
    monkeypatch.setattr(
        offload.Trainer,
        "training_step",
        lambda self, *_args, **_kwargs: self.model(torch.randn(2, 4)),
    )
    assert context.param_storages == {12345}
    with pytest.raises(RuntimeError, match="injected forward failure"):
        Harness.training_step(harness)
    assert context.param_storages == set()

    model.fail = False
    Harness.training_step(harness)
    assert seen == [(True, False), (True, False)]
    assert context.param_storages == set()
