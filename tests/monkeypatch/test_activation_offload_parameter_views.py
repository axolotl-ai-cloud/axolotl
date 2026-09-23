"""Compute-stream cloning must preserve TRL's parameter-storage exclusions."""

from types import SimpleNamespace

import pytest
import torch

from axolotl.core.trainers.mixins import activation_checkpointing as offload


class AcceleratorView(torch.Tensor):
    @property
    def device(self):
        # Exercise the accelerator branch using real CPU storage and clone operations.
        return torch.device("cuda")


class LocalShardView(AcceleratorView):
    pass


@pytest.mark.parametrize("parameter_view", [False, True])
@pytest.mark.parametrize("offset", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
def test_clone_preserves_parameter_storage_filter(
    parameter_view, offset, sharded, monkeypatch
):
    from trl.models import activation_offloading

    class Offloader:
        def __init__(self):
            self.param_storages = set()
            self.pack_hook = lambda tensor: tensor

    monkeypatch.setattr(activation_offloading, "OffloadActivations", Offloader)
    monkeypatch.setattr(offload, "DTensor", LocalShardView)
    offload._patch_trl_offload_compute_stream_clone()
    first_init = Offloader.__init__
    offload._patch_trl_offload_compute_stream_clone()
    assert Offloader.__init__ is first_init
    manager = Offloader()
    weight = torch.randn(8, 8)
    local = weight[1:] if offset else weight.t()
    source = weight.clone() if sharded else weight
    source = source[1:] if offset else source.t()
    view = source.as_subclass(LocalShardView if sharded else AcceleratorView)
    if sharded:
        view._local_tensor = local
    if parameter_view:
        manager.param_storages.add(weight.untyped_storage().data_ptr())
    actual = manager.pack_hook(view)
    if parameter_view:
        assert actual is view
        assert (
            actual.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        )
    else:
        assert (
            actual.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        )
        assert actual.is_contiguous() and actual.storage_offset() == 0
        torch.testing.assert_close(actual.as_subclass(torch.Tensor), local)


def test_inaccessible_storage_keeps_existing_fallback():
    class Opaque:
        def untyped_storage(self):
            raise RuntimeError("storage unavailable")

    assert not offload._is_offload_parameter_view(Opaque(), set())
    assert not offload._is_offload_parameter_view(SimpleNamespace(), set())
