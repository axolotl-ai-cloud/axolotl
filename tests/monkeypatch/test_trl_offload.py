"""Saved-tensor layouts under TRL's activation offloader."""

from unittest import mock

import pytest
import torch

from axolotl.monkeypatch.trainer import trl_offload
from axolotl.monkeypatch.trainer.trl_offload import patch_trl_offload_dense_layout


class _CompactingOffloader:
    """Stands in for OffloadActivations: saves contiguous copies, ids restart per step."""

    def __init__(self):
        self.saved = {}
        self.next_id = 0
        self.pack_hook = self._pack
        self.unpack_hook = self._unpack

    def _pack(self, tensor):
        tensor_id = self.next_id
        self.next_id += 1
        self.saved[tensor_id] = tensor.contiguous()
        return tensor_id

    def _unpack(self, tensor_id):
        return self.saved.pop(tensor_id)


@pytest.fixture
def dense_layout(monkeypatch):
    from trl.models import activation_offloading

    monkeypatch.setattr(
        activation_offloading, "OffloadActivations", _CompactingOffloader
    )
    monkeypatch.setattr(_CompactingOffloader, "__init__", _CompactingOffloader.__init__)
    patch_trl_offload_dense_layout()
    first_init = _CompactingOffloader.__init__
    patch_trl_offload_dense_layout()
    assert _CompactingOffloader.__init__ is first_init
    yield
    del _CompactingOffloader._axolotl_dense_layout


@pytest.mark.parametrize(
    "make",
    [
        lambda: torch.randn(2, 3, 4).transpose(1, 2),
        lambda: torch.randn(2, 3, 4, 5).permute(0, 2, 1, 3),
        lambda: torch.randn(8, 8).t(),
    ],
    ids=["transpose", "permute", "weight_t"],
)
def test_transposed_tensor_comes_back_with_its_strides(dense_layout, make):
    offloader = _CompactingOffloader()
    tensor = make()

    restored = offloader.unpack_hook(offloader.pack_hook(tensor))

    assert restored.stride() == tensor.stride()
    assert torch.equal(restored, tensor)


def test_reused_id_does_not_keep_a_stale_order(dense_layout):
    offloader = _CompactingOffloader()
    offloader.pack_hook(torch.randn(2, 3, 4).transpose(1, 2))
    offloader.next_id = 0
    tensor = torch.randn(4, 3, 2)

    restored = offloader.unpack_hook(offloader.pack_hook(tensor))

    assert restored.shape == tensor.shape and torch.equal(restored, tensor)


def test_contiguous_order():
    base = torch.randn(2, 3, 4)
    assert trl_offload._contiguous_order(base) is None
    assert trl_offload._contiguous_order(base.transpose(1, 2)) == [0, 2, 1]
    # sliced and broadcast views have no free contiguous permutation
    assert trl_offload._contiguous_order(base[:, :-1]) is None
    assert trl_offload._contiguous_order(base[:, :1].expand(2, 5, 4)) is None
    assert (
        trl_offload._contiguous_order(torch.nn.Parameter(base.transpose(1, 2))) is None
    )


class _SavesTransposed(torch.autograd.Function):
    seen: list = []

    @staticmethod
    def forward(ctx, inputs):
        transposed = inputs.transpose(1, 2)
        ctx.save_for_backward(transposed)
        return inputs.sum() * 1.0

    @staticmethod
    def backward(ctx, grad):
        (transposed,) = ctx.saved_tensors
        _SavesTransposed.seen.append(transposed.stride())
        return grad * transposed.transpose(1, 2)


def test_real_offloader_returns_transposed_strides(monkeypatch):
    from trl.models.activation_offloading import OffloadActivations

    monkeypatch.setattr(OffloadActivations, "__init__", OffloadActivations.__init__)
    monkeypatch.setattr(
        OffloadActivations, "_axolotl_dense_layout", False, raising=False
    )
    patch_trl_offload_dense_layout()
    _SavesTransposed.seen.clear()
    inputs = torch.randn(2, 64, 32, requires_grad=True)
    with (
        mock.patch.object(
            torch.accelerator, "current_accelerator", return_value=torch.device("cpu")
        ),
        mock.patch.object(torch.cpu, "empty_cache", lambda: None, create=True),
    ):
        offload = OffloadActivations(
            use_pin_memory=False, use_streams=False, min_offload_size=0
        )
        with offload:
            _SavesTransposed.apply(inputs).backward()

    assert _SavesTransposed.seen == [inputs.transpose(1, 2).stride()]
    assert torch.equal(inputs.grad, inputs.detach())
