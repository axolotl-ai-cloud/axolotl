"""Saved-tensor layouts under TRL's activation offloader."""

from unittest import mock

import pytest
import torch

from axolotl.monkeypatch.trainer import trl_offload
from axolotl.monkeypatch.trainer.trl_offload import (
    patch_trl_logprob_entropy_saved_layout,
    patch_trl_offload_dense_layout,
)


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


@pytest.fixture
def logprob_function():
    kernels = pytest.importorskip("trl.kernels.logprob_entropy")
    return kernels._LogProbEntropyFunction


class _Ctx:
    def __init__(self, saved, **attrs):
        self._saved = saved
        self.unpacks = 0
        self.__dict__.update(attrs)

    @property
    def saved_tensors(self):
        self.unpacks += 1
        return self._saved


@pytest.fixture
def patched(monkeypatch, logprob_function):
    seen = {}

    def backward(ctx, grad_logprobs, grad_entropy):
        seen["saved"] = ctx.saved_tensors
        seen["layout"] = ctx.layout
        seen["index_strides"] = getattr(ctx, "index_strides", None)
        seen["mask_strides"] = getattr(ctx, "mask_strides", None)
        seen["temperature"] = ctx.temperature
        return None, None, None, None

    monkeypatch.setattr(logprob_function, "backward", staticmethod(backward))
    monkeypatch.setattr(logprob_function, "_axolotl_saved_layout", False, raising=False)
    patch_trl_logprob_entropy_saved_layout()
    return seen


@pytest.mark.parametrize("with_mask", [False, True])
def test_backward_uses_the_unpacked_layout(patched, logprob_function, with_mask):
    # what the offloader hands back for a shifted logits[:, :-1] / labels[:, 1:] view
    logits = torch.zeros(4, 7, 11)
    index = torch.zeros(4, 7, dtype=torch.long)
    row_mask = torch.ones(4, 7, dtype=torch.bool) if with_mask else None
    saved = (logits, index, torch.zeros(4, 7), torch.zeros(4, 7), row_mask)
    ctx = _Ctx(
        saved,
        layout=(4, 7, 8 * 11, 11),
        index_strides=(8, 1),
        mask_strides=(8, 1) if with_mask else (0, 0),
        temperature=0.7,
    )

    logprob_function.backward(ctx, torch.ones(4, 7), None)

    assert ctx.unpacks == 1
    assert patched["saved"] is saved
    assert patched["layout"] == (4, 7, 7 * 11, 11)
    assert patched["index_strides"] == (7, 1)
    assert patched["mask_strides"] == ((7, 1) if with_mask else (0, 0))
    assert patched["temperature"] == 0.7


def test_older_ctx_without_index_strides(patched, logprob_function):
    logits = torch.zeros(2, 3, 5)
    saved = (logits, torch.zeros(2, 3, dtype=torch.long), None, None, None)
    ctx = _Ctx(saved, layout=(2, 3, 4 * 5, 5), temperature=1.0)

    logprob_function.backward(ctx, torch.ones(2, 3), None)

    assert patched["layout"] == (2, 3, 15, 5)
    assert patched["index_strides"] is None
    assert patched["mask_strides"] is None


def test_skips_a_backward_that_no_longer_reads_the_forward_layout(
    monkeypatch, logprob_function
):
    def backward(ctx, grad_logprobs, grad_entropy):
        return None, None, None, None

    monkeypatch.setattr(logprob_function, "backward", staticmethod(backward))
    monkeypatch.setattr(logprob_function, "_axolotl_saved_layout", False, raising=False)
    patch_trl_logprob_entropy_saved_layout()

    assert logprob_function.backward is backward
    assert logprob_function._axolotl_saved_layout is False
