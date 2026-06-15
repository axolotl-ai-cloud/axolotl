"""CPU unit tests for dequantize_weight's tensor-subclass discrimination."""

import torch
from torch import nn

from axolotl.kernels.quantize import dequantize_weight


def test_dequantize_weight_plain_tensor():
    """Test that dequantize_weight passes through unquantized tensors unchanged"""
    W = torch.randn(32, 64)
    result = dequantize_weight(W, quant_state=None, transpose=False)
    assert torch.equal(result, W)


def test_dequantize_weight_plain_tensor_transpose():
    """Test that dequantize_weight transposes unquantized tensors"""
    W = torch.randn(32, 64)
    result = dequantize_weight(W, quant_state=None, transpose=True)
    assert result.shape == (64, 32)
    assert torch.equal(result, W.t())


def test_dequantize_weight_nn_parameter():
    """nn.Parameter is a Tensor subclass but not a quantized one; it must take
    the plain-tensor path."""
    W = nn.Parameter(torch.randn(32, 64), requires_grad=False)
    result = dequantize_weight(W, quant_state=None, transpose=False)
    assert torch.equal(result, W)

    result_t = dequantize_weight(W, quant_state=None, transpose=True)
    assert result_t.shape == (64, 32)
    assert torch.equal(result_t, W.t())


class _FakeAffineQuantizedTensor(torch.Tensor):
    """Stand-in for ``torchao.dtypes.AffineQuantizedTensor`` for unit tests.

    Only models the surface ``dequantize_weight`` touches: a ``dequantize()``
    method returning the unquantized tensor, and not being a plain
    ``torch.Tensor`` (so ``type(W) is not torch.Tensor`` is True).
    """

    @staticmethod
    def __new__(cls, original, *args, **kwargs):
        return torch.Tensor._make_subclass(cls, original)

    def __init__(self, original):
        super().__init__()
        self._original = original

    def dequantize(self):
        return self._original


class _FakeNF4Tensor(torch.Tensor):
    """Stand-in for ``torchao.dtypes.NF4Tensor`` for unit tests."""

    @staticmethod
    def __new__(cls, original, *args, **kwargs):
        return torch.Tensor._make_subclass(cls, original)

    def __init__(self, original):
        super().__init__()
        self._original = original

    def get_original_weight(self):
        return self._original


def test_dequantize_weight_affine_quantized_tensor():
    """AffineQuantizedTensor: dequantize_weight calls .dequantize()."""
    base = torch.randn(8, 16)
    W = _FakeAffineQuantizedTensor(torch.zeros_like(base))
    W._original = base

    result = dequantize_weight(W, quant_state=None)
    assert torch.equal(result, base)


def test_dequantize_weight_affine_quantized_tensor_transpose():
    """AffineQuantizedTensor: dequantize_weight honors transpose=True."""
    base = torch.randn(8, 16)
    W = _FakeAffineQuantizedTensor(torch.zeros_like(base))
    W._original = base

    result = dequantize_weight(W, quant_state=None, transpose=True)
    assert result.shape == (16, 8)
    assert torch.equal(result, base.t())


def test_dequantize_weight_nf4_tensor():
    """NF4Tensor: dequantize_weight calls .get_original_weight()."""
    base = torch.randn(8, 16)
    W = _FakeNF4Tensor(torch.zeros_like(base))
    W._original = base

    result = dequantize_weight(W, quant_state=None)
    assert torch.equal(result, base)


def test_dequantize_weight_nf4_tensor_transpose():
    """NF4Tensor: dequantize_weight honors transpose=True."""
    base = torch.randn(8, 16)
    W = _FakeNF4Tensor(torch.zeros_like(base))
    W._original = base

    result = dequantize_weight(W, quant_state=None, transpose=True)
    assert result.shape == (16, 8)
    assert torch.equal(result, base.t())
