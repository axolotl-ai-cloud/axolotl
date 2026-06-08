"""Tests for quantization utility functions."""

import bitsandbytes as bnb
import torch

from axolotl.kernels.quantize import dequantize


def _nf4_pair(shape=(64, 64), device="cuda", dtype=torch.float16, double_quant=True):
    """Real bnb NF4 (packed_weight, quant_state) pair for the given shape."""
    W = torch.randn(shape, device=device, dtype=dtype)
    packed, quant_state = bnb.functional.quantize_4bit(
        W, quant_type="nf4", compress_statistics=double_quant
    )
    return packed, quant_state


def test_dequantize_null_state():
    """dequantize returns input unchanged when quant_state is None."""
    W = torch.randn(64, 64)
    assert torch.equal(dequantize(W, None), W)


def test_dequantize_shape_preservation():
    shape = (128, 64)
    packed, quant_state = _nf4_pair(shape)
    result = dequantize(packed, quant_state)
    assert result.shape == shape
    assert result.dtype == torch.float16
    assert result.device == packed.device


def test_dequantize_transposed():
    """Transposed input → transposed output (values, not just shape)."""
    shape = (128, 64)  # non-square: catches dim-swap bugs
    packed, quant_state = _nf4_pair(shape)
    # packed is (4096, 1); packed.t() is (1, 4096), the leading-dim-1 signal.
    # bnb dequants to quant_state.shape (128, 64) then returns out.t() → (64, 128).
    expected = bnb.functional.dequantize_4bit(packed, quant_state, quant_type="nf4").t()
    result = dequantize(packed.t(), quant_state)
    assert tuple(result.shape) == (shape[1], shape[0])
    torch.testing.assert_close(result, expected)


def test_dequantize_non_nested():
    """Single-quant (compress_statistics=False) falls back to bnb wrapper."""
    shape = (128, 64)
    packed, quant_state = _nf4_pair(shape, double_quant=False)
    result = dequantize(packed, quant_state)
    assert result.shape == shape


def test_dequantize_torch_compile_nested():
    """NF4 double-quant under torch.compile (the QLoRA hot path)."""
    shape = (128, 64)
    packed, quant_state = _nf4_pair(shape, double_quant=True)

    eager = dequantize(packed, quant_state)
    compiled = torch.compile(dequantize)(packed, quant_state)

    assert compiled.shape == eager.shape
    assert compiled.dtype == eager.dtype
    torch.testing.assert_close(compiled, eager)


def test_dequantize_torch_compile_non_nested():
    """torch.compile also works for the single-quant fallback path."""
    shape = (128, 64)
    packed, quant_state = _nf4_pair(shape, double_quant=False)

    eager = dequantize(packed, quant_state)
    compiled = torch.compile(dequantize)(packed, quant_state)

    torch.testing.assert_close(compiled, eager)
