"""Tests for quantization utility functions."""

import bitsandbytes as bnb
import pytest
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_dequantize_matches_bnb(dtype):
    """Bit-exact vs bnb's own dequant for every supported output dtype. fp32 previously
    fell through to the bf16 kernel writing into an fp32 buffer and produced garbage."""
    shape = (128, 64)
    packed, quant_state = _nf4_pair(shape, dtype=dtype)
    expected = bnb.functional.dequantize_4bit(packed, quant_state, quant_type="nf4")

    result = dequantize(packed, quant_state)
    assert result.dtype == dtype
    torch.testing.assert_close(result, expected)

    # transposed input → transposed output (values, not just shape)
    result_t = dequantize(packed.t(), quant_state)
    torch.testing.assert_close(result_t, expected.t())


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
