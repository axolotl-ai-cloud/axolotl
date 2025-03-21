"""Tests for quantization utility functions."""

# pylint: disable=invalid-name

import torch
from bitsandbytes.functional import QuantState

from axolotl.kernels.quantize import dequantize


def test_dequantize_null_state():
    """Test that dequantize returns input unchanged when quant_state is None"""
    W = torch.randn(32, 32)
    assert torch.equal(dequantize(W, None), W)


def test_dequantize_shape_preservation():
    """Test that dequantization preserves expected shapes"""
    shape = (32, 32)
    W = torch.randn(shape, device="cuda")

    quant_state = QuantState(
        absmax=torch.ones(shape[0], device="cuda"),
        shape=shape,
        code=torch.randint(0, 15, shape, device="cuda"),
        dtype=torch.float16,
        blocksize=32,
        quant_type="nf4",
        offset=torch.zeros(shape[0], dtype=torch.int32, device="cuda"),
        state2=QuantState(
            absmax=torch.ones(shape[0], device="cuda"),
            shape=shape,
            code=torch.randint(0, 15, shape, device="cuda"),
            dtype=torch.float16,
            blocksize=32,
            quant_type="nf4",
            offset=None,
            state2=None,
        ),
    )

    result = dequantize(W, quant_state)
    assert result.shape == shape
    assert result.dtype == torch.float16
    assert result.device == W.device


def test_dequantize_transposed():
    """Test that transposed input produces transposed output"""
    shape = (32, 32)
    W = torch.randn(1, shape[1], device="cuda")  # Transposed input

    quant_state = QuantState(
        absmax=torch.ones(1),
        shape=shape,
        code=torch.randint(0, 15, shape),
        dtype=torch.float16,
        blocksize=32,
        quant_type="nf4",
        offset=torch.zeros(1, dtype=torch.int32),
        state2=QuantState(
            absmax=torch.ones(1),
            shape=shape,
            code=torch.randint(0, 15, shape),
            dtype=torch.float16,
            blocksize=32,
            quant_type="nf4",
            offset=None,
            state2=None,
        ),
    )

    result = dequantize(W, quant_state)
    assert result.shape[0] == shape[0]


def test_dequantize_output_tensor():
    """Test dequantization with provided output tensor"""
    shape = (32, 32)
    W = torch.randn(shape, device="cuda")
    out = torch.empty(shape, dtype=torch.float16, device="cuda")

    quant_state = QuantState(
        absmax=torch.ones(shape[0]),
        shape=shape,
        code=torch.randint(0, 15, shape),
        dtype=torch.float16,
        blocksize=32,
        quant_type="nf4",
        offset=torch.zeros(shape[0], dtype=torch.int32),
        state2=QuantState(
            absmax=torch.ones(shape[0]),
            shape=shape,
            code=torch.randint(0, 15, shape),
            dtype=torch.float16,
            blocksize=32,
            quant_type="nf4",
            offset=None,
            state2=None,
        ),
    )

    result = dequantize(W, quant_state, out=out)
    assert result is out
