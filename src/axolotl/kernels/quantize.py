"""Module for quantization state, quantization, and dequantization methods."""

from dataclasses import dataclass

import torch


@dataclass
class QuantState:
    """
    Holds quantization parameters for a tensor.

    Attributes:
        scale: Scale factor(s) for dequantization. Can be per-tensor or per-channel.
        zero_point: Zero point for asymmetric quantization.
        bits: Number of bits used for quantization (4, 8).
        scheme: Quantization scheme ('symmetric' or 'asymmetric').
        axis: Axis for per-channel quantization (None for per-tensor).
    """

    scale: torch.Tensor
    zero_point: torch.Tensor | None = None
    bits: int = 8
    scheme: str = "symmetric"
    axis: int | None = None


def quantize(
    weight: torch.Tensor,
    bits: int = 8,
    scheme: str = "symmetric",
    axis: int | None = None,
) -> tuple[torch.Tensor, QuantState]:
    """
    Quantize weights and return quantization state.

    Args:
        weight: Weight tensor to quantize.
        bits: Number of bits (4, 8).
        scheme: "symmetric" or "asymmetric".
        axis: Axis for per-channel quantization (`None` for per-tensor).

    Returns:
        Tuple of `(quantized_weight, quant_state)`.
    """
    if bits not in [4, 8]:
        raise ValueError(f"Unsupported bit width: {bits}")

    # Compute scale and optionally zero point
    if axis is not None:
        # Per-channel quantization
        dim_size = weight.shape[axis]
        weight_flatten = weight.transpose(axis, -1).reshape(-1, dim_size)

        if scheme == "symmetric":
            max_abs = torch.max(torch.abs(weight_flatten), dim=0)[0]
            scale = max_abs / (2 ** (bits - 1) - 1)
            zero_point = None
        else:
            min_val = torch.min(weight_flatten, dim=0)[0]
            max_val = torch.max(weight_flatten, dim=0)[0]
            scale = (max_val - min_val) / (2**bits - 1)
            zero_point = torch.round(-min_val / scale)
    else:
        # Per-tensor quantization
        if scheme == "symmetric":
            max_abs = torch.max(torch.abs(weight))
            scale = max_abs / (2 ** (bits - 1) - 1)
            zero_point = None
        else:
            min_val = torch.min(weight)
            max_val = torch.max(weight)
            scale = (max_val - min_val) / (2**bits - 1)
            zero_point = torch.round(-min_val / scale)

    # Create quantization state
    quant_state = QuantState(
        scale=scale, zero_point=zero_point, bits=bits, scheme=scheme, axis=axis
    )

    # Quantize weights
    if scheme == "symmetric":
        q_weight = torch.round(weight / scale)
    else:
        q_weight = torch.round(weight / scale + zero_point)

    # Pack 4-bit values if needed
    if bits == 4:
        q_weight = q_weight.reshape(-1, 2)
        packed = (q_weight[:, 0] & 0xF) | ((q_weight[:, 1] & 0xF) << 4)
        q_weight = packed.reshape(weight.shape)

    return q_weight.to(torch.int8), quant_state


def dequantize(weight: torch.Tensor, quant_state: QuantState | None) -> torch.Tensor:
    """
    Fast dequantization of weights supporting different quantization schemes.

    Args:
        weight: Quantized weight tensor.
        quant_state: Quantization parameters. If None, returns original weight.

    Returns:
        Dequantized weight tensor.

    Note:
        Supports:
        - 4, 8 bit quantization.
        - Symmetric, asymmetric quantization.
        - Per-tensor, per-channel scaling.
    """
    if quant_state is None:
        return weight

    # Get quantization parameters
    scale = quant_state.scale
    zero_point = quant_state.zero_point
    bits = quant_state.bits

    # Handle different bit widths
    if bits == 8:
        # 8-bit quantization
        if quant_state.scheme == "symmetric":
            return weight * scale  # Symmetric: w = q * s
        return (weight - zero_point) * scale  # Asymmetric: w = (q - z) * s
    if bits == 4:
        # 4-bit quantization requires unpacking
        # Each byte contains 2 4-bit values
        # Convert to 8-bit first
        weight_unpacked = torch.zeros(
            weight.shape + (2,), dtype=torch.int8, device=weight.device
        )
        weight_unpacked[..., 0] = weight & 0xF  # Lower 4 bits
        weight_unpacked[..., 1] = (weight >> 4) & 0xF  # Upper 4 bits

        # Reshape to original dimensions
        weight_8bit = weight_unpacked.reshape(-1, 2).squeeze(-1)

        if quant_state.scheme == "symmetric":
            return weight_8bit * scale
        return (weight_8bit - zero_point) * scale

    raise ValueError(f"Unsupported bit width: {bits}")
