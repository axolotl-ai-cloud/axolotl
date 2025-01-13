"""
Module for definition of Low-Rank Adaptation (LoRA) Triton kernels.

See "LoRA: Low-Rank Adaptation of Large Language Models"
    (https://arxiv.org/abs/2106.09685).
"""
# pylint: disable=invalid-name

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

from .geglu import GEGLU
from .swiglu import SwiGLU
from .utils import torch_amp_custom_bwd, torch_amp_custom_fwd


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
    zero_point: Optional[torch.Tensor] = None
    bits: int = 8
    scheme: str = "symmetric"
    axis: Optional[int] = None


def fast_dequantize(
    weight: torch.Tensor, quant_state: Optional[QuantState]
) -> torch.Tensor:
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


def quantize_weight(
    weight: torch.Tensor,
    bits: int = 8,
    scheme: str = "symmetric",
    axis: Optional[int] = None,
) -> Tuple[torch.Tensor, QuantState]:
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


def get_lora_parameters(layer):
    """Extract LoRA parameters from layer."""
    weight = getattr(layer, "weight", None)
    quant_state = getattr(layer, "quant_state", None)
    lora_A = getattr(layer, "lora_A", None)
    lora_B = getattr(layer, "lora_B", None)
    scaling = getattr(layer, "scaling", 1.0)

    return weight, quant_state, lora_A, lora_B, scaling


@triton.jit
def swiglu_forward_kernel(e: tl.tensor, g: tl.tensor):
    """SwiGLU forward activation function."""
    f = e * tl.sigmoid(e)

    return f * g


@triton.jit
def swiglu_backward_kernel(dW: tl.tensor, e: tl.tensor, g: tl.tensor):
    """SwiGLU backward pass computation."""
    sigmoid_e = tl.sigmoid(e)
    f = e * sigmoid_e
    df = sigmoid_e * (1 - f) + f

    return dW * df * g, dW * f, dW * df * g


@triton.jit
def geglu_exact_forward_kernel(e: tl.tensor, g: tl.tensor):
    """GEGLU forward activation with exact GELU."""
    x = e * 0.7978845608028654  # sqrt(2/π)
    cdf = 0.5 * (1.0 + tl.erf(x))

    return e * cdf * g


@triton.jit
def geglu_exact_backward_kernel(dW: tl.tensor, e: tl.tensor, g: tl.tensor):
    """GEGLU backward pass with exact GELU derivative."""
    x = e * 0.7978845608028654
    cdf = 0.5 * (1.0 + tl.erf(x))
    pdf = 0.7978845608028654 * tl.exp(-0.5 * x * x)

    return dW * (cdf + e * pdf) * g, dW * e * cdf, dW * (cdf + e * pdf) * g


class LoRA_MLP(torch.autograd.Function):
    """Optimized LoRA MLP implementation with support for SwiGLU and GEGLU."""

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx,
        X,
        gate_weight,
        gate_quant,
        gate_A,
        gate_B,
        gate_scale,
        up_weight,
        up_quant,
        up_A,
        up_B,
        up_scale,
        down_weight,
        down_quant,
        down_A,
        down_B,
        down_scale,
        activation_fn,  # This will be swiglu_fg_kernel
        inplace=True,
    ):
        batch, seq_len, hidden_dim = X.shape
        X_reshaped = X.view(-1, hidden_dim)
        X_orig = X_reshaped.clone()

        # Gate projection
        gate = torch.matmul(
            X_reshaped,
            fast_dequantize(gate_weight, gate_quant),
        )
        if gate_A is not None:
            gate += X_orig @ gate_A @ gate_B * gate_scale

        # Up projection
        up = torch.matmul(X_orig, fast_dequantize(up_weight, up_quant))
        if up_A is not None:
            up += X_orig @ up_A @ up_B * up_scale

        # Activation - using the kernel wrapper function directly
        hidden = activation_fn(
            gate, up
        )  # This calls swiglu_fg_kernel which handles the grid

        # Down projection
        output = torch.matmul(hidden, fast_dequantize(down_weight, down_quant))
        if down_A is not None:
            output += hidden @ down_A @ down_B * down_scale

        ctx.save_for_backward(
            X_orig, gate, up, hidden, gate_A, gate_B, up_A, up_B, down_A, down_B
        )
        ctx.scales = (gate_scale, up_scale, down_scale)
        ctx.quants = (gate_quant, up_quant, down_quant)
        ctx.weights = (gate_weight, up_weight, down_weight)
        ctx.activation_fn = activation_fn
        ctx.inplace = inplace

        return output.view(batch, seq_len, -1)

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, grad_output):
        (
            X,
            gate,
            up,
            hidden,
            gate_A,
            gate_B,
            up_A,
            up_B,
            down_A,
            down_B,
        ) = ctx.saved_tensors
        gate_scale, up_scale, down_scale = ctx.scales
        gate_quant, up_quant, down_quant = ctx.quants
        gate_weight, up_weight, down_weight = ctx.weights

        batch_size = grad_output.shape[0]
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])

        # Down projection gradients
        d_hidden = torch.matmul(
            grad_output,
            fast_dequantize(down_weight, down_quant).t(),
            out=grad_output if ctx.inplace else None,
        )
        if down_A is not None:
            d_down_A = hidden.t() @ (grad_output @ down_B.t()) * down_scale
            d_down_B = (down_A.t() @ hidden.t()) @ grad_output * down_scale
            d_hidden += grad_output @ down_B.t() @ (down_scale * down_A.t())

        # Backprop through activation
        d_gate, d_up = ctx.activation_fn.backward(d_hidden, gate, up)

        # Up projection gradients
        grad_X = torch.matmul(d_up, fast_dequantize(up_weight, up_quant).t())
        if up_A is not None:
            d_up_A = X.t() @ (d_up @ up_B.t()) * up_scale
            d_up_B = (up_A.t() @ X.t()) @ d_up * up_scale
            grad_X += d_up @ up_B.t() @ (up_scale * up_A.t())

        # Gate projection gradients
        grad_X += torch.matmul(d_gate, fast_dequantize(gate_weight, gate_quant).t())
        if gate_A is not None:
            d_gate_A = X.t() @ (d_gate @ gate_B.t()) * gate_scale
            d_gate_B = (gate_A.t() @ X.t()) @ d_gate * gate_scale
            grad_X += d_gate @ gate_B.t() @ (gate_scale * gate_A.t())

        return (
            grad_X.view(batch_size, -1, X.shape[-1]),
            None,
            None,
            d_gate_A,
            d_gate_B,
            None,
            None,
            None,
            d_up_A,
            d_up_B,
            None,
            None,
            None,
            d_down_A,
            d_down_B,
            None,
            None,
            None,
        )


class LoRA_QKV(torch.autograd.Function):
    """Optimized LoRA QKV implementation with quantization support."""

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx,
        X,
        q_weight,
        q_quant,
        q_A,
        q_B,
        q_scale,
        k_weight,
        k_quant,
        k_A,
        k_B,
        k_scale,
        v_weight,
        v_quant,
        v_A,
        v_B,
        v_scale,
        num_heads,
        head_dim,
        inplace=True,
    ):
        batch_size, seq_len, hidden_dim = X.shape
        X_reshaped = X.view(-1, hidden_dim)

        # Q projection
        Q = torch.matmul(
            X_reshaped,
            fast_dequantize(q_weight, q_quant),
            out=X_reshaped if inplace else None,
        )
        if q_A is not None:
            Q += X_reshaped @ q_A @ q_B * q_scale

        # K projection
        K = torch.matmul(X_reshaped, fast_dequantize(k_weight, k_quant))
        if k_A is not None:
            K += X_reshaped @ k_A @ k_B * k_scale

        # V projection
        V = torch.matmul(X_reshaped, fast_dequantize(v_weight, v_quant))
        if v_A is not None:
            V += X_reshaped @ v_A @ v_B * v_scale

        # Reshape output
        Q = Q.view(batch_size, seq_len, num_heads, head_dim)
        K = K.view(batch_size, seq_len, num_heads, head_dim)
        V = V.view(batch_size, seq_len, num_heads, head_dim)

        ctx.save_for_backward(X_reshaped, q_A, q_B, k_A, k_B, v_A, v_B)
        ctx.scales = (q_scale, k_scale, v_scale)
        ctx.quants = (q_quant, k_quant, v_quant)
        ctx.weights = (q_weight, k_weight, v_weight)
        ctx.dims = (num_heads, head_dim)
        ctx.inplace = inplace

        return Q, K, V

    @staticmethod
    @torch_amp_custom_fwd
    def backward(ctx, grad_Q, grad_K, grad_V):
        X, q_A, q_B, k_A, k_B, v_A, v_B = ctx.saved_tensors
        q_scale, k_scale, v_scale = ctx.scales
        q_quant, k_quant, v_quant = ctx.quants
        q_weight, k_weight, v_weight = ctx.weights

        batch_size = grad_Q.shape[0]
        grad_Q = grad_Q.reshape(-1, grad_Q.shape[-1])
        grad_K = grad_K.reshape(-1, grad_K.shape[-1])
        grad_V = grad_V.reshape(-1, grad_V.shape[-1])

        # Q gradients
        grad_X = torch.matmul(
            grad_Q,
            fast_dequantize(q_weight, q_quant).t(),
            out=grad_Q if ctx.inplace else None,
        )
        if q_A is not None:
            d_q_A = X.t() @ (grad_Q @ q_B.t()) * q_scale
            d_q_B = (q_A.t() @ X.t()) @ grad_Q * q_scale
            grad_X += grad_Q @ q_B.t() @ (q_scale * q_A.t())

        # K gradients
        grad_X += torch.matmul(grad_K, fast_dequantize(k_weight, k_quant).t())
        if k_A is not None:
            d_k_A = X.t() @ (grad_K @ k_B.t()) * k_scale
            d_k_B = (k_A.t() @ X.t()) @ grad_K * k_scale
            grad_X += grad_K @ k_B.t() @ (k_scale * k_A.t())

        # V gradients
        grad_X += torch.matmul(grad_V, fast_dequantize(v_weight, v_quant).t())
        if v_A is not None:
            d_v_A = X.t() @ (grad_V @ v_B.t()) * v_scale
            d_v_B = (v_A.t() @ X.t()) @ grad_V * v_scale
            grad_X += grad_V @ v_B.t() @ (v_scale * v_A.t())

        return (
            grad_X.view(batch_size, -1, X.shape[-1]),
            None,
            None,
            d_q_A,
            d_q_B,
            None,
            None,
            None,
            d_k_A,
            d_k_B,
            None,
            None,
            None,
            d_v_A,
            d_v_B,
            None,
            None,
            None,
            None,
        )


def create_lora_mlp(
    config,
    gate_weight=None,
    up_weight=None,
    down_weight=None,
    gate_A=None,
    gate_B=None,
    up_A=None,
    up_B=None,
    down_A=None,
    down_B=None,
    activation="swiglu",
):
    device = config.get("device", "cuda")
    dtype = config.get("dtype", torch.float16)
    rank = config.get("rank", 8)

    # Main weights
    if gate_weight is None:
        gate_weight = 0.01 * torch.randn(
            config["in_features"],
            config["hidden_features"],
            device=device,
            dtype=dtype,
        )

    if up_weight is None:
        up_weight = 0.01 * torch.randn(
            config["in_features"],
            config["hidden_features"],
            device=device,
            dtype=dtype,
        )

    if down_weight is None:
        down_weight = 0.01 * torch.randn(
            config["hidden_features"],
            config["out_features"],
            device=device,
            dtype=dtype,
        )

    # Gate projection
    if gate_A is None:
        gate_A = torch.randn(
            config["in_features"], rank, device=device, dtype=dtype  # (768, r)
        ) / np.sqrt(config["in_features"])
    if gate_B is None:
        gate_B = torch.randn(
            rank, config["hidden_features"], device=device, dtype=dtype  # (r, 3072)
        ) / np.sqrt(rank)

    # Up projection
    if up_A is None:
        up_A = torch.randn(
            config["in_features"], rank, device=device, dtype=dtype  # (768, r)
        ) / np.sqrt(config["in_features"])
    if up_B is None:
        up_B = torch.randn(
            rank, config["hidden_features"], device=device, dtype=dtype  # (r, 3072)
        ) / np.sqrt(rank)

    # Down projection
    if down_A is None:
        down_A = torch.randn(
            config["hidden_features"], rank, device=device, dtype=dtype  # (3072, r)
        ) / np.sqrt(config["hidden_features"])
    if down_B is None:
        down_B = torch.randn(
            rank, config["out_features"], device=device, dtype=dtype  # (r, 768)
        ) / np.sqrt(rank)

    activation_fns = {
        "swiglu": SwiGLU.forward,
        "geglu_exact": GEGLU.forward,
    }

    if activation not in activation_fns:
        raise ValueError(f"Unsupported activation: {activation}")

    activation_fn = activation_fns[activation]

    def forward_fn(x):
        return LoRA_MLP.apply(
            x,
            gate_weight,
            None,
            gate_A,
            gate_B,
            1.0,
            up_weight,
            None,
            up_A,
            up_B,
            1.0,
            down_weight,
            None,
            down_A,
            down_B,
            1.0,
            activation_fn,
            True,  # inplace
        )

    return forward_fn


def create_lora_attention(config):
    """Factory function for creating LoRA attention layers.

    Args:
        config: Dictionary containing:
            - hidden_size: Model hidden size
            - num_heads: Number of attention heads
            - head_dim: Dimension per head
            - rank: LoRA rank
            - device: Device to place tensors on
            - dtype: Data type for tensors
    """
    device = config.get("device", "cuda")
    dtype = config.get("dtype", torch.float16)
    rank = config.get("rank", 8)

    # Initialize weights
    q_weight = torch.randn(
        config["hidden_size"], config["hidden_size"], device=device, dtype=dtype
    )
    k_weight = torch.randn(
        config["hidden_size"], config["hidden_size"], device=device, dtype=dtype
    )
    v_weight = torch.randn(
        config["hidden_size"], config["hidden_size"], device=device, dtype=dtype
    )

    # Initialize LoRA matrices
    q_A = torch.randn(
        config["hidden_size"], rank, device=device, dtype=dtype
    ) / np.sqrt(config["hidden_size"])
    q_B = torch.randn(
        rank, config["hidden_size"], device=device, dtype=dtype
    ) / np.sqrt(rank)

    k_A = torch.randn(
        config["hidden_size"], rank, device=device, dtype=dtype
    ) / np.sqrt(config["hidden_size"])
    k_B = torch.randn(
        rank, config["hidden_size"], device=device, dtype=dtype
    ) / np.sqrt(rank)

    v_A = torch.randn(
        config["hidden_size"], rank, device=device, dtype=dtype
    ) / np.sqrt(config["hidden_size"])
    v_B = torch.randn(
        rank, config["hidden_size"], device=device, dtype=dtype
    ) / np.sqrt(rank)

    def forward_fn(x):
        return LoRA_QKV.apply(
            x,
            q_weight,
            None,
            q_A,
            q_B,
            1.0,
            k_weight,
            None,
            k_A,
            k_B,
            1.0,
            v_weight,
            None,
            v_A,
            v_B,
            1.0,
            config["num_heads"],
            config["head_dim"],
        )

    return forward_fn
