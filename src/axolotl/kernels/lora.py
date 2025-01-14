"""
Module for definition of Low-Rank Adaptation (LoRA) Triton kernels.

See "LoRA: Low-Rank Adaptation of Large Language Models"
    (https://arxiv.org/abs/2106.09685).
"""
# pylint: disable=invalid-name

import numpy as np
import torch

from axolotl.kernels.quantize import dequantize

from .swiglu import swiglu_backward, swiglu_forward
from .utils import torch_amp_custom_bwd, torch_amp_custom_fwd


def get_lora_parameters(layer):
    """Extract LoRA parameters from layer."""
    weight = getattr(layer, "weight", None)
    quant_state = getattr(layer, "quant_state", None)
    lora_A = getattr(layer, "lora_A", None)
    lora_B = getattr(layer, "lora_B", None)
    scaling = getattr(layer, "scaling", 1.0)

    return weight, quant_state, lora_A, lora_B, scaling


def matmul_lora(X, W, W_quant, A, B, s, out=None):
    """
    Efficient fused matmul + LoRA computation.

    Args:
        X: Input tensor [*, in_features]
        W: Base weight matrix [out_features, in_features]
        W_quant: Quantization state for W
        A: LoRA A matrix [rank, in_features]
        B: LoRA B matrix [out_features, rank]
        s: LoRA scaling factor
        out: Optional output tensor for inplace operations
    """
    dtype = X.dtype
    W = dequantize(W.t(), W_quant)

    if X.dim() == 3:
        batch, seq_len, _ = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False

    out = torch.matmul(X, W, out=out)
    if W_quant is not None:
        del W

    if A is not None:
        A, B = A.t(), B.t()
        out += (X @ A.to(dtype)) @ (s * B.to(dtype))

    return out.view(batch, seq_len, -1) if reshape else out


class LoRA_MLP(torch.autograd.Function):
    """Optimized LoRA MLP implementation with memory management."""

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx,
        X: torch.Tensor,
        gate_weight: torch.Tensor,
        gate_quant: object | None,
        gate_A: torch.Tensor | None,
        gate_B: torch.Tensor | None,
        gate_scale: float,
        up_weight: torch.Tensor,
        up_quant: object | None,
        up_A: torch.Tensor | None,
        up_B: torch.Tensor | None,
        up_scale: float,
        down_weight: torch.Tensor,
        down_quant: object | None,
        down_A: torch.Tensor | None,
        down_B: torch.Tensor | None,
        down_scale: float,
        activation_fn,
        activation_fn_backward,
        inplace: bool = True,
    ) -> torch.Tensor:
        # Compute projections using helper function
        gate = matmul_lora(X, gate_weight, gate_quant, gate_A, gate_B, gate_scale)
        up = matmul_lora(X, up_weight, up_quant, up_A, up_B, up_scale)

        # Activation
        hidden = activation_fn(gate, up)

        # Down projection
        output = matmul_lora(
            hidden, down_weight, down_quant, down_A, down_B, down_scale
        )

        # Save tensors needed for backward
        ctx.save_for_backward(
            X, gate, up, hidden, gate_A, gate_B, up_A, up_B, down_A, down_B
        )
        ctx.scales = (gate_scale, up_scale, down_scale)
        ctx.quants = (gate_quant, up_quant, down_quant)
        ctx.weights = (gate_weight, up_weight, down_weight)
        ctx.activation_fn = activation_fn
        ctx.activation_fn_backward = activation_fn_backward
        ctx.inplace = inplace

        return output

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
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

        # Down projection gradients
        d_hidden = matmul_lora(
            grad_output,
            down_weight.t(),
            down_quant,
            down_B,
            down_A,
            down_scale,
            out=grad_output if ctx.inplace else None,
        )

        if down_A is not None:
            d_down_A = hidden.t() @ (grad_output @ down_B.t()) * down_scale
            d_down_B = (down_A.t() @ hidden.t()) @ grad_output * down_scale

        # Activation gradients
        d_gate, d_up = ctx.backward_activation_fn(d_hidden, gate, up)

        # Up/gate projection gradients
        grad_X = matmul_lora(d_up, up_weight.t(), up_quant, up_B, up_A, up_scale)
        if up_A is not None:
            d_up_A = X.t() @ (d_up @ up_B.t()) * up_scale
            d_up_B = (up_A.t() @ X.t()) @ d_up * up_scale

        grad_X_gate = matmul_lora(
            d_gate,
            gate_weight.t(),
            gate_quant,
            gate_B,
            gate_A,
            gate_scale,
            out=d_gate if ctx.inplace else None,
        )
        grad_X += grad_X_gate

        if gate_A is not None:
            d_gate_A = X.t() @ (d_gate @ gate_B.t()) * gate_scale
            d_gate_B = (gate_A.t() @ X.t()) @ d_gate * gate_scale

        return (
            grad_X,
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
            dequantize(q_weight, q_quant),
            out=X_reshaped if inplace else None,
        )
        if q_A is not None:
            Q += X_reshaped @ q_A @ q_B * q_scale

        # K projection
        K = torch.matmul(X_reshaped, dequantize(k_weight, k_quant))
        if k_A is not None:
            K += X_reshaped @ k_A @ k_B * k_scale

        # V projection
        V = torch.matmul(X_reshaped, dequantize(v_weight, v_quant))
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
            dequantize(q_weight, q_quant).t(),
            out=grad_Q if ctx.inplace else None,
        )
        if q_A is not None:
            d_q_A = X.t() @ (grad_Q @ q_B.t()) * q_scale
            d_q_B = (q_A.t() @ X.t()) @ grad_Q * q_scale
            grad_X += grad_Q @ q_B.t() @ (q_scale * q_A.t())

        # K gradients
        grad_X += torch.matmul(grad_K, dequantize(k_weight, k_quant).t())
        if k_A is not None:
            d_k_A = X.t() @ (grad_K @ k_B.t()) * k_scale
            d_k_B = (k_A.t() @ X.t()) @ grad_K * k_scale
            grad_X += grad_K @ k_B.t() @ (k_scale * k_A.t())

        # V gradients
        grad_X += torch.matmul(grad_V, dequantize(v_weight, v_quant).t())
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
            config["hidden_features"],
            config["in_features"],
            device=device,
            dtype=dtype,
        )

    if up_weight is None:
        up_weight = 0.01 * torch.randn(
            config["hidden_features"],
            config["in_features"],
            device=device,
            dtype=dtype,
        )

    if down_weight is None:
        down_weight = 0.01 * torch.randn(
            config["out_features"],
            config["hidden_features"],
            device=device,
            dtype=dtype,
        )

    # Gate projection
    if gate_A is None:
        gate_A = torch.randn(
            rank, config["in_features"], device=device, dtype=dtype  # (768, r)
        ) / np.sqrt(config["in_features"])
    if gate_B is None:
        gate_B = torch.randn(
            config["hidden_features"], rank, device=device, dtype=dtype  # (r, 3072)
        ) / np.sqrt(rank)

    # Up projection
    if up_A is None:
        up_A = torch.randn(
            rank, config["in_features"], device=device, dtype=dtype  # (768, r)
        ) / np.sqrt(config["in_features"])
    if up_B is None:
        up_B = torch.randn(
            config["hidden_features"], rank, device=device, dtype=dtype  # (r, 3072)
        ) / np.sqrt(rank)

    # Down projection
    if down_A is None:
        down_A = torch.randn(
            rank, config["hidden_features"], device=device, dtype=dtype  # (3072, r)
        ) / np.sqrt(config["hidden_features"])
    if down_B is None:
        down_B = torch.randn(
            config["out_features"], rank, device=device, dtype=dtype  # (r, 768)
        ) / np.sqrt(rank)

    activation_fns = {
        "swiglu": (swiglu_forward, swiglu_backward),
        # "geglu_exact": geglu_forward,
    }

    if activation not in activation_fns:
        raise ValueError(f"Unsupported activation: {activation}")

    activation_fn, backward_activation_fn = activation_fns[activation]

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
            backward_activation_fn,
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
