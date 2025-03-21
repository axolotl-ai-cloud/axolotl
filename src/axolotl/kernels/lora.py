"""
Module for definition of Low-Rank Adaptation (LoRA) Triton kernels.

See "LoRA: Low-Rank Adaptation of Large Language Models"
(https://arxiv.org/abs/2106.09685).

Credit to `unsloth` (https://unsloth.ai/) for inspiration for this implementation.
"""

# pylint: disable=invalid-name

from typing import Callable

import torch
from bitsandbytes.functional import QuantState
from torch import nn

from .geglu import geglu_backward, geglu_forward
from .quantize import dequantize
from .swiglu import swiglu_backward, swiglu_forward
from .utils import torch_amp_custom_bwd, torch_amp_custom_fwd


def get_lora_parameters(
    proj: nn.Module,
) -> tuple[
    torch.Tensor,
    QuantState | None,
    torch.Tensor | None,
    torch.Tensor | None,
    float | None,
]:
    """
    Gets LoRA parameters from a projection module.

    Args:
        proj: The projection module to extract parameters from.

    Returns:
        A tuple containing the base weight matrix, quantization state, LoRA A matrix,
        LoRA B matrix, and scaling factor. States and matrices may be None if not
        available.
    """
    # For DPO or disabled adapters
    base_layer = proj.base_layer if hasattr(proj, "base_layer") else proj
    W = base_layer.weight

    if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
        quant_state = getattr(W, "quant_state", None)
        return W, quant_state, None, None, None

    active_adapter = (
        proj.active_adapters[0]
        if hasattr(proj, "active_adapters")
        else proj.active_adapter
    )
    A = proj.lora_A[active_adapter].weight
    B = proj.lora_B[active_adapter].weight
    s = proj.scaling[active_adapter]

    quant_state = getattr(W, "quant_state", None)

    return W, quant_state, A, B, s


def matmul_lora(
    X: torch.Tensor,
    W: torch.Tensor,
    W_quant: QuantState,
    A: torch.Tensor,
    B: torch.Tensor,
    s: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
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

    Returns:
        Result of X @ W + X @ A @ B
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
    """Optimized LoRA MLP implementation."""

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
        activation_fn: Callable,
        activation_fn_backward: Callable,
        inplace: bool | None = True,
    ) -> torch.Tensor:
        """
        Forward pass for LoRA MLP.

        Args:
            ctx: Autograd context
            X: Input features
            gate_weight: Gate projection weight
            gate_quant: Gate quantization state
            gate_A: Gate LoRA A matrix
            gate_B: Gate LoRA B matrix
            gate_scale: Gate LoRA scale
            up_weight: Up-projection weight
            up_quant: Up-projection quantization state
            up_A: Up-projection LoRA A matrix
            up_B: Up-projection LoRA B matrix
            up_scale: Up-projection LoRA scale
            down_weight: Down-projection weight
            down_quant: Down-projection quantization state
            down_A: Down-projection LoRA A matrix
            down_B: Down-projection LoRA B matrix
            down_scale: Down-projection LoRA scale
            activation_fn: Forward activation function
            activation_fn_backward: Backward activation function
            inplace: Whether to perform operations in-place

        Returns:
            Output transformed by multi-layer perceptron and activation function
        """
        # Compute projections
        gate = matmul_lora(X, gate_weight, gate_quant, gate_A, gate_B, gate_scale)
        up = matmul_lora(X, up_weight, up_quant, up_A, up_B, up_scale)

        # Activation
        hidden = activation_fn(gate, up)

        # Down projection
        output = matmul_lora(
            hidden, down_weight, down_quant, down_A, down_B, down_scale
        )

        # Save for backward
        ctx.save_for_backward(X, gate, up, gate_A, gate_B, up_A, up_B, down_A, down_B)
        ctx.scales = (gate_scale, up_scale, down_scale)
        ctx.quants = (gate_quant, up_quant, down_quant)
        ctx.weights = (gate_weight, up_weight, down_weight)
        ctx.activation_fn = activation_fn
        ctx.activation_fn_backward = activation_fn_backward
        ctx.inplace = inplace

        return output

    @staticmethod
    @torch_amp_custom_bwd
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        None,
        None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
    ]:
        """
        Performs backward pass computation for LoRA MLP.

        Args:
            ctx: Context object storing tensors saved during forward pass
            grad_output: Gradient of loss with respect to layer output

        Returns:
            Tuple containing gradients for all inputs from forward pass:
            - Input gradient tensor (or `None`)
            - `None` for weights/quantization states
            - LoRA A/B matrix gradients (or `None`)
            - `None` for scaling factors
            - `None` for activation functions and flags
        """
        (
            X,
            gate,
            up,
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

        # Transpose all LoRA matrices
        gate_A, gate_B = (
            gate_A.t() if gate_A is not None else None,
            gate_B.t() if gate_B is not None else None,
        )
        up_A, up_B = (
            up_A.t() if up_A is not None else None,
            up_B.t() if up_B is not None else None,
        )
        down_A, down_B = (
            down_A.t() if down_A is not None else None,
            down_B.t() if down_B is not None else None,
        )

        # Reshape inputs
        batch, seq_len, hd = X.shape
        grad_output = grad_output.view(-1, grad_output.shape[-1])
        X = X.view(-1, X.shape[-1])
        gate = gate.view(-1, gate.shape[-1])
        up = up.view(-1, up.shape[-1])
        dtype = X.dtype

        # Down projection
        DW = matmul_lora(
            grad_output,
            down_weight.t(),
            down_quant,
            down_B,
            down_A,
            down_scale,
        )

        # Activation backward
        h, grad_gate, grad_up = ctx.activation_fn_backward(DW, gate, up)

        # Initialize and compute LoRA gradients
        d_down_A = d_down_B = d_up_A = d_up_B = d_gate_A = d_gate_B = None

        if down_A is not None:
            d_down_A = h.t() @ (grad_output @ down_B.t())
            d_down_B = (down_A.t() @ h.t()) @ grad_output
            d_down_A *= down_scale
            d_down_B *= down_scale

        if up_A is not None:
            d_up_A = X.t() @ (grad_up @ up_B.t())
            d_up_B = (up_A.t() @ X.t()) @ grad_up
            d_up_A *= up_scale
            d_up_B *= up_scale

        if gate_A is not None:
            d_gate_A = X.t() @ (grad_gate @ gate_B.t())
            d_gate_B = (gate_A.t() @ X.t()) @ grad_gate
            d_gate_A *= gate_scale
            d_gate_B *= gate_scale

        # Compute input gradients
        dX = torch.zeros_like(X) if ctx.needs_input_grad[0] else None

        if dX is not None:
            # Up projection gradients
            up_weight = dequantize(up_weight.t(), up_quant)
            if ctx.inplace:
                dX = torch.matmul(grad_up, up_weight.t(), out=X)
            else:
                dX = torch.matmul(grad_up, up_weight.t())
            del up_weight

            # Note the .to(dtype) only where mixing LoRA with base weights
            if up_A is not None:
                dX += grad_up @ up_B.to(dtype).t() @ (up_scale * up_A.to(dtype).t())

            # Gate projection gradients
            gate_weight = dequantize(gate_weight.t(), gate_quant)
            dX += grad_gate @ gate_weight.t()
            del gate_weight

            if gate_A is not None:
                dX += (
                    grad_gate
                    @ gate_B.to(dtype).t()
                    @ (gate_scale * gate_A.to(dtype).t())
                )

            # Reshape back
            dX = dX.view(batch, seq_len, hd)

        # Return gradients in correct order matching forward inputs
        return (
            dX,
            None,
            None,
            d_gate_A.t() if d_gate_A is not None else None,
            d_gate_B.t() if d_gate_B is not None else None,
            None,
            None,
            None,
            d_up_A.t() if d_up_A is not None else None,
            d_up_B.t() if d_up_B is not None else None,
            None,
            None,
            None,
            d_down_A.t() if d_down_A is not None else None,
            d_down_B.t() if d_down_B is not None else None,
            None,
            None,
            None,
            None,
        )


def apply_lora_mlp_swiglu(self, X: torch.Tensor, inplace: bool = True) -> torch.Tensor:
    """
    Applies LoRA to MLP layer with SwiGLU activation.

    Args:
        X: Input tensor for the MLP layer
        inplace: Whether to perform operations in-place to save memory

    Returns:
        Output tensor after applying LoRA-adapted MLP with SwiGLU activation
    """
    gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upW, upW_quant, upA, upB, upS = get_lora_parameters(self.up_proj)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)

    out = LoRA_MLP.apply(
        X,
        gateW,
        gateW_quant,
        gateA,
        gateB,
        gateS,
        upW,
        upW_quant,
        upA,
        upB,
        upS,
        downW,
        downW_quant,
        downA,
        downB,
        downS,
        swiglu_forward,
        swiglu_backward,
        inplace,
    )

    return out


def apply_lora_mlp_geglu(self, X: torch.Tensor, inplace: bool = True) -> torch.Tensor:
    """
    Applies LoRA to MLP layer with GEGLU activation.

    Args:
        X: Input tensor for the MLP layer
        inplace: Whether to perform operations in-place to save memory

    Returns:
        Output tensor after applying LoRA-adapted MLP with GEGLU activation
    """
    gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upW, upW_quant, upA, upB, upS = get_lora_parameters(self.up_proj)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)
    out = LoRA_MLP.apply(
        X,
        gateW,
        gateW_quant,
        gateA,
        gateB,
        gateS,
        upW,
        upW_quant,
        upA,
        upB,
        upS,
        downW,
        downW_quant,
        downA,
        downB,
        downS,
        geglu_forward,
        geglu_backward,
        inplace,
    )

    return out


class LoRA_QKV(torch.autograd.Function):
    """
    Optimized LoRA QKV implementation with quantization support.

    Implements efficient computation of query, key, value projections with LoRA,
    supporting quantization and memory optimization.
    """

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        X: torch.Tensor,
        q_weight: torch.Tensor,
        q_quant: QuantState | None,
        q_A: torch.Tensor | None,
        q_B: torch.Tensor | None,
        q_scale: float,
        k_weight: torch.Tensor,
        k_quant: QuantState | None,
        k_A: torch.Tensor | None,
        k_B: torch.Tensor | None,
        k_scale: float,
        v_weight: torch.Tensor,
        v_quant: QuantState | None,
        v_A: torch.Tensor | None,
        v_B: torch.Tensor | None,
        v_scale: float,
        inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computing Q, K, V projections with LoRA.

        Args:
            ctx: Autograd context
            X: Input tensor
            q_weight: Query projection weight
            q_quant: Query quantization state
            q_A: Query LoRA A matrix
            q_B: Query LoRA B matrix
            q_scale: Query LoRA scale
            k_weight: Key projection weight
            k_quant: Key quantization state
            k_A: Key LoRA A matrix
            k_B: Key LoRA B matrix
            k_scale: Key LoRA scale
            v_weight: Value projection weight
            v_quant: Value quantization state
            v_A: Value LoRA A matrix
            v_B: Value LoRA B matrix
            v_scale: Value LoRA scale
            inplace: Whether to perform operations in-place

        Returns:
            Tuple of (Query, Key, Value) projection tensors
        """
        Q = matmul_lora(X, q_weight, q_quant, q_A, q_B, q_scale)
        K = matmul_lora(X, k_weight, k_quant, k_A, k_B, k_scale)
        V = matmul_lora(X, v_weight, v_quant, v_A, v_B, v_scale)

        ctx.save_for_backward(X, q_A, q_B, k_A, k_B, v_A, v_B)
        ctx.scales = (q_scale, k_scale, v_scale)
        ctx.quants = (q_quant, k_quant, v_quant)
        ctx.weights = (q_weight, k_weight, v_weight)
        ctx.inplace = inplace

        return Q, K, V

    @staticmethod
    @torch_amp_custom_fwd
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        q_grad: torch.Tensor,
        k_grad: torch.Tensor,
        v_grad: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        None,
        None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
    ]:
        """
        Backward pass computing gradients for LoRA QKV.

        Args:
            ctx: Autograd context
            q_grad: Gradient for query projection
            k_grad: Gradient for key projection
            v_grad: Gradient for value projection

        Returns:
            Tuple containing gradients for all forward inputs
        """
        X, A_q, B_q, A_k, B_k, A_v, B_v = ctx.saved_tensors
        q_weight, k_weight, v_weight = ctx.weights
        q_quant, k_quant, v_quant = ctx.quants
        q_scale, k_scale, v_scale = ctx.scales
        dtype = X.dtype

        # Reshape gradients
        batch, seq_len = X.shape[:2]
        q_grad = q_grad.view(-1, q_grad.shape[-1])
        k_grad = k_grad.reshape(-1, k_grad.shape[-1])
        v_grad = v_grad.view(-1, v_grad.shape[-1])
        X = X.view(-1, X.shape[-1])

        # Pre-transpose X once
        X_t = X.t()

        # Initialize LoRA gradients as None
        d_A_q = d_B_q = d_A_k = d_B_k = d_A_v = d_B_v = None

        # Compute q path LoRA gradients if adapters exist
        if A_q is not None and B_q is not None:
            A_q_scaled = (q_scale * A_q).to(dtype)
            B_q_scaled = B_q.to(dtype)
            d_A_q = torch.mm(X_t, torch.mm(q_grad, B_q_scaled))
            d_B_q = torch.mm(torch.mm(A_q_scaled, X_t), q_grad)

        # Compute k path LoRA gradients if adapters exist
        if A_k is not None and B_k is not None:
            A_k_scaled = (k_scale * A_k).to(dtype)
            B_k_scaled = B_k.to(dtype)
            d_A_k = torch.mm(X_t, torch.mm(k_grad, B_k_scaled))
            d_B_k = torch.mm(torch.mm(A_k_scaled, X_t), k_grad)

        # Compute v path LoRA gradients if adapters exist
        if A_v is not None and B_v is not None:
            A_v_scaled = (v_scale * A_v).to(dtype)
            B_v_scaled = B_v.to(dtype)
            d_A_v = torch.mm(X_t, torch.mm(v_grad, B_v_scaled))
            d_B_v = torch.mm(torch.mm(A_v_scaled, X_t), v_grad)

        # Compute input gradient, reusing X memory if possible
        out_buffer = X if ctx.inplace else None

        # Q path
        q_weight_t = dequantize(q_weight, q_quant)
        grad_X = torch.mm(q_grad, q_weight_t, out=out_buffer)
        del q_weight
        del q_weight_t
        if A_q is not None and B_q is not None:
            grad_X.addmm_(q_grad, torch.mm(B_q_scaled, A_q_scaled))

        # K path
        k_weight_t = dequantize(k_weight, k_quant)
        grad_X.addmm_(k_grad, k_weight_t)
        del k_weight
        del k_weight_t
        if A_k is not None and B_k is not None:
            grad_X.addmm_(k_grad, torch.mm(B_k_scaled, A_k_scaled))

        # V path
        v_weight_t = dequantize(v_weight, v_quant)
        grad_X.addmm_(v_grad, v_weight_t)
        del v_weight
        del v_weight_t
        if A_v is not None and B_v is not None:
            grad_X.addmm_(v_grad, torch.mm(B_v_scaled, A_v_scaled))

        # Transpose gradients if needed
        if d_A_q is not None:
            d_A_q = d_A_q.t()
        if d_B_q is not None:
            d_B_q = d_B_q.t()
        if d_A_k is not None:
            d_A_k = d_A_k.t()
        if d_B_k is not None:
            d_B_k = d_B_k.t()
        if d_A_v is not None:
            d_A_v = d_A_v.t()
        if d_B_v is not None:
            d_B_v = d_B_v.t()

        return (
            grad_X.view(batch, seq_len, -1),
            None,
            None,
            d_A_q,
            d_B_q,
            None,
            None,
            None,
            d_A_k,
            d_B_k,
            None,
            None,
            None,
            d_A_v,
            d_B_v,
            None,
            None,
        )


def apply_lora_qkv(
    self, X: torch.Tensor, inplace: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies LoRA to compute Query, Key, Value projections.

    Args:
        X: Input tensor
        inplace: Whether to perform operations in-place

    Returns:
        Tuple of (Query, Key, Value) projection tensors
    """
    QW, QW_quant, QA, QB, QS = get_lora_parameters(self.q_proj)
    KW, KW_quant, KA, KB, KS = get_lora_parameters(self.k_proj)
    VW, VW_quant, VA, VB, VS = get_lora_parameters(self.v_proj)
    Q, K, V = LoRA_QKV.apply(
        X,
        QW,
        QW_quant,
        QA,
        QB,
        QS,
        KW,
        KW_quant,
        KA,
        KB,
        KS,
        VW,
        VW_quant,
        VA,
        VB,
        VS,
        inplace,
    )

    return Q, K, V


class LoRA_O(torch.autograd.Function):
    """Optimized LoRA implementation for output projection."""

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        X: torch.Tensor,
        W: torch.Tensor,
        W_quant: QuantState | None,
        A: torch.Tensor | None,
        B: torch.Tensor | None,
        S: float,
    ) -> torch.Tensor:
        """
        Forward pass for output projection with LoRA.

        Args:
            ctx: Autograd context
            X: Input tensor
            W: Output projection weight
            W_quant: Weight quantization state
            A: LoRA A matrix
            B: LoRA B matrix
            S: LoRA scaling factor

        Returns:
            Output projection tensor
        """
        XW = matmul_lora(X, W, W_quant, A, B, S)
        ctx.custom_saved_tensors = (
            W,
            W_quant,
            S,
        )
        ctx.save_for_backward(A, B, X)

        return XW

    @staticmethod
    @torch_amp_custom_bwd
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        dY: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        None,
        None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
    ]:
        """
        Backward pass computing gradients for LoRA output projection.

        Args:
            ctx: Autograd context
            dY: Gradient of loss with respect to output

        Returns:
            Tuple containing gradients for all forward inputs
        """
        W, W_quant, S = ctx.custom_saved_tensors
        A, B, X = ctx.saved_tensors

        batch, seq_len, hd = X.shape
        dY = dY.reshape(-1, dY.shape[-1])
        X = X.reshape(-1, X.shape[-1])
        dtype = X.dtype

        # Weight projection
        dY_X = X.t() @ dY
        d_A = S * dY_X @ B
        d_B = S * A @ dY_X

        # Get derivative for dX
        W = dequantize(W.t(), W_quant)
        dX = dY @ W.t()
        del W
        dX += dY @ B.to(dtype) @ (S * A.to(dtype))

        # W, W_quant, A, B, S
        return dX.view(batch, seq_len, hd), None, None, d_A.t(), d_B.t(), None


def apply_lora_o(self, X: torch.Tensor) -> torch.Tensor:
    """
    Applies LoRA to output projection layer.

    Args:
        X: Input tensor

    Returns:
        Transformed output tensor
    """
    OW, OW_quant, OA, OB, OS = get_lora_parameters(self.o_proj)
    output = LoRA_O.apply(X, OW, OW_quant, OA, OB, OS)

    return output
