"""
Module for definition of Low-Rank Adaptation (LoRA) Triton kernels.

See "LoRA: Low-Rank Adaptation of Large Language Models"
(https://arxiv.org/abs/2106.09685).

Also supports DoRA (Weight-Decomposed Low-Rank Adaptation):
See "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).

Credit to `unsloth` (https://unsloth.ai/) for inspiration for this implementation.
"""

from typing import Callable

import torch
from bitsandbytes.functional import QuantState
from torch import nn
from torch.distributed.tensor import DTensor

from .geglu import geglu_backward, geglu_forward
from .quantize import dequantize
from .swiglu import swiglu_backward, swiglu_forward
from .utils import torch_amp_custom_bwd, torch_amp_custom_fwd


def get_lora_parameters(
    proj: nn.Module,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    QuantState | torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    float | None,
    torch.Tensor | None,
    nn.Module | None,
    torch.Tensor | None,
]:
    """
    Gets LoRA parameters from a projection module.

    Args:
        proj: The projection module to extract parameters from.

    Returns:
        A tuple containing:
        - W: base weight tensor
        - b: base layer bias (or None)
        - quant_state: quantization state (or None)
        - A: LoRA A weight (or None)
        - B: LoRA B weight (or None)
        - s: LoRA scaling factor (or None)
        - lora_bias: LoRA B bias (or None)
        - dropout: dropout module (or None)
        - magnitude: DoRA magnitude vector (or None)
    """
    # For DPO or disabled adapters
    base_layer = proj.base_layer if hasattr(proj, "base_layer") else proj
    W = base_layer.weight
    b = base_layer.bias

    if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
        quant_state = getattr(W, "quant_state", None)
        if quant_state is None and W.dtype == torch.float8_e4m3fn:
            quant_state = getattr(base_layer, "weight_scale_inv", None)
        return W, b, quant_state, None, None, None, None, None, None

    quant_state = getattr(W, "quant_state", None)
    if quant_state is None and W.dtype == torch.float8_e4m3fn:
        quant_state = getattr(base_layer, "weight_scale_inv", None)

    active_adapter = (
        proj.active_adapters[0]
        if hasattr(proj, "active_adapters")
        else proj.active_adapter
    )

    linear_A = proj.lora_A[active_adapter]
    linear_B = proj.lora_B[active_adapter]

    # This manual unsharding is needed for FSDP2 + LoRA kernels compatibility.
    # We fuse linear layers + LoRA adapters calculations into a single
    # torch.autograd.Function, bypassing the registered unshard / reshard behavior.
    # Note that we don't apply resharding later in this module (it gets messy quickly),
    # but LoRA parameters are generally small enough that this is not an issue.
    if isinstance(linear_A.weight, DTensor):
        linear_A.unshard()
        linear_B.unshard()

    A = linear_A.weight
    B = linear_B.weight
    s = proj.scaling[active_adapter]

    # LoRA bias from lora_B (when bias="lora_only" or bias="all")
    lora_bias = linear_B.bias  # None if bias=False

    # Dropout module
    dropout = None
    if hasattr(proj, "lora_dropout") and active_adapter in proj.lora_dropout:
        dropout = proj.lora_dropout[active_adapter]

    # DoRA magnitude vector
    magnitude = None
    if (
        hasattr(proj, "lora_magnitude_vector")
        and proj.lora_magnitude_vector
        and active_adapter in proj.lora_magnitude_vector
    ):
        mag_layer = proj.lora_magnitude_vector[active_adapter]
        magnitude = mag_layer.weight
        # FSDP2 DTensor unshard for magnitude vector
        if isinstance(magnitude, DTensor):
            magnitude = magnitude.full_tensor()

    return W, b, quant_state, A, B, s, lora_bias, dropout, magnitude


def _apply_dropout(
    dropout: nn.Module | None, X: torch.Tensor, training: bool
) -> torch.Tensor | None:
    """Apply dropout to X if dropout module exists and is active.

    Returns X_drop (different tensor) or None if no dropout needed.
    """
    if dropout is None or isinstance(dropout, nn.Identity) or not training:
        return None
    return dropout(X)


_USE_TRITON_DORA: bool | None = None


def _should_use_triton_dora() -> bool:
    """Check if Triton DoRA kernel is available."""
    global _USE_TRITON_DORA
    if _USE_TRITON_DORA is None:
        try:
            from .dora import triton_dora_scale  # noqa: F401

            _USE_TRITON_DORA = True
        except (ImportError, RuntimeError):
            _USE_TRITON_DORA = False
    return _USE_TRITON_DORA


def _compute_dora_scale(
    W: torch.Tensor,
    W_quant: QuantState | torch.Tensor | None,
    A: torch.Tensor,
    B: torch.Tensor,
    s: float,
    magnitude: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute DoRA magnitude/norm scaling factor with optional caching.

    Uses Triton kernel when available for better performance.
    Caches weight_norm on the magnitude tensor. Cache invalidated when
    LoRA A/B data changes (after optimizer step).

    Returns:
        mag_norm_scale: [out_features] tensor = magnitude / ||W + s * B @ A||_2
    """
    # Check cache on magnitude tensor (avoids expensive norm recomputation)
    # Use tensor._version which increments on any in-place modification
    # (data_ptr doesn't change when optimizers update params in-place)
    cache = getattr(magnitude, "_dora_cache", None)
    if cache is not None:
        cached_a_ver, cached_b_ver, cached_norm = cache
        if cached_a_ver == A._version and cached_b_ver == B._version:
            return magnitude.to(dtype) / cached_norm

    # Full recomputation - try Triton first
    if _should_use_triton_dora() and W.is_cuda:
        from .dora import triton_dora_scale

        result = triton_dora_scale(W, W_quant, A, B, s, magnitude, dtype)
        weight_norm = (magnitude.to(dtype) / result).detach()
        magnitude._dora_cache = (A._version, B._version, weight_norm)
        return result

    # PyTorch fallback
    W_full = dequantize(W.t(), W_quant).t().to(dtype)  # [out, in]
    lora_weight = B.to(dtype) @ A.to(dtype)
    combined = W_full + s * lora_weight
    weight_norm = torch.linalg.norm(combined, dim=1).to(dtype)
    weight_norm = weight_norm.detach()

    magnitude._dora_cache = (A._version, B._version, weight_norm)

    return magnitude.to(dtype) / weight_norm


def _compute_dora_scale_cached(
    proj: nn.Module,
    W: torch.Tensor,
    W_quant: QuantState | torch.Tensor | None,
    A: torch.Tensor,
    B: torch.Tensor,
    s: float,
    magnitude: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute DoRA scale with caching. Recomputes only when LoRA params change.

    Caches the weight norm on the projection module. The cache is invalidated
    when LoRA A/B data pointers change (indicating an optimizer step occurred).
    """
    cache = getattr(proj, "_dora_norm_cache", None)
    if cache is not None:
        cached_a_ver, cached_b_ver, cached_norm = cache
        if cached_a_ver == A._version and cached_b_ver == B._version:
            return magnitude.to(dtype) / cached_norm

    # Cache miss - full recomputation
    W_full = dequantize(W.t(), W_quant).t().to(dtype)
    lora_weight = B.to(dtype) @ A.to(dtype)
    combined = W_full + s * lora_weight
    weight_norm = torch.linalg.norm(combined, dim=1).to(dtype).detach()

    proj._dora_norm_cache = (A._version, B._version, weight_norm)

    return magnitude.to(dtype) / weight_norm


def matmul_lora(
    X: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None,
    W_quant: QuantState | torch.Tensor | None,
    A: torch.Tensor | None,
    B: torch.Tensor | None,
    s: float | None,
    out: torch.Tensor | None = None,
    X_drop: torch.Tensor | None = None,
    lora_bias: torch.Tensor | None = None,
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
        X_drop: Optional dropout-applied input for LoRA path (if None, uses X)
        lora_bias: Optional LoRA B layer bias [out_features]

    Returns:
        Result of X @ W + s * X_drop @ A @ B + b + s * lora_bias
    """
    dtype = X.dtype
    W = dequantize(W.t(), W_quant)

    reshape = False
    if X.dim() == 3:
        batch, seq_len, _ = X.shape
        X = X.view(-1, X.shape[-1])
        if X_drop is not None:
            X_drop = X_drop.view(-1, X_drop.shape[-1])
        reshape = True

    out = torch.matmul(X, W, out=out)
    if W_quant is not None:
        del W

    if A is not None:
        X_lora = X_drop if X_drop is not None else X
        A, B = A.t().to(dtype), B.t().to(dtype)  # type: ignore[union-attr]
        out += s * X_lora @ A @ B
        if lora_bias is not None:
            out += s * lora_bias

    if b is not None:
        out += b

    return out.view(batch, seq_len, -1) if reshape else out


class LoRA_MLP(torch.autograd.Function):
    """Optimized LoRA MLP implementation.

    Supports bias, dropout, and DoRA. Dropout is applied to the input for
    gate/up projections. The down projection uses hidden states (post-activation)
    as input, so dropout is not applied there.
    """

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx,
        X: torch.Tensor,
        X_drop: torch.Tensor | None,
        # Gate params
        gate_weight: torch.Tensor,
        gate_bias: torch.Tensor | None,
        gate_quant: QuantState | None,
        gate_A: torch.Tensor | None,
        gate_B: torch.Tensor | None,
        gate_scale: float,
        gate_lora_bias: torch.Tensor | None,
        gate_magnitude: torch.Tensor | None,
        # Up params
        up_weight: torch.Tensor,
        up_bias: torch.Tensor | None,
        up_quant: QuantState | None,
        up_A: torch.Tensor | None,
        up_B: torch.Tensor | None,
        up_scale: float,
        up_lora_bias: torch.Tensor | None,
        up_magnitude: torch.Tensor | None,
        # Down params
        down_weight: torch.Tensor,
        down_bias: torch.Tensor | None,
        down_quant: QuantState | None,
        down_A: torch.Tensor | None,
        down_B: torch.Tensor | None,
        down_scale: float,
        down_lora_bias: torch.Tensor | None,
        down_magnitude: torch.Tensor | None,
        # Activation and flags
        activation_fn: Callable,
        activation_fn_backward: Callable,
        inplace: bool | None = True,
    ) -> torch.Tensor:
        has_dropout = X_drop is not None
        has_dora = gate_magnitude is not None
        dtype = X.dtype
        X_lora = X_drop if has_dropout else X

        if has_dora:
            # Gate with DoRA
            gate_base = matmul_lora(X, gate_weight, None, gate_quant, None, None, None)
            gate_lora = _lora_only(
                X_lora, gate_A, gate_B, gate_scale, gate_lora_bias, dtype
            )
            gate_mag_scale = _compute_dora_scale(
                gate_weight,
                gate_quant,
                gate_A,
                gate_B,
                gate_scale,
                gate_magnitude,
                dtype,
            )
            gate = gate_mag_scale.unsqueeze(0) * (gate_base + gate_lora)
            if gate_bias is not None:
                gate = gate + gate_bias

            # Up with DoRA
            up_base = matmul_lora(X, up_weight, None, up_quant, None, None, None)
            up_lora = _lora_only(X_lora, up_A, up_B, up_scale, up_lora_bias, dtype)
            up_mag_scale = _compute_dora_scale(
                up_weight, up_quant, up_A, up_B, up_scale, up_magnitude, dtype
            )
            up = up_mag_scale.unsqueeze(0) * (up_base + up_lora)
            if up_bias is not None:
                up = up + up_bias

            gate_combined = gate_base + gate_lora
            up_combined = up_base + up_lora
        else:
            gate = matmul_lora(
                X,
                gate_weight,
                gate_bias,
                gate_quant,
                gate_A,
                gate_B,
                gate_scale,
                X_drop=X_drop,
                lora_bias=gate_lora_bias,
            )
            up = matmul_lora(
                X,
                up_weight,
                up_bias,
                up_quant,
                up_A,
                up_B,
                up_scale,
                X_drop=X_drop,
                lora_bias=up_lora_bias,
            )

        # Activation
        hidden = activation_fn(gate, up)

        # Down projection (no dropout on hidden - it's an intermediate)
        if has_dora:
            down_base = matmul_lora(
                hidden, down_weight, None, down_quant, None, None, None
            )
            down_lora = _lora_only(
                hidden, down_A, down_B, down_scale, down_lora_bias, dtype
            )
            down_mag_scale = _compute_dora_scale(
                down_weight,
                down_quant,
                down_A,
                down_B,
                down_scale,
                down_magnitude,
                dtype,
            )
            down_combined = down_base + down_lora
            output = down_mag_scale.unsqueeze(0) * down_combined
            if down_bias is not None:
                output = output + down_bias
        else:
            output = matmul_lora(
                hidden,
                down_weight,
                down_bias,
                down_quant,
                down_A,
                down_B,
                down_scale,
                lora_bias=down_lora_bias,
            )

        # Save for backward
        if has_dora:
            ctx.save_for_backward(
                X,
                X_drop if has_dropout else X,
                gate,
                up,
                gate_A.to(dtype) if gate_A is not None else gate_A,
                gate_B.to(dtype) if gate_B is not None else gate_B,
                up_A.to(dtype) if up_A is not None else up_A,
                up_B.to(dtype) if up_B is not None else up_B,
                down_A.to(dtype) if down_A is not None else down_A,
                down_B.to(dtype) if down_B is not None else down_B,
                gate_magnitude,
                up_magnitude,
                down_magnitude,
                gate_mag_scale,
                up_mag_scale,
                down_mag_scale,
                gate_combined,
                up_combined,
                down_combined,
                gate_lora_bias,
                up_lora_bias,
                down_lora_bias,
            )
        else:
            # Pre-convert LoRA matrices to compute dtype for backward
            dtype = X.dtype
            ctx.save_for_backward(
                X,
                X_drop if has_dropout else X,
                gate,
                up,
                gate_A.to(dtype) if gate_A is not None else gate_A,
                gate_B.to(dtype) if gate_B is not None else gate_B,
                up_A.to(dtype) if up_A is not None else up_A,
                up_B.to(dtype) if up_B is not None else up_B,
                down_A.to(dtype) if down_A is not None else down_A,
                down_B.to(dtype) if down_B is not None else down_B,
                gate_lora_bias,
                up_lora_bias,
                down_lora_bias,
            )

        ctx.scales = (gate_scale, up_scale, down_scale)
        ctx.quants = (gate_quant, up_quant, down_quant)
        ctx.weights = (gate_weight, up_weight, down_weight)
        ctx.activation_fn = activation_fn
        ctx.activation_fn_backward = activation_fn_backward
        ctx.inplace = inplace
        ctx.has_dropout = has_dropout
        ctx.has_dora = has_dora

        return output

    @staticmethod
    @torch_amp_custom_bwd
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ):
        gate_scale, up_scale, down_scale = ctx.scales
        gate_quant, up_quant, down_quant = ctx.quants
        gate_weight, up_weight, down_weight = ctx.weights
        has_dropout = ctx.has_dropout
        has_dora = ctx.has_dora

        if has_dora:
            (
                X,
                X_lora,
                gate,
                up,
                gate_A,
                gate_B,
                up_A,
                up_B,
                down_A,
                down_B,
                gate_magnitude,
                up_magnitude,
                down_magnitude,
                gate_mag_scale,
                up_mag_scale,
                down_mag_scale,
                gate_combined,
                up_combined,
                down_combined,
                gate_lora_bias,
                up_lora_bias,
                down_lora_bias,
            ) = ctx.saved_tensors
        else:
            (
                X,
                X_lora,
                gate,
                up,
                gate_A,
                gate_B,
                up_A,
                up_B,
                down_A,
                down_B,
                gate_lora_bias,
                up_lora_bias,
                down_lora_bias,
            ) = ctx.saved_tensors
            gate_magnitude = up_magnitude = down_magnitude = None
            gate_mag_scale = up_mag_scale = down_mag_scale = None
            gate_combined = up_combined = down_combined = None

        # Transpose all LoRA matrices
        gate_A_t = gate_A.t() if gate_A is not None else None
        gate_B_t = gate_B.t() if gate_B is not None else None
        up_A_t = up_A.t() if up_A is not None else None
        up_B_t = up_B.t() if up_B is not None else None
        down_A_t = down_A.t() if down_A is not None else None
        down_B_t = down_B.t() if down_B is not None else None

        # Reshape inputs
        batch, seq_len, hd = X.shape
        grad_output = grad_output.view(-1, grad_output.shape[-1])
        X = X.view(-1, X.shape[-1])
        X_lora = X_lora.view(-1, X_lora.shape[-1])
        gate = gate.view(-1, gate.shape[-1])
        up = up.view(-1, up.shape[-1])

        # DoRA magnitude gradients for down projection
        d_gate_mag = d_up_mag = d_down_mag = None
        d_gate_lora_bias = d_up_lora_bias = d_down_lora_bias = None

        if has_dora:
            down_combined_flat = down_combined.view(-1, down_combined.shape[-1])
            d_down_mag = (
                (grad_output * down_combined_flat).sum(dim=0)
                * down_mag_scale
                / down_magnitude
            )
            grad_output = grad_output * down_mag_scale.unsqueeze(0)

        # Down lora bias gradient
        if down_lora_bias is not None:
            d_down_lora_bias = down_scale * grad_output.sum(dim=0)

        # Down projection backward
        grad_down = matmul_lora(
            grad_output,
            down_weight.t(),
            None,
            down_quant,
            down_B_t,
            down_A_t,
            down_scale,
        )

        # Activation backward
        h, grad_gate, grad_up = ctx.activation_fn_backward(grad_down, gate, up)

        # DoRA magnitude gradients for gate and up
        if has_dora:
            gate_combined_flat = gate_combined.view(-1, gate_combined.shape[-1])
            up_combined_flat = up_combined.view(-1, up_combined.shape[-1])
            d_gate_mag = (
                (grad_gate * gate_combined_flat).sum(dim=0)
                * gate_mag_scale
                / gate_magnitude
            )
            d_up_mag = (
                (grad_up * up_combined_flat).sum(dim=0) * up_mag_scale / up_magnitude
            )
            grad_gate = grad_gate * gate_mag_scale.unsqueeze(0)
            grad_up = grad_up * up_mag_scale.unsqueeze(0)

        # LoRA bias gradients for gate and up
        if gate_lora_bias is not None:
            d_gate_lora_bias = gate_scale * grad_gate.sum(dim=0)
        if up_lora_bias is not None:
            d_up_lora_bias = up_scale * grad_up.sum(dim=0)

        # LoRA parameter gradients (already in compute dtype from forward)
        # Compute grad @ B once per projection, reuse for dA and dX_lora
        # Note: _t suffix means transposed from saved shape (A_t = A.t(), etc.)
        d_down_A = d_down_B = d_up_A = d_up_B = d_gate_A = d_gate_B = None
        grad_B_up = grad_B_gate = None

        if down_A_t is not None and down_B_t is not None:
            grad_B_down = grad_output @ down_B_t.t()  # reused in matmul_lora above too
            d_down_A = torch.empty_like(down_A_t)
            d_down_B = torch.empty_like(down_B_t)
            d_down_A.addmm_(h.t(), grad_B_down, alpha=down_scale, beta=0)
            d_down_B.addmm_(down_A_t.t() @ h.t(), grad_output, alpha=down_scale, beta=0)

        if up_A_t is not None and up_B_t is not None:
            grad_B_up = grad_up @ up_B_t.t()  # [T, rank] — reuse for dX
            d_up_A = torch.empty_like(up_A_t)
            d_up_B = torch.empty_like(up_B_t)
            d_up_A.addmm_(X_lora.t(), grad_B_up, alpha=up_scale, beta=0)
            d_up_B.addmm_(up_A_t.t() @ X_lora.t(), grad_up, alpha=up_scale, beta=0)

        if gate_A_t is not None and gate_B_t is not None:
            grad_B_gate = grad_gate @ gate_B_t.t()  # [T, rank] — reuse for dX
            d_gate_A = torch.empty_like(gate_A_t)
            d_gate_B = torch.empty_like(gate_B_t)
            d_gate_A.addmm_(X_lora.t(), grad_B_gate, alpha=gate_scale, beta=0)
            d_gate_B.addmm_(
                gate_A_t.t() @ X_lora.t(), grad_gate, alpha=gate_scale, beta=0
            )

        # Compute input gradients
        dX = None
        dX_drop = None

        if ctx.needs_input_grad[0]:
            # Base path gradients through gate and up
            up_weight_deq = dequantize(up_weight.t(), up_quant)
            if ctx.inplace:
                dX = torch.matmul(grad_up, up_weight_deq.t(), out=X)
            else:
                dX = torch.matmul(grad_up, up_weight_deq.t())
            del up_weight_deq

            gate_weight_deq = dequantize(gate_weight, gate_quant)
            dX += grad_gate @ gate_weight_deq
            del gate_weight_deq

            # LoRA path: reuse grad_B_up and grad_B_gate from above
            if has_dropout:
                dX_drop = torch.zeros_like(X_lora)
                if grad_B_up is not None:
                    dX_drop.addmm_(grad_B_up, up_A_t.t(), alpha=up_scale)  # type: ignore[union-attr]
                if grad_B_gate is not None:
                    dX_drop.addmm_(grad_B_gate, gate_A_t.t(), alpha=gate_scale)  # type: ignore[union-attr]
                dX_drop = dX_drop.view(batch, seq_len, hd)
            else:
                if grad_B_up is not None:
                    dX.addmm_(grad_B_up, up_A_t.t(), alpha=up_scale)  # type: ignore[union-attr]
                if grad_B_gate is not None:
                    dX.addmm_(grad_B_gate, gate_A_t.t(), alpha=gate_scale)  # type: ignore[union-attr]

            dX = dX.view(batch, seq_len, hd)

        # Return gradients matching forward input order:
        # X, X_drop,
        # gate: weight, bias, quant, A, B, scale, lora_bias, magnitude
        # up: weight, bias, quant, A, B, scale, lora_bias, magnitude
        # down: weight, bias, quant, A, B, scale, lora_bias, magnitude
        # activation_fn, activation_fn_backward, inplace
        return (
            dX,
            dX_drop,
            # Gate
            None,
            None,
            None,
            d_gate_A.t() if d_gate_A is not None else None,
            d_gate_B.t() if d_gate_B is not None else None,
            None,
            d_gate_lora_bias,
            d_gate_mag,
            # Up
            None,
            None,
            None,
            d_up_A.t() if d_up_A is not None else None,
            d_up_B.t() if d_up_B is not None else None,
            None,
            d_up_lora_bias,
            d_up_mag,
            # Down
            None,
            None,
            None,
            d_down_A.t() if d_down_A is not None else None,
            d_down_B.t() if d_down_B is not None else None,
            None,
            d_down_lora_bias,
            d_down_mag,
            # Activation fns and flags
            None,
            None,
            None,
        )


def apply_lora_mlp_swiglu(self, X: torch.Tensor, inplace: bool = True) -> torch.Tensor:
    """Applies LoRA to MLP layer with SwiGLU activation.

    Supports bias, dropout, and DoRA.
    """
    gateW, gateb, gateW_quant, gateA, gateB, gateS, gateLB, gateDrop, gateMag = (
        get_lora_parameters(self.gate_proj)
    )
    upW, upb, upW_quant, upA, upB, upS, upLB, upDrop, upMag = get_lora_parameters(
        self.up_proj
    )
    downW, downb, downW_quant, downA, downB, downS, downLB, downDrop, downMag = (
        get_lora_parameters(self.down_proj)
    )

    # Shared dropout mask for gate and up (same input)
    X_drop = _apply_dropout(gateDrop, X, self.training)

    out = LoRA_MLP.apply(
        X,
        X_drop,
        # Gate
        gateW,
        gateb,
        gateW_quant,
        gateA,
        gateB,
        gateS,
        gateLB,
        gateMag,
        # Up
        upW,
        upb,
        upW_quant,
        upA,
        upB,
        upS,
        upLB,
        upMag,
        # Down
        downW,
        downb,
        downW_quant,
        downA,
        downB,
        downS,
        downLB,
        downMag,
        # Activation and flags
        swiglu_forward,
        swiglu_backward,
        inplace,
    )

    return out


def apply_lora_mlp_geglu(self, X: torch.Tensor, inplace: bool = True) -> torch.Tensor:
    """Applies LoRA to MLP layer with GEGLU activation.

    Supports bias, dropout, and DoRA.
    """
    gateW, gateb, gateW_quant, gateA, gateB, gateS, gateLB, gateDrop, gateMag = (
        get_lora_parameters(self.gate_proj)
    )
    upW, upb, upW_quant, upA, upB, upS, upLB, upDrop, upMag = get_lora_parameters(
        self.up_proj
    )
    downW, downb, downW_quant, downA, downB, downS, downLB, downDrop, downMag = (
        get_lora_parameters(self.down_proj)
    )

    X_drop = _apply_dropout(gateDrop, X, self.training)

    out = LoRA_MLP.apply(
        X,
        X_drop,
        # Gate
        gateW,
        gateb,
        gateW_quant,
        gateA,
        gateB,
        gateS,
        gateLB,
        gateMag,
        # Up
        upW,
        upb,
        upW_quant,
        upA,
        upB,
        upS,
        upLB,
        upMag,
        # Down
        downW,
        downb,
        downW_quant,
        downA,
        downB,
        downS,
        downLB,
        downMag,
        # Activation and flags
        geglu_forward,
        geglu_backward,
        inplace,
    )

    return out


class LoRA_QKV(torch.autograd.Function):
    """
    Optimized LoRA QKV implementation with quantization support.

    Supports bias, dropout, and DoRA (Weight-Decomposed Low-Rank Adaptation).
    Dropout is applied outside this Function so autograd handles its backward.
    """

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        X: torch.Tensor,
        X_drop: torch.Tensor | None,
        # Q params
        q_weight: torch.Tensor,
        q_bias: torch.Tensor | None,
        q_quant: QuantState | None,
        q_A: torch.Tensor | None,
        q_B: torch.Tensor | None,
        q_scale: float,
        q_lora_bias: torch.Tensor | None,
        q_magnitude: torch.Tensor | None,
        # K params
        k_weight: torch.Tensor,
        k_bias: torch.Tensor | None,
        k_quant: QuantState | None,
        k_A: torch.Tensor | None,
        k_B: torch.Tensor | None,
        k_scale: float,
        k_lora_bias: torch.Tensor | None,
        k_magnitude: torch.Tensor | None,
        # V params
        v_weight: torch.Tensor,
        v_bias: torch.Tensor | None,
        v_quant: QuantState | None,
        v_A: torch.Tensor | None,
        v_B: torch.Tensor | None,
        v_scale: float,
        v_lora_bias: torch.Tensor | None,
        v_magnitude: torch.Tensor | None,
        # Flags
        inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        has_dropout = X_drop is not None
        has_dora = q_magnitude is not None

        if has_dora:
            dtype = X.dtype
            X_lora = X_drop if has_dropout else X

            # Compute Q with DoRA
            Q_base = matmul_lora(X, q_weight, None, q_quant, None, None, None)
            Q_lora = _lora_only(X_lora, q_A, q_B, q_scale, q_lora_bias, dtype)
            q_mag_scale = _compute_dora_scale(
                q_weight, q_quant, q_A, q_B, q_scale, q_magnitude, dtype
            )
            Q = q_mag_scale.unsqueeze(0) * (Q_base + Q_lora)
            if q_bias is not None:
                Q = Q + q_bias

            # Compute K with DoRA
            K_base = matmul_lora(X, k_weight, None, k_quant, None, None, None)
            K_lora = _lora_only(X_lora, k_A, k_B, k_scale, k_lora_bias, dtype)
            k_mag_scale = _compute_dora_scale(
                k_weight, k_quant, k_A, k_B, k_scale, k_magnitude, dtype
            )
            K = k_mag_scale.unsqueeze(0) * (K_base + K_lora)
            if k_bias is not None:
                K = K + k_bias

            # Compute V with DoRA
            V_base = matmul_lora(X, v_weight, None, v_quant, None, None, None)
            V_lora = _lora_only(X_lora, v_A, v_B, v_scale, v_lora_bias, dtype)
            v_mag_scale = _compute_dora_scale(
                v_weight, v_quant, v_A, v_B, v_scale, v_magnitude, dtype
            )
            V = v_mag_scale.unsqueeze(0) * (V_base + V_lora)
            if v_bias is not None:
                V = V + v_bias

            # Save for backward: need combined (base+lora) and mag_scale for DoRA grads
            Q_combined = Q_base + Q_lora
            K_combined = K_base + K_lora
            V_combined = V_base + V_lora

            ctx.save_for_backward(
                X,
                X_drop if has_dropout else X,
                q_A.to(dtype) if q_A is not None else q_A,
                q_B.to(dtype) if q_B is not None else q_B,
                k_A.to(dtype) if k_A is not None else k_A,
                k_B.to(dtype) if k_B is not None else k_B,
                v_A.to(dtype) if v_A is not None else v_A,
                v_B.to(dtype) if v_B is not None else v_B,
                q_magnitude,
                k_magnitude,
                v_magnitude,
                q_mag_scale,
                k_mag_scale,
                v_mag_scale,
                Q_combined,
                K_combined,
                V_combined,
                q_lora_bias,
                k_lora_bias,
                v_lora_bias,
            )
        else:
            # Standard LoRA (with optional dropout and bias)
            Q = matmul_lora(
                X,
                q_weight,
                q_bias,
                q_quant,
                q_A,
                q_B,
                q_scale,
                X_drop=X_drop,
                lora_bias=q_lora_bias,
            )
            K = matmul_lora(
                X,
                k_weight,
                k_bias,
                k_quant,
                k_A,
                k_B,
                k_scale,
                X_drop=X_drop,
                lora_bias=k_lora_bias,
            )
            V = matmul_lora(
                X,
                v_weight,
                v_bias,
                v_quant,
                v_A,
                v_B,
                v_scale,
                X_drop=X_drop,
                lora_bias=v_lora_bias,
            )

            # Pre-convert LoRA matrices to compute dtype to avoid
            # redundant fp32→bf16 conversion in backward
            dtype = X.dtype
            ctx.save_for_backward(
                X,
                X_drop if has_dropout else X,
                q_A.to(dtype) if q_A is not None else q_A,
                q_B.to(dtype) if q_B is not None else q_B,
                k_A.to(dtype) if k_A is not None else k_A,
                k_B.to(dtype) if k_B is not None else k_B,
                v_A.to(dtype) if v_A is not None else v_A,
                v_B.to(dtype) if v_B is not None else v_B,
                q_lora_bias,
                k_lora_bias,
                v_lora_bias,
            )

        ctx.scales = (q_scale, k_scale, v_scale)
        ctx.quants = (q_quant, k_quant, v_quant)
        ctx.weights = (q_weight, k_weight, v_weight)
        ctx.inplace = inplace
        ctx.has_dropout = has_dropout
        ctx.has_dora = has_dora

        return Q, K, V

    @staticmethod
    @torch_amp_custom_bwd
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        q_grad: torch.Tensor,
        k_grad: torch.Tensor,
        v_grad: torch.Tensor,
    ):
        q_weight, k_weight, v_weight = ctx.weights
        q_quant, k_quant, v_quant = ctx.quants
        q_scale, k_scale, v_scale = ctx.scales
        has_dropout = ctx.has_dropout
        has_dora = ctx.has_dora

        if has_dora:
            (
                X,
                X_lora,
                A_q,
                B_q,
                A_k,
                B_k,
                A_v,
                B_v,
                q_magnitude,
                k_magnitude,
                v_magnitude,
                q_mag_scale,
                k_mag_scale,
                v_mag_scale,
                Q_combined,
                K_combined,
                V_combined,
                q_lora_bias,
                k_lora_bias,
                v_lora_bias,
            ) = ctx.saved_tensors
        else:
            (
                X,
                X_lora,
                A_q,
                B_q,
                A_k,
                B_k,
                A_v,
                B_v,
                q_lora_bias,
                k_lora_bias,
                v_lora_bias,
            ) = ctx.saved_tensors
            q_magnitude = k_magnitude = v_magnitude = None
            q_mag_scale = k_mag_scale = v_mag_scale = None
            Q_combined = K_combined = V_combined = None

        batch, seq_len = X.shape[:2]
        q_grad = q_grad.view(-1, q_grad.shape[-1])
        k_grad = k_grad.reshape(-1, k_grad.shape[-1])
        v_grad = v_grad.view(-1, v_grad.shape[-1])
        X = X.view(-1, X.shape[-1])
        X_lora = X_lora.view(-1, X_lora.shape[-1])

        # DoRA: scale gradients through mag_norm_scale
        d_q_mag = d_k_mag = d_v_mag = None
        d_q_lora_bias = d_k_lora_bias = d_v_lora_bias = None

        if has_dora:
            Q_combined = Q_combined.view(-1, Q_combined.shape[-1])
            K_combined = K_combined.view(-1, K_combined.shape[-1])
            V_combined = V_combined.view(-1, V_combined.shape[-1])

            # Magnitude gradients: d_mag = sum_t(grad * combined) / weight_norm
            # Since mag_scale = magnitude / weight_norm, and weight_norm is detached:
            # d_magnitude = sum_t(grad * combined) * (1 / weight_norm)
            # But we have mag_scale = magnitude / weight_norm
            # d_mag_scale_j = grad_j * combined_j (per element)
            # d_magnitude_j = d_mag_scale_j / weight_norm_j = sum_t(grad * combined) / weight_norm
            # Simpler: d_magnitude = sum(grad * combined, dim=0) * mag_scale / magnitude
            # Actually: mag_scale = m/wn, d_output/d_m = combined/wn = combined * mag_scale/m
            # d_m = sum(grad * combined * mag_scale / m, dim=0) ... no.
            # Let's be precise: output = mag_scale * combined, mag_scale = m / wn (wn detached)
            # d_loss/d_m = d_loss/d_output * d_output/d_m = sum_t(grad_t * combined_t / wn)
            # = sum_t(grad_t * combined_t) * (1/wn) = sum_t(grad_t * combined_t) * mag_scale / m
            # Or just: d_m = sum(grad * combined, dim=0) / weight_norm
            # Since we don't have weight_norm saved, use mag_scale/magnitude:
            # 1/wn = mag_scale/magnitude
            d_q_mag = (q_grad * Q_combined).sum(dim=0) * q_mag_scale / q_magnitude
            d_k_mag = (k_grad * K_combined).sum(dim=0) * k_mag_scale / k_magnitude
            d_v_mag = (v_grad * V_combined).sum(dim=0) * v_mag_scale / v_magnitude

            # Chain rule: grad through combined = grad * mag_scale
            q_grad = q_grad * q_mag_scale.unsqueeze(0)
            k_grad = k_grad * k_mag_scale.unsqueeze(0)
            v_grad = v_grad * v_mag_scale.unsqueeze(0)

        # LoRA bias gradients
        if q_lora_bias is not None:
            d_q_lora_bias = q_scale * q_grad.sum(dim=0)
        if k_lora_bias is not None:
            d_k_lora_bias = k_scale * k_grad.sum(dim=0)
        if v_lora_bias is not None:
            d_v_lora_bias = v_scale * v_grad.sum(dim=0)

        # Pre-transpose X_lora for LoRA gradients
        X_lora_t = X_lora.t()

        # Initialize LoRA gradients as None
        d_A_q = d_B_q = d_A_k = d_B_k = d_A_v = d_B_v = None

        # Compute LoRA gradients using X_lora (before any inplace ops on X)
        # A_q, B_q etc. are already in compute dtype (converted in forward)
        # Key optimization: compute grad @ B once, reuse for both dA and dX_lora
        # A has shape [rank, in], B has shape [out, rank]
        grad_B_q = grad_B_k = grad_B_v = None

        if A_q is not None and B_q is not None:
            grad_B_q = q_grad @ B_q  # [T, rank] — reused for dA and dX
            d_A_q = torch.empty_like(A_q.t())
            d_B_q = torch.empty_like(B_q.t())
            d_A_q.addmm_(X_lora_t, grad_B_q, alpha=q_scale, beta=0)
            d_B_q.addmm_(A_q @ X_lora_t, q_grad, alpha=q_scale, beta=0)

        if A_k is not None and B_k is not None:
            grad_B_k = k_grad @ B_k
            d_A_k = torch.empty_like(A_k.t())
            d_B_k = torch.empty_like(B_k.t())
            d_A_k.addmm_(X_lora_t, grad_B_k, alpha=k_scale, beta=0)
            d_B_k.addmm_(A_k @ X_lora_t, k_grad, alpha=k_scale, beta=0)

        if A_v is not None and B_v is not None:
            grad_B_v = v_grad @ B_v
            d_A_v = torch.empty_like(A_v.t())
            d_B_v = torch.empty_like(B_v.t())
            d_A_v.addmm_(X_lora_t, grad_B_v, alpha=v_scale, beta=0)
            d_B_v.addmm_(A_v @ X_lora_t, v_grad, alpha=v_scale, beta=0)

        # Base path input gradient (can use inplace on X since X_lora refs are done)
        out_buffer = X if ctx.inplace else None

        q_weight_t = dequantize(q_weight, q_quant)
        grad_X = torch.mm(q_grad, q_weight_t, out=out_buffer)
        del q_weight_t

        k_weight_t = dequantize(k_weight, k_quant)
        grad_X.addmm_(k_grad, k_weight_t)
        del k_weight_t

        v_weight_t = dequantize(v_weight, v_quant)
        grad_X.addmm_(v_grad, v_weight_t)
        del v_weight_t

        # LoRA path input gradient: s * grad @ B @ A (reuses grad_B_* from above)
        if has_dropout:
            grad_X_drop = torch.zeros_like(X_lora)
            if grad_B_q is not None:
                grad_X_drop.addmm_(grad_B_q, A_q, alpha=q_scale)
            if grad_B_k is not None:
                grad_X_drop.addmm_(grad_B_k, A_k, alpha=k_scale)
            if grad_B_v is not None:
                grad_X_drop.addmm_(grad_B_v, A_v, alpha=v_scale)
        else:
            grad_X_drop = None
            if grad_B_q is not None:
                grad_X.addmm_(grad_B_q, A_q, alpha=q_scale)
            if grad_B_k is not None:
                grad_X.addmm_(grad_B_k, A_k, alpha=k_scale)
            if grad_B_v is not None:
                grad_X.addmm_(grad_B_v, A_v, alpha=v_scale)

        # Transpose LoRA gradients
        if d_A_q is not None:
            d_A_q = d_A_q.t()
            d_B_q = d_B_q.t()  # type: ignore[union-attr]
        if d_A_k is not None:
            d_A_k = d_A_k.t()
            d_B_k = d_B_k.t()  # type: ignore[union-attr]
        if d_A_v is not None:
            d_A_v = d_A_v.t()
            d_B_v = d_B_v.t()  # type: ignore[union-attr]

        grad_X = grad_X.view(batch, seq_len, -1)
        if grad_X_drop is not None:
            grad_X_drop = grad_X_drop.view(batch, seq_len, -1)

        # Return gradients for all forward inputs:
        # X, X_drop,
        # q: weight, bias, quant, A, B, scale, lora_bias, magnitude
        # k: weight, bias, quant, A, B, scale, lora_bias, magnitude
        # v: weight, bias, quant, A, B, scale, lora_bias, magnitude
        # inplace
        return (
            grad_X,
            grad_X_drop,
            # Q
            None,
            None,
            None,
            d_A_q,
            d_B_q,
            None,
            d_q_lora_bias,
            d_q_mag,
            # K
            None,
            None,
            None,
            d_A_k,
            d_B_k,
            None,
            d_k_lora_bias,
            d_k_mag,
            # V
            None,
            None,
            None,
            d_A_v,
            d_B_v,
            None,
            d_v_lora_bias,
            d_v_mag,
            # inplace
            None,
        )


def _lora_only(
    X: torch.Tensor,
    A: torch.Tensor | None,
    B: torch.Tensor | None,
    s: float | None,
    lora_bias: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute only the LoRA contribution: s * X @ A^T @ B^T + s * lora_bias."""
    if A is None:
        return torch.zeros(
            X.shape[:-1] + (B.shape[0] if B is not None else 1,),
            device=X.device,
            dtype=dtype,
        )
    reshape = False
    if X.dim() == 3:
        batch, seq_len, _ = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    At, Bt = A.t().to(dtype), B.t().to(dtype)  # type: ignore[union-attr]
    out = s * X @ At @ Bt
    if lora_bias is not None:
        out = out + s * lora_bias
    return out.view(batch, seq_len, -1) if reshape else out


def apply_lora_qkv(
    self, X: torch.Tensor, inplace: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies LoRA to compute Query, Key, Value projections.

    Supports bias, dropout, and DoRA. Dropout is applied outside the autograd
    Function so PyTorch handles its backward automatically. A single shared
    dropout mask is used across Q, K, V projections for memory efficiency.
    """
    QW, Qb, QW_quant, QA, QB, QS, Qlb, Qdrop, Qmag = get_lora_parameters(self.q_proj)
    KW, Kb, KW_quant, KA, KB, KS, Klb, Kdrop, Kmag = get_lora_parameters(self.k_proj)
    VW, Vb, VW_quant, VA, VB, VS, Vlb, Vdrop, Vmag = get_lora_parameters(self.v_proj)

    # Apply dropout outside autograd.Function (shared mask for Q, K, V)
    X_drop = _apply_dropout(Qdrop, X, self.training)

    Q, K, V = LoRA_QKV.apply(
        X,
        X_drop,
        # Q
        QW,
        Qb,
        QW_quant,
        QA,
        QB,
        QS,
        Qlb,
        Qmag,
        # K
        KW,
        Kb,
        KW_quant,
        KA,
        KB,
        KS,
        Klb,
        Kmag,
        # V
        VW,
        Vb,
        VW_quant,
        VA,
        VB,
        VS,
        Vlb,
        Vmag,
        # Flags
        inplace,
    )

    return Q, K, V


class LoRA_O(torch.autograd.Function):
    """Optimized LoRA implementation for output projection.

    Supports bias, dropout, and DoRA.
    """

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        X: torch.Tensor,
        X_drop: torch.Tensor | None,
        W: torch.Tensor,
        b: torch.Tensor | None,
        W_quant: QuantState | None,
        A: torch.Tensor | None,
        B: torch.Tensor | None,
        s: float,
        lora_bias: torch.Tensor | None,
        magnitude: torch.Tensor | None,
    ) -> torch.Tensor:
        has_dropout = X_drop is not None
        has_dora = magnitude is not None
        dtype = X.dtype

        if has_dora:
            X_lora = X_drop if has_dropout else X
            base_out = matmul_lora(X, W, None, W_quant, None, None, None)
            lora_out = _lora_only(X_lora, A, B, s, lora_bias, dtype)
            mag_scale = _compute_dora_scale(W, W_quant, A, B, s, magnitude, dtype)
            combined = base_out + lora_out
            XW = mag_scale.unsqueeze(0) * combined
            if b is not None:
                XW = XW + b

            ctx.save_for_backward(
                A.to(dtype) if A is not None else A,
                B.to(dtype) if B is not None else B,
                X,
                X_drop if has_dropout else X,
                magnitude,
                mag_scale,
                combined,
                lora_bias,
            )
        else:
            XW = matmul_lora(
                X,
                W,
                b,
                W_quant,
                A,
                B,
                s,
                X_drop=X_drop,
                lora_bias=lora_bias,
            )
            # Pre-convert LoRA matrices to compute dtype for backward
            dtype = X.dtype
            ctx.save_for_backward(
                A.to(dtype) if A is not None else A,
                B.to(dtype) if B is not None else B,
                X,
                X_drop if has_dropout else X,
                lora_bias,
            )

        ctx.custom_saved_tensors = (W, W_quant, s)
        ctx.has_dropout = has_dropout
        ctx.has_dora = has_dora

        return XW

    @staticmethod
    @torch_amp_custom_bwd
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        dY: torch.Tensor,
    ):
        W, W_quant, s = ctx.custom_saved_tensors
        has_dropout = ctx.has_dropout
        has_dora = ctx.has_dora

        if has_dora:
            A, B, X, X_lora, magnitude, mag_scale, combined, lora_bias = (
                ctx.saved_tensors
            )
        else:
            A, B, X, X_lora, lora_bias = ctx.saved_tensors
            magnitude = mag_scale = combined = None

        batch, seq_len, hd = X.shape
        dY = dY.reshape(-1, dY.shape[-1])
        X = X.reshape(-1, X.shape[-1])
        X_lora = X_lora.reshape(-1, X_lora.shape[-1])

        d_mag = d_lora_bias = None

        if has_dora:
            combined = combined.view(-1, combined.shape[-1])
            d_mag = (dY * combined).sum(dim=0) * mag_scale / magnitude
            dY = dY * mag_scale.unsqueeze(0)

        # LoRA bias gradient
        if lora_bias is not None:
            d_lora_bias = s * dY.sum(dim=0)

        # LoRA parameter gradients (A, B already in compute dtype from forward)
        # Compute dY @ B once, reuse for both dA and dX_lora
        d_A = d_B = None
        grad_B = None
        if A is not None:
            grad_B = dY @ B  # [T, rank] — reused below
            X_lora_t = X_lora.t()
            d_A = torch.empty_like(A.t())
            d_B = torch.empty_like(B.t())
            d_A.addmm_(X_lora_t, grad_B, alpha=s, beta=0)
            d_B.addmm_(A @ X_lora_t, dY, alpha=s, beta=0)

        # Base path input gradient
        W_deq = dequantize(W.t(), W_quant)
        dX = dY @ W_deq.t()
        del W_deq

        if has_dropout:
            dX_drop = None
            if grad_B is not None:
                dX_drop = (grad_B @ A * s).view(batch, seq_len, hd)
        else:
            dX_drop = None
            if grad_B is not None:
                dX.addmm_(grad_B, A, alpha=s)

        # X, X_drop, W, b, W_quant, A, B, s, lora_bias, magnitude
        return (
            dX.view(batch, seq_len, hd),
            dX_drop,
            None,
            None,
            None,
            d_A.t() if d_A is not None else None,
            d_B.t() if d_B is not None else None,
            None,
            d_lora_bias,
            d_mag,
        )


def apply_lora_o(self, X: torch.Tensor) -> torch.Tensor:
    """
    Applies LoRA to output projection layer.

    Supports bias, dropout, and DoRA.
    """
    OW, Ob, OW_quant, OA, OB, OS, Olb, Odrop, Omag = get_lora_parameters(self.o_proj)
    X_drop = _apply_dropout(Odrop, X, self.training)
    output = LoRA_O.apply(X, X_drop, OW, Ob, OW_quant, OA, OB, OS, Olb, Omag)

    return output


# ============================================================
# Embedding LoRA kernel
# ============================================================


def get_embedding_lora_parameters(
    embed: nn.Module,
) -> tuple[
    torch.Tensor,  # W (base embedding weight)
    torch.Tensor | None,  # A (lora_embedding_A)
    torch.Tensor | None,  # B (lora_embedding_B)
    float | None,  # scaling
    nn.Module | None,  # dropout
    torch.Tensor | None,  # magnitude (DoRA)
    nn.Module,  # base_layer
]:
    """Extract LoRA parameters from a PEFT Embedding module."""
    base_layer = embed.base_layer if hasattr(embed, "base_layer") else embed
    W = base_layer.weight

    if not hasattr(embed, "disable_adapters") or embed.disable_adapters or embed.merged:
        return W, None, None, None, None, None, base_layer

    active_adapter = (
        embed.active_adapters[0]
        if hasattr(embed, "active_adapters")
        else embed.active_adapter
    )

    A = embed.lora_embedding_A[active_adapter]  # nn.Parameter [rank, vocab]
    B = embed.lora_embedding_B[active_adapter]  # nn.Parameter [hidden_dim, rank]
    s = embed.scaling[active_adapter]

    # FSDP2 DTensor unshard (mirrors linear path logic)
    if isinstance(A, DTensor):
        A = A.full_tensor()
    if isinstance(B, DTensor):
        B = B.full_tensor()

    dropout = None
    if hasattr(embed, "lora_dropout") and active_adapter in embed.lora_dropout:
        dropout = embed.lora_dropout[active_adapter]

    magnitude = None
    if (
        hasattr(embed, "lora_magnitude_vector")
        and embed.lora_magnitude_vector
        and active_adapter in embed.lora_magnitude_vector
    ):
        mag_layer = embed.lora_magnitude_vector[active_adapter]
        magnitude = mag_layer.weight
        if isinstance(magnitude, DTensor):
            magnitude = magnitude.full_tensor()

    return W, A, B, s, dropout, magnitude, base_layer


class LoRA_Embedding(torch.autograd.Function):
    """Fused LoRA embedding: F.embedding(x, W) + s * F.embedding(x, A^T) @ B^T.

    Supports dropout and DoRA.
    """

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        W: torch.Tensor,
        A: torch.Tensor | None,
        B: torch.Tensor | None,
        s: float | None,
        magnitude: torch.Tensor | None,
        padding_idx: int | None,
        # base_layer fields for F.embedding
        max_norm: float | None,
        norm_type: float,
        scale_grad_by_freq: bool,
        sparse: bool,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        has_dora = magnitude is not None
        dtype = W.dtype

        # Base embedding lookup
        result = F.embedding(
            x,
            W,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )

        if A is not None:
            # LoRA: F.embedding(x, A^T) @ B^T * s
            A_T = A.t()  # type: ignore[union-attr]  # [vocab, rank]
            B_T = B.t()  # type: ignore[union-attr]  # [rank, hidden_dim]
            after_A = F.embedding(
                x,
                A_T,
                padding_idx=padding_idx,
                max_norm=max_norm,
                norm_type=norm_type,
                scale_grad_by_freq=scale_grad_by_freq,
                sparse=sparse,
            )  # [batch, seq, rank]

            lora_result = after_A @ B_T  # [batch, seq, hidden_dim]

            if has_dora:
                mag_scale = _compute_dora_scale(W.t(), None, A, B, s, magnitude, dtype)  # type: ignore[arg-type]
                # DoRA: mag_scale * (base + s * lora)
                # base embedding has no bias
                pre_scaled = result + s * lora_result  # unscaled combined
                result = mag_scale.unsqueeze(0) * pre_scaled
                ctx.save_for_backward(
                    x,
                    A.to(dtype),
                    B.to(dtype),  # type: ignore[union-attr]
                    after_A,
                    magnitude,
                    mag_scale,
                    pre_scaled,  # save unscaled for correct d_mag
                )
            else:
                result = result + s * lora_result
                ctx.save_for_backward(x, A.to(dtype), B.to(dtype), after_A)  # type: ignore[union-attr]
        else:
            ctx.save_for_backward(
                x,
            )

        ctx.s = s
        ctx.has_dora = has_dora
        ctx.has_lora = A is not None
        ctx.padding_idx = padding_idx
        ctx.scale_grad_by_freq = scale_grad_by_freq

        return result

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, grad_output):
        s = ctx.s
        has_dora = ctx.has_dora
        has_lora = ctx.has_lora

        d_A = d_B = d_mag = None

        if not has_lora:
            (x,) = ctx.saved_tensors
        elif has_dora:
            x, A, B, after_A, magnitude, mag_scale, combined = ctx.saved_tensors
            # DoRA magnitude gradient
            combined_flat = combined.view(-1, combined.shape[-1])
            grad_flat = grad_output.view(-1, grad_output.shape[-1])
            d_mag = (grad_flat * combined_flat).sum(dim=0) * mag_scale / magnitude
            # Chain rule through mag_scale
            grad_output = grad_output * mag_scale.unsqueeze(0).unsqueeze(0)
        else:
            x, A, B, after_A = ctx.saved_tensors

        if has_lora:
            # Use float32 for gradient computation (LoRA params are fp32)
            compute_dtype = torch.float32

            after_A_flat = after_A.view(-1, after_A.shape[-1]).to(compute_dtype)
            grad_flat = grad_output.view(-1, grad_output.shape[-1]).to(compute_dtype)
            B_f = B.to(compute_dtype)

            # B is [hidden_dim, rank], B_T = B.t() = [rank, hidden_dim]
            # lora_result = after_A @ B_T → d/d(B_T) = s * after_A^T @ grad
            B_T = B_f.t()  # [rank, hidden_dim]
            d_B_T = torch.empty_like(B_T)
            d_B_T.addmm_(after_A_flat.t(), grad_flat, alpha=s, beta=0)
            d_B = d_B_T.t()  # [hidden_dim, rank]

            # d_A: gradient flows through F.embedding lookup
            # d_after_A = s * grad @ B = [T, hidden] @ [hidden, rank] = [T, rank]
            d_after_A = s * grad_flat @ B_f

            # F.embedding backward: scatter d_after_A into A^T gradient
            x_flat = x.view(-1)

            # Zero out padding_idx contributions (matches F.embedding behavior)
            if ctx.padding_idx is not None:
                pad_mask = x_flat != ctx.padding_idx
                d_after_A = d_after_A * pad_mask.unsqueeze(1).to(d_after_A.dtype)

            # scale_grad_by_freq: divide each contribution by token frequency
            if ctx.scale_grad_by_freq:
                counts = torch.bincount(x_flat, minlength=A.shape[1]).clamp(min=1)
                freq_scale = 1.0 / counts[x_flat].unsqueeze(1).to(d_after_A.dtype)
                d_after_A = d_after_A * freq_scale

            A_f = A.to(compute_dtype)
            d_A_T = torch.zeros_like(A_f.t())  # [vocab, rank]
            d_A_T.index_add_(0, x_flat, d_after_A)
            d_A = d_A_T.t()  # [rank, vocab]

        # x, W, A, B, s, magnitude, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
        return (
            None,  # x
            None,  # W (base embedding weight grad handled by PyTorch)
            d_A,  # A
            d_B,  # B
            None,  # s
            d_mag,  # magnitude
            None,
            None,
            None,
            None,
            None,  # padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
        )


def apply_lora_embedding(self, x: torch.Tensor) -> torch.Tensor:
    """Applies LoRA to embedding layer."""
    W, A, B, s, dropout, magnitude, base_layer = get_embedding_lora_parameters(self)

    # Capture base output dtype (bf16 for bf16 models) to cast back at end
    output_dtype = W.dtype

    # Note: PEFT's Embedding forward does not apply dropout for embeddings
    # (integer indices can't be dropped; PEFT silently ignores lora_dropout here)
    result = LoRA_Embedding.apply(
        x,
        W,
        A,
        B,
        s,
        magnitude,
        base_layer.padding_idx,
        base_layer.max_norm,
        base_layer.norm_type,
        base_layer.scale_grad_by_freq,
        base_layer.sparse,
    )

    # Cast to model dtype (LoRA ops may upcast to float32)
    return result.to(output_dtype)
