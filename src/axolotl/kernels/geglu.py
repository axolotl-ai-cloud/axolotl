"""Module for definition of GEGLU Triton kernels.

See "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202).

Credit to `unsloth` (https://unsloth.ai/) for inspiration for this implementation.
"""

import torch
import triton
import triton.language as tl

from axolotl.kernels.op_registry import register_kernel_op


@triton.jit
def _gelu_partial(gate, APPROXIMATE: tl.constexpr):
    """GELU(gate) / gate. The tanh form uses 0.5 * (1 + tanh(z)) == sigmoid(2 * z)."""
    if APPROXIMATE:
        inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
        partial = tl.sigmoid(2.0 * inner)
    else:
        partial = 0.5 * (tl.math.erf(tl.math.rsqrt(2.0) * gate) + 1.0)
    return partial


@triton.jit
def _geglu_fwd_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    APPROXIMATE: tl.constexpr,
):
    """GEGLU forward kernel.

    Args:
        gate_ptr: Pointer to gate tensor [*, hidden_dim].
        up_ptr: Pointer to up-projection tensor [*, hidden_dim].
        out_ptr: Pointer to output tensor [*, hidden_dim].
        n_elements: Total number of elements in the input tensors.
        BLOCK_SIZE: Size of thread blocks for parallel computation.
        APPROXIMATE: Use the tanh approximation of GELU instead of the exact erf form.
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0)

    # Compute activation in fp32 then convert back
    gelu_gate = _gelu_partial(gate, APPROXIMATE) * gate
    gelu_gate = gelu_gate.to(up.dtype)
    result = gelu_gate * up

    tl.store(out_ptr + offsets, result, mask=mask)


@register_kernel_op("geglu_fwd")
def geglu_forward(
    gate: torch.Tensor, up: torch.Tensor, approximate: bool = False
) -> torch.Tensor:
    """GEGLU forward pass.

    Args:
        gate: Input gate tensor of shape [batch, seq_len, hidden_dim].
        up: Up-projection tensor of shape [batch, seq_len, hidden_dim].
        approximate: Use the tanh approximation of GELU (`gelu_pytorch_tanh`).

    Returns:
        torch.Tensor: Output tensor of shape [batch, seq_len, hidden_dim].
    """
    batch, seq_len, hidden_dim = gate.shape
    n_elements = gate.numel()
    out = torch.empty((batch, seq_len, hidden_dim), dtype=gate.dtype, device="cuda")

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    _geglu_fwd_kernel[grid](
        gate_ptr=gate,
        up_ptr=up,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=1024,
        APPROXIMATE=approximate,
    )
    return out


@triton.jit
def _geglu_bwd_kernel(
    grad_out_ptr,
    gate_ptr,
    up_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    APPROXIMATE: tl.constexpr,
):
    """GEGLU backward kernel. Stores gradient results in-place.

    Args:
        grad_out_ptr: Pointer to gradient output tensor [*, hidden_dim].
        gate_ptr: Pointer to gate tensor [*, hidden_dim].
        up_ptr: Pointer to up-projection tensor [*, hidden_dim].
        n_elements: Total number of elements in the input tensors.
        BLOCK_SIZE: Size of thread blocks for parallel computation.

    Note:
        After kernel execution, tensors are modified in-place:
        - `grad_out_ptr` contains GEGLU activation output (`h`)
        - `gate_ptr` contains gradient w.r.t gate (`grad_gate`)
        - `up_ptr` contains gradient w.r.t up (`grad_up`)
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0)

    # Forward pass
    gelu_partial = _gelu_partial(gate, APPROXIMATE)
    gelu_gate = gelu_partial * gate
    gelu_gate = gelu_gate.to(grad_out.dtype)

    # Forward output
    h = gelu_gate * up

    # Compute gradients
    grad_up = grad_out * gelu_gate

    # Compute gate gradient using GELU derivative
    temp = grad_out * up
    if APPROXIMATE:
        dgelu_dgate = gelu_partial + gate * 2.0 * gelu_partial * (
            1.0 - gelu_partial
        ) * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * gate * gate)
    else:
        t = 0.3989422804014327  # 1/sqrt(2*pi)
        dgelu_dgate = gelu_partial + t * gate * tl.exp(-0.5 * gate * gate)
    grad_gate = temp.to(tl.float32) * dgelu_dgate
    grad_gate = grad_gate.to(grad_out.dtype)

    # Store results
    tl.store(grad_out_ptr + offsets, h, mask=mask)
    tl.store(gate_ptr + offsets, grad_gate, mask=mask)
    tl.store(up_ptr + offsets, grad_up, mask=mask)


@geglu_forward.register_fake
def _(gate, up, approximate=False):
    return torch.empty_like(gate)


@register_kernel_op("geglu_bwd", mutates_args=("grad_output", "gate", "up"))
def _geglu_bwd_op(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    approximate: bool = False,
) -> None:
    n_elements = grad_output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    _geglu_bwd_kernel[grid](
        grad_out_ptr=grad_output,
        gate_ptr=gate,
        up_ptr=up,
        n_elements=n_elements,
        BLOCK_SIZE=1024,
        APPROXIMATE=approximate,
    )


@_geglu_bwd_op.register_fake
def _(grad_output, gate, up, approximate=False):
    return None


def geglu_backward(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    approximate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GEGLU backward pass using in-place operations.

    Args:
        grad_output: Gradient of loss with respect to output, shape `[batch, seq_len, hidden_dim]`.
        gate: Gate tensor from forward pass, shape `[batch, seq_len, hidden_dim]`.
        up: Up-projection tensor from forward pass, shape `[batch, seq_len, hidden_dim]`.
        approximate: Use the tanh approximation of GELU (`gelu_pytorch_tanh`).

    Returns:
        Tuple containing:
            - GEGLU activation output (`h`)
            - Gradient with respect to gate (`grad_gate`)
            - Gradient with respect to up (`grad_up`)

    Note:
        This function modifies its input tensors in-place to store results.
    """
    # op mutates in place; return the aliases here — op outputs must not alias inputs
    _geglu_bwd_op(grad_output, gate, up, approximate)
    return grad_output, gate, up
