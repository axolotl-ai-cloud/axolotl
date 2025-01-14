"""
Module for definition of GEGLU Triton kernels.

See "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202).
"""
# pylint: disable=invalid-name,duplicate-code

import torch
import triton
import triton.language as tl

from axolotl.kernels.utils import SQRT_2_PI


@triton.jit
def _geglu_fwd_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """GEGLU forward kernel."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0).to(tl.float32)

    x = gate * SQRT_2_PI
    cdf = 0.5 * (1.0 + tl.erf(x))
    result = gate * cdf * up

    tl.store(out_ptr + offsets, result, mask=mask)


# pylint: disable=unnecessary-lambda-assignment
def geglu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Direct wrapper for GEGLU forward kernel."""
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
    )

    return out


@triton.jit
def _geglu_bwd_kernel(
    grad_out_ptr,
    gate_ptr,
    up_ptr,
    grad_gate_ptr,
    grad_up_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """GEGLU backward kernel."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0).to(tl.float32)

    x = gate * SQRT_2_PI
    cdf = 0.5 * (1.0 + tl.erf(x))
    pdf = SQRT_2_PI * tl.exp(-0.5 * x * x)

    grad_gate = grad_out * (cdf + gate * pdf) * up
    grad_up = grad_out * gate * cdf

    tl.store(grad_gate_ptr + offsets, grad_gate, mask=mask)
    tl.store(grad_up_ptr + offsets, grad_up, mask=mask)


# pylint: disable=unnecessary-lambda-assignment
def geglu_backward(grad_out: torch.Tensor, gate: torch.Tensor, up: torch.Tensor):
    """Direct wrapper for GEGLU backward kernel."""
    n_elements = grad_out.numel()
    grad_gate = torch.empty_like(gate)
    grad_up = torch.empty_like(up)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    _geglu_bwd_kernel[grid](
        grad_out_ptr=grad_out,
        gate_ptr=gate,
        up_ptr=up,
        grad_gate_ptr=grad_gate,
        grad_up_ptr=grad_up,
        n_elements=n_elements,
        BLOCK_SIZE=1024,
    )

    return grad_gate, grad_up
