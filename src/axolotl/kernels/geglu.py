"""
Module for definition of GEGLU Triton kernels.

See "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202).

Credit to `unsloth` (https://unsloth.ai/) for inspiration for this implementation.
"""

import torch
import triton
import triton.language as tl

SQRT_2_PI: tl.constexpr = 0.7978845608028654  # sqrt(2/π)


@triton.jit
def _geglu_fwd_kernel(
    gate,
    up,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate_row = tl.load(gate + offsets, mask=mask, other=0).to(tl.float32)
    up_row = tl.load(up + offsets, mask=mask, other=0)

    # Match unsloth's implementation exactly
    f_row = 0.5 * gate_row * (tl.math.erf(tl.math.rsqrt(2.0) * gate_row) + 1.0)
    f_row = f_row.to(up_row.dtype)
    result = f_row * up_row

    tl.store(out + offsets, result, mask=mask)


def geglu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """GEGLU forward pass."""
    batch, seq_len, hidden_dim = gate.shape
    n_elements = gate.numel()
    out = torch.empty((batch, seq_len, hidden_dim), dtype=gate.dtype, device="cuda")

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    _geglu_fwd_kernel[grid](
        gate=gate,
        up=up,
        out=out,
        n_elements=n_elements,
        BLOCK_SIZE=1024,
    )
    return out


@triton.jit
def _geglu_bwd_kernel(
    doutput,
    gate,
    up,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    doutput_row = tl.load(doutput + offsets, mask=mask, other=0)
    e_row = tl.load(gate + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(up + offsets, mask=mask, other=0)

    # Forward pass
    f_partial_row = 0.5 * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
    f_row = f_partial_row * e_row
    f_row = f_row.to(doutput_row.dtype)

    # Forward output
    h_row = f_row * g_row

    # Gradients
    df_row = doutput_row * f_row
    dg_row = doutput_row * g_row

    # df/de using same f_partial_row from forward
    t = 0.3989422804014327  # 1/sqrt(2*pi)
    df_de = f_partial_row + t * e_row * tl.exp(-0.5 * e_row * e_row)
    de_row = dg_row.to(tl.float32) * df_de
    de_row = de_row.to(doutput_row.dtype)

    # Store exactly as unsloth does
    tl.store(doutput + offsets, h_row, mask=mask)  # h = f * g
    tl.store(gate + offsets, df_row, mask=mask)  # df = DW * f
    tl.store(up + offsets, de_row, mask=mask)  # de


def geglu_backward(grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor):
    """GEGLU backward pass using in-place operations."""
    n_elements = grad_output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    _geglu_bwd_kernel[grid](
        doutput=grad_output,
        gate=gate,
        up=up,
        n_elements=n_elements,
        BLOCK_SIZE=1024,
    )

    # After kernel execution, tensors contain:
    # grad_output: h (forward output)
    # gate: df (grad wrt gate)
    # up: de (grad wrt up)
    return grad_output, gate, up
