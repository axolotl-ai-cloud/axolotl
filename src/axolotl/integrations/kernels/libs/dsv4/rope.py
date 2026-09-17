"""Fused interleaved partial-RoPE kernel for DeepSeek-V4 (fwd + bwd).

Reference: ``transformers.models.deepseek_v4.modeling_deepseek_v4.apply_rotary_pos_emb``.

V4 uses *interleaved* RoPE (GPT-J style: consecutive channel pairs) on the trailing
``rope_dim`` channels of each head, leaving the leading ``nope`` channels untouched.
``cos``/``sin`` arrive half-width (one entry per pair, shape ``[B, S, rope_dim//2]``)
and are expanded by ``repeat_interleave(2)`` in the reference. Per pair ``i``:

    out[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
    out[2i+1] = x[2i+1]*cos[i] + x[2i]*sin[i]

The rotation is orthogonal, so the backward is the same map with ``sin -> -sin``.
The model also calls the reference with a pre-negated ``sin`` (output inverse-rope);
that composes naturally — this Function just rotates by whatever ``(cos, sin)`` it gets
and negates ``sin`` for its own backward.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    S,
    H,
    cs_b,  # cos/sin strides: may be non-contiguous (the rotary's transpose view)
    cs_s,
    cs_i,
    D: tl.constexpr,
    ROPE: tl.constexpr,
    Rh: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
):
    row = tl.program_id(0)  # one program per (b, h, s) row of length D
    s = row % S
    b = (row // S) // H
    nope = D - ROPE
    cs_off = b * cs_b + s * cs_s

    x_row = x_ptr + row * D
    out_row = out_ptr + row * D

    # passthrough leading nope channels
    off = tl.arange(0, BLOCK_NOPE)
    m = off < nope
    tl.store(out_row + off, tl.load(x_row + off, mask=m, other=0.0), mask=m)

    i = tl.arange(0, Rh)
    even = nope + 2 * i
    odd = even + 1
    e = tl.load(x_row + even).to(tl.float32)
    o = tl.load(x_row + odd).to(tl.float32)
    c = tl.load(cos_ptr + cs_off + i * cs_i).to(tl.float32)
    sn = tl.load(sin_ptr + cs_off + i * cs_i).to(tl.float32)
    tl.store(out_row + even, e * c - o * sn)
    tl.store(out_row + odd, o * c + e * sn)


def _launch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, S, D] contiguous; cos/sin: [B, S, rope_dim//2] (any strides). Returns
    [B, H, S, D]. cos/sin come from the rotary as a ``transpose(1,2)`` view, so they are
    typically *not* contiguous — we pass their strides instead of assuming a layout."""
    B, H, S, D = x.shape
    Rh = cos.shape[-1]
    ROPE = 2 * Rh
    if sin.stride() != cos.stride():
        sin = sin.contiguous()
        cos = cos.contiguous()
    out = torch.empty_like(x)
    M = B * H * S
    _rope_kernel[(M,)](
        x,
        cos,
        sin,
        out,
        S,
        H,
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        D=D,
        ROPE=ROPE,
        Rh=Rh,
        BLOCK_NOPE=triton.next_power_of_2(D - ROPE),
        num_warps=4,
    )
    return out


class _RoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        x = x.contiguous()
        ctx.save_for_backward(cos, sin)
        return _launch(x, cos, sin)

    @staticmethod
    def backward(ctx, grad_out):
        cos, sin = ctx.saved_tensors
        grad_x = _launch(grad_out.contiguous(), cos, -sin)
        return grad_x, None, None


def apply_rotary_pos_emb_triton(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    """Drop-in for ``apply_rotary_pos_emb``. ``x``: [B, H, S, D]; ``cos``/``sin``:
    [B, S, rope_dim//2]. ``unsqueeze_dim`` is accepted for signature parity (the
    reference broadcasts cos/sin over the head axis; this kernel indexes it directly)."""
    # rotary cos/sin are typically fp32; match them to x's compute dtype.
    if cos.dtype != x.dtype:
        cos = cos.to(x.dtype)
    if sin.dtype != x.dtype:
        sin = sin.to(x.dtype)
    return _RoPE.apply(x, cos, sin)
