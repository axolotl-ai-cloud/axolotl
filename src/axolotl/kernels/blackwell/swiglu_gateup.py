"""Stride-aware fused SwiGLU forward/backward Triton kernels.

Unlike the flat-indexed kernels in `axolotl.kernels.swiglu`, these take an
explicit row stride so gate/up may be non-contiguous column slices of one
shared ``[M, 2*INTER]`` buffer -- the adjacent-buffer layout the B200 factored
LoRA MLP kernel relies on to make its backward concat free.
"""

import torch
import triton
import triton.language as tl

# Small 2D tiles on purpose: large tiles (e.g. 32x1024) put far too many
# elements in one program (register pressure / poor occupancy) and measured
# ~16x slower than the flat baseline kernels.
_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w)
    for bm, bn in [
        (4, 256),
        (8, 256),
        (4, 512),
        (8, 512),
        (16, 256),
        (2, 1024),
        (4, 1024),
    ]
    for w in [4, 8]
]


@triton.autotune(configs=_CONFIGS, key=["M", "N"])
@triton.jit
def _swiglu_fwd_view_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    M,
    N,
    stride_gm,
    stride_um,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (om[:, None] < M) & (on[None, :] < N)

    gate = tl.load(
        gate_ptr + om[:, None] * stride_gm + on[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    up = tl.load(up_ptr + om[:, None] * stride_um + on[None, :], mask=mask, other=0.0)
    f = gate * tl.sigmoid(gate)
    f = f.to(up.dtype)
    result = f * up
    tl.store(out_ptr + om[:, None] * stride_om + on[None, :], result, mask=mask)


# This kernel mutates its own inputs in place (grad_out/gate/up all become
# outputs). Autotuning would rerun trials against progressively-mutated inputs
# (-> NaN) without `restore_value`, which snapshots and restores the named
# arguments around each trial.
@triton.autotune(
    configs=_CONFIGS,
    key=["M", "N"],
    restore_value=["grad_out_ptr", "gate_ptr", "up_ptr"],
)
@triton.jit
def _swiglu_bwd_view_kernel(
    grad_out_ptr,
    gate_ptr,
    up_ptr,
    M,
    N,
    stride_dm,
    stride_gm,
    stride_um,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (om[:, None] < M) & (on[None, :] < N)

    grad_out = tl.load(
        grad_out_ptr + om[:, None] * stride_dm + on[None, :], mask=mask, other=0.0
    )
    gate = tl.load(
        gate_ptr + om[:, None] * stride_gm + on[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    up = tl.load(up_ptr + om[:, None] * stride_um + on[None, :], mask=mask, other=0.0)

    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = sigmoid_gate * gate
    silu_gate = silu_gate.to(grad_out.dtype)
    h = silu_gate * up
    grad_up = grad_out * silu_gate
    temp = grad_out * up
    grad_gate = temp.to(tl.float32) * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    grad_gate = grad_gate.to(grad_out.dtype)

    tl.store(grad_out_ptr + om[:, None] * stride_dm + on[None, :], h, mask=mask)
    tl.store(gate_ptr + om[:, None] * stride_gm + on[None, :], grad_gate, mask=mask)
    tl.store(up_ptr + om[:, None] * stride_um + on[None, :], grad_up, mask=mask)


def _check_view_contract(*tensors: torch.Tensor) -> None:
    # The kernels index ptr + row*stride_m + col: rows may be strided (column
    # slices of a wider buffer) but the last dim must be unit-stride, and all
    # operands must share a shape -- anything else silently corrupts.
    shape = tensors[0].shape
    for t in tensors:
        if t.shape != shape or t.stride(1) != 1:
            raise ValueError(
                "swiglu_gateup kernels require same-shape 2D tensors with "
                f"unit last-dim stride; got shape={tuple(t.shape)}, "
                f"stride={t.stride()}"
            )


def swiglu_forward_view(
    gate: torch.Tensor, up: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    _check_view_contract(gate, up, out)
    M, N = gate.shape
    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    _swiglu_fwd_view_kernel[grid](
        gate, up, out, M, N, gate.stride(0), up.stride(0), out.stride(0)
    )
    return out


def swiglu_backward_view(
    grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """In-place: returns (h, grad_gate, grad_up) as the mutated
    (grad_output, gate, up)."""
    _check_view_contract(grad_output, gate, up)
    M, N = gate.shape
    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    _swiglu_bwd_view_kernel[grid](
        grad_output, gate, up, M, N, grad_output.stride(0), gate.stride(0), up.stride(0)
    )
    return grad_output, gate, up
