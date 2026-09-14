"""B200 (sm_100)-tuned factored LoRA MLP kernel.

Computes the standard factored LoRA form per branch (no merged/augmented
weight buffer, one weight copy):

    e = X @ gate_W.T + gate_s * (X @ gate_A.T) @ gate_B.T
    g = X @ up_W.T   + up_s   * (X @ up_A.T)   @ up_B.T
    h = swiglu(e, g)
    out = h @ down_W.T + down_s * (h @ down_A.T) @ down_B.T

cuBLAS GEMMs throughout plus the stride-aware fused SwiGLU kernels in
`swiglu_gateup`. Structural wins over the reference factored implementation:
forward intermediates (XAg/XAu/hAd) are saved instead of re-derived in
backward, the skinny adapter GEMMs are deduplicated and concatenated
(X @ [gate_A.T | up_A.T] as one GEMM, and the dual on the dX accumulation),
dX is built with 1 write + 2 read-modify-write passes instead of 1 + 3, the
big base GEMMs are dispatched before the small adapter ops so the CPU can
queue the small ops while the GPU is busy, and scalar multiplies are fused
into their adjacent GEMM's addmm alpha.

Certified on B200 (compute capability (10, 0)) ONLY, bf16 only, no dropout,
no bias, no quantized base -- callers must gate on
`support.lora_mlp_b200_config_eligible` / `lora_mlp_b200_module_supported`
before dispatching here. Not traceable by torch.compile: the forward writes
into non-contiguous buffer slices via `out=`, which Dynamo cannot trace, so
the public wrapper is marked `torch.compiler.disable`.
"""

import torch

from axolotl.utils.logging import get_logger

from . import swiglu_gateup as sgv

LOG = get_logger(__name__)


def _scaled_matmul(A, B, scale):
    """(A @ B) * scale as one GEMM via addmm's alpha (beta=0).

    With beta=0 the input matrix is never read (standard BLAS semantics), so
    an uninitialized dummy is safe, and torch.empty() launches no kernel --
    this saves one elementwise-multiply launch per call vs
    `scale * torch.matmul(A, B)`.
    """
    out_shape = (A.shape[0], B.shape[1])
    dummy = torch.empty(out_shape, device=A.device, dtype=A.dtype)
    return torch.addmm(dummy, A, B, alpha=scale, beta=0.0)


class LoRA_MLP_B200_Factored(torch.autograd.Function):
    """Factored LoRA SwiGLU MLP, B200-tuned. See module docstring."""

    @staticmethod
    def forward(
        ctx,
        X,
        gate_W,
        gate_A,
        gate_B,
        gate_s,
        up_W,
        up_A,
        up_B,
        up_s,
        down_W,
        down_A,
        down_B,
        down_s,
        inplace=True,
    ):
        reshape = X.dim() == 3
        if reshape:
            batch, seq_len, hd = X.shape
            X = X.reshape(-1, hd)
        T, HIDDEN = X.shape
        INTER = gate_W.shape[0]
        r = gate_A.shape[0]

        # Dispatch the two big base GEMMs first: PyTorch's async dispatch needs
        # a backlog of queued GPU work for the CPU to race ahead; starting with
        # the small adapter ops leaves the GPU idle waiting on CPU dispatch.
        # e/g's base terms depend only on X and the base weights, so ordering
        # them before the adapter concat is safe.
        buf_eg = torch.empty(T, 2 * INTER, device=X.device, dtype=X.dtype)
        e = buf_eg[:, :INTER]
        g = buf_eg[:, INTER:]
        torch.matmul(X, gate_W.t(), out=e)
        torch.matmul(X, up_W.t(), out=g)

        # One skinny GEMM for both adapters: X @ [gate_A.T | up_A.T].
        A_gu = torch.cat([gate_A, up_A], dim=0)  # [2r, HIDDEN]
        XA_gu = torch.matmul(X, A_gu.t())  # [T, 2r]
        XAg = XA_gu[:, :r]
        XAu = XA_gu[:, r:]

        e.addmm_(XAg, gate_B.t(), alpha=gate_s)
        g.addmm_(XAu, up_B.t(), alpha=up_s)

        h = torch.empty(T, INTER, device=X.device, dtype=X.dtype)
        sgv.swiglu_forward_view(e, g, h)

        # hAd is saved for backward -- recomputing it there would re-read all
        # of h for a [T, r] result.
        hAd = torch.matmul(h, down_A.t())  # [T, r]
        out = torch.matmul(h, down_W.t())  # [T, HIDDEN]
        out.addmm_(hAd, down_B.t(), alpha=down_s)

        if reshape:
            out = out.view(batch, seq_len, -1)

        ctx.save_for_backward(
            X, buf_eg, XAg, XAu, hAd, A_gu, gate_A, gate_B, up_A, up_B, down_A, down_B
        )
        ctx.mats = (gate_W, up_W, down_W, gate_s, up_s, down_s, r, HIDDEN, INTER)
        ctx.reshape = (batch, seq_len, hd) if reshape else None
        ctx.inplace = inplace
        return out

    @staticmethod
    def backward(ctx, dY):
        # backward consumes its saved buffers in place (swiglu_backward_view
        # mutates buf_eg; inplace=True writes dX into X's storage) and the
        # Triton writes bypass autograd's version counters, so a second
        # backward over the same graph would silently compute garbage.
        if getattr(ctx, "_consumed", False):
            raise RuntimeError(
                "LoRA_MLP_B200_Factored does not support backward through the "
                "same graph twice (saved buffers are consumed in place)"
            )
        ctx._consumed = True

        (X, buf_eg, XAg, XAu, hAd, A_gu, gate_A, gate_B, up_A, up_B, down_A, down_B) = (
            ctx.saved_tensors
        )
        gate_W, up_W, down_W, gate_s, up_s, down_s, r, HIDDEN, INTER = ctx.mats

        if ctx.reshape is not None:
            batch, seq_len, hd = ctx.reshape
            dY = dY.reshape(-1, hd)
        T = dY.shape[0]

        e = buf_eg[:, :INTER]
        g = buf_eg[:, INTER:]

        # Down projection: dhAd computed once, reused for d_down_A and dh.
        dhAd = torch.matmul(dY, down_B)  # [T, r]
        d_down_B = _scaled_matmul(dY.t(), hAd, down_s)  # [HIDDEN, r]

        dh = torch.empty(T, INTER, device=X.device, dtype=X.dtype)
        torch.matmul(dY, down_W, out=dh)
        dh.addmm_(dhAd, down_A, alpha=down_s)

        # Recomputes h (cheaper than saving it), mutates e->grad_gate,
        # g->grad_up in place so buf_eg becomes [grad_gate|grad_up] adjacently.
        h, grad_gate, grad_up = sgv.swiglu_backward_view(dh, e, g)

        d_down_A = _scaled_matmul(dhAd.t(), h, down_s)  # [r, INTER]

        # Up/gate: each dXA* computed once, reused for the adapter-A gradient
        # and the dX accumulation.
        dXAu = _scaled_matmul(grad_up, up_B, up_s)  # [T, r]
        d_up_B = _scaled_matmul(grad_up.t(), XAu, up_s)  # [INTER, r]
        d_up_A = torch.matmul(dXAu.t(), X)  # [r, HIDDEN]

        dXAg = _scaled_matmul(grad_gate, gate_B, gate_s)  # [T, r]
        d_gate_B = _scaled_matmul(grad_gate.t(), XAg, gate_s)  # [INTER, r]
        d_gate_A = torch.matmul(dXAg.t(), X)  # [r, HIDDEN]

        # dX: 1 write + 2 RMW passes. When inplace, dX reuses X's storage --
        # safe because every read of X (d_up_A/d_gate_A above) has already
        # happened, and it saves a full [T, HIDDEN] allocation.
        dX = X if ctx.inplace else torch.empty_like(X)
        torch.matmul(grad_up, up_W, out=dX)
        dX.addmm_(grad_gate, gate_W)
        dXA_gu = torch.cat([dXAg, dXAu], dim=1)  # [T, 2r]
        dX.addmm_(dXA_gu, A_gu)

        if ctx.reshape is not None:
            dX = dX.view(batch, seq_len, hd)

        return (
            dX,
            None,
            d_gate_A,
            d_gate_B,
            None,
            None,
            d_up_A,
            d_up_B,
            None,
            None,
            d_down_A,
            d_down_B,
            None,
            None,
        )


def _dropout_is_noop(dropout, training: bool) -> bool:
    if dropout is None or not training:
        return True
    if isinstance(dropout, torch.nn.Identity):
        return True
    return getattr(dropout, "p", 0.0) == 0.0


@torch.compiler.disable
def apply_lora_mlp_swiglu_b200(self, X: torch.Tensor, inplace: bool = True):
    """Module-swap forward for the B200 factored LoRA MLP kernel.

    Patch-time gating (`support.lora_mlp_b200_config_eligible` /
    `lora_mlp_b200_module_supported`) should make the fallback here
    unreachable; it exists so a state change after patching (e.g. an adapter
    toggled off) degrades to the standard kernel instead of wrong numbers.
    """
    from axolotl.kernels.lora import apply_lora_mlp_swiglu, get_lora_parameters

    gate_W, gate_b, gate_q, gate_A, gate_B, gate_s, gate_lb, gate_drop, gate_mag = (
        get_lora_parameters(self.gate_proj)
    )
    up_W, up_b, up_q, up_A, up_B, up_s, up_lb, _, up_mag = get_lora_parameters(
        self.up_proj
    )
    down_W, down_b, down_q, down_A, down_B, down_s, down_lb, _, down_mag = (
        get_lora_parameters(self.down_proj)
    )

    supported = (
        all(t is None for t in (gate_b, up_b, down_b, gate_q, up_q, down_q))
        and all(
            t is None for t in (gate_lb, up_lb, down_lb, gate_mag, up_mag, down_mag)
        )
        and all(t is not None for t in (gate_A, gate_B, up_A, up_B, down_A, down_B))
        and _dropout_is_noop(gate_drop, self.training)
        and X.dtype == torch.bfloat16
        and gate_W.dtype == torch.bfloat16
    )
    if not supported:
        LOG.warning_once(
            "lora_mlp_kernel_b200: MLP state changed after patching; using the "
            "standard LoRA MLP kernel for this module"
        )
        return apply_lora_mlp_swiglu(self, X, inplace)

    return LoRA_MLP_B200_Factored.apply(
        X,
        gate_W,
        gate_A,
        gate_B,
        gate_s,
        up_W,
        up_A,
        up_B,
        up_s,
        down_W,
        down_A,
        down_B,
        down_s,
        inplace,
    )
