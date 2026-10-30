"""Fused token gather/scatter for the grouped-expert padded layout.

The grouped forward moves tokens between the [N, H] token space and the [Mt, H] expert-padded
space twice per layer. As torch ops each move is 2-3 full passes over the data (gather temp,
weight multiply temp, index_add/index_copy); each kernel here is ONE pass with the routing-weight
multiply applied in-register (which also makes a GEMM-epilogue weight multiply moot).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_rows(
    SRC,
    DST,
    SIDX,
    DIDX,
    W,
    H,
    ss,
    sd,
    ATOMIC: tl.constexpr,
    HAS_W: tl.constexpr,
    BLK: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLK + tl.arange(0, BLK)
    m = c < H
    si = tl.load(SIDX + t).to(tl.int64)
    di = tl.load(DIDX + t).to(tl.int64)
    v = tl.load(SRC + si * ss + c, mask=m, other=0.0).to(tl.float32)
    if HAS_W:
        v = v * tl.load(W + t).to(tl.float32)
    if ATOMIC:
        tl.atomic_add(DST + di * sd + c, v.to(DST.dtype.element_ty), mask=m)
    else:
        tl.store(DST + di * sd + c, v.to(DST.dtype.element_ty), mask=m)


def _launch(src, dst, sidx, didx, w, atomic):
    T = sidx.numel()
    if T == 0:
        return
    H = src.size(1)
    _copy_rows[(T, triton.cdiv(H, 1024))](
        src,
        dst,
        sidx,
        didx,
        w if w is not None else src,
        H,
        src.stride(0),
        dst.stride(0),
        ATOMIC=atomic,
        HAS_W=w is not None,
        BLK=1024,
    )


class _PaddedGather(torch.autograd.Function):
    """A = zeros(Mt, H); A[padded_row] = hidden[rep] — one pass each way."""

    @staticmethod
    def forward(ctx, hidden, rep, padded_row, Mt):
        ctx.save_for_backward(rep, padded_row)
        ctx.N = hidden.size(0)
        A = hidden.new_zeros(Mt, hidden.size(1))
        _launch(hidden, A, rep, padded_row, None, atomic=False)
        return A

    @staticmethod
    def backward(ctx, gA):
        rep, padded_row = ctx.saved_tensors
        gh = gA.new_zeros(ctx.N, gA.size(1))
        # rep repeats each token top_k times -> accumulate
        _launch(gA.contiguous(), gh, padded_row, rep, None, atomic=True)
        return gh, None, None, None


_ROUTER_GRAD_ROWS = 8192


class _WeightedScatter(torch.autograd.Function):
    """out = zeros(N, H); out[rep[t]] += dn[padded_row[t]] * w[t] — multiply in-register."""

    @staticmethod
    def forward(ctx, dn, w, rep, padded_row, N):
        if w.requires_grad:
            ctx.save_for_backward(w, rep, padded_row, dn)
        else:
            ctx.save_for_backward(w, rep, padded_row)
        ctx.Mt = dn.size(0)
        out = dn.new_zeros(N, dn.size(1))
        _launch(dn, out, padded_row, rep, w, atomic=True)
        return out

    @staticmethod
    def backward(ctx, go):
        # one unpack: non-reentrant activation checkpointing rejects a second one
        saved = ctx.saved_tensors
        w, rep, padded_row = saved[:3]
        go = go.contiguous()
        gdn = go.new_zeros(ctx.Mt, go.size(1))
        _launch(go, gdn, rep, padded_row, w, atomic=False)
        gw = None
        if ctx.needs_input_grad[1]:
            dn = saved[3]
            # row chunks bound the two fp32 [rows, H] temporaries (rows = tokens * top_k)
            gw = torch.empty(rep.numel(), device=go.device, dtype=torch.float32)
            for s in range(0, rep.numel(), _ROUTER_GRAD_ROWS):
                e = s + _ROUTER_GRAD_ROWS
                gw[s:e] = (dn[padded_row[s:e]].float() * go[rep[s:e]].float()).sum(-1)
            gw = gw.to(w.dtype)
        return gdn, gw, None, None, None


def padded_gather(hidden, rep, padded_row, Mt):
    if not hidden.is_cuda:
        return hidden.new_zeros(Mt, hidden.size(1)).index_copy(
            0, padded_row, hidden[rep]
        )
    return _PaddedGather.apply(hidden.contiguous(), rep, padded_row, Mt)


def weighted_scatter(dn, wflat, rep, padded_row, N):
    if not dn.is_cuda:
        return dn.new_zeros(N, dn.size(1)).index_add(
            0, rep, dn[padded_row] * wflat[:, None].to(dn.dtype)
        )
    return _WeightedScatter.apply(dn.contiguous(), wflat, rep, padded_row, N)
