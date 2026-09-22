"""Llama3-style varlen ring attention (all-gather KV, per-rank varlen flash attention).

Vendored from ring-flash-attention (https://github.com/zhuzilin/ring-flash-attention,
MIT License, Copyright 2024 Zilin Zhu), generalised to run on any `FlashAttnBackend`.
"""

import torch
import torch.distributed as dist

from axolotl.monkeypatch.ring_attn.backends import NO_WINDOW, FlashAttnBackend
from axolotl.monkeypatch.ring_attn.utils import AllGatherComm


def llama3_flash_attn_prepare_cu_seqlens(
    cu_seqlens: torch.Tensor, causal: bool, rank: int, world_size: int
):
    """Slice the global packed-sequence boundaries down to this rank's chunk.

    Args:
        cu_seqlens: Cumulative sequence lengths across the whole ring process group.

    Returns:
        cu_seqlens_q: Boundaries of this rank's q slice.
        cu_seqlens_k: Boundaries of the k slice the local q needs; may be longer than
            `total_seq_len // world_size`.
        max_seqlen_q, max_seqlen_k: Longest sequence in each slice.
        local_k_slice: Slice of the gathered k that the local q needs.
    """
    total_length = cu_seqlens[-1]
    assert total_length % world_size == 0
    length_per_rank = total_length // world_size
    left = torch.searchsorted(cu_seqlens, rank * length_per_rank)
    right = torch.searchsorted(cu_seqlens, (rank + 1) * length_per_rank)
    length_per_rank = length_per_rank.item()

    # after this, cu_seqlens[left:right + 1] contains all the sequence for this rank
    if cu_seqlens[left] != rank * length_per_rank:
        left -= 1
    left = left.item()
    right = right.item()

    # q is always the same. just calculate the cu_seqlens for the local slice
    cu_seqlens_q = cu_seqlens[left : right + 1].clone()
    cu_seqlens_q -= rank * length_per_rank
    cu_seqlens_q[0] = 0
    cu_seqlens_q[-1] = length_per_rank

    cu_seqlens_k = cu_seqlens[left : right + 1].clone()
    if causal:
        # the last k seq ends where the last q seq ends
        slice_right = (rank + 1) * length_per_rank
        cu_seqlens_k[-1] = slice_right
    else:
        # the last k is the full sequence
        slice_right = cu_seqlens[right].item()

    slice_left = cu_seqlens[left].item()
    cu_seqlens_k -= slice_left

    max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
    local_k_slice = slice(slice_left, slice_right)
    return cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, local_k_slice


def llama3_flash_attn_varlen_forward(  # pylint: disable=too-many-arguments
    backend: FlashAttnBackend,
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    heads_k_stride,
    local_k_slice,
    softmax_scale,
    dropout_p=0.0,
    causal=True,
    window_size=NO_WINDOW,
    softcap=0.0,
):
    out_list = []
    lse_list = []

    nheads = q.shape[1]
    total_k, nheads_k, head_dim = k.shape
    assert nheads_k % heads_k_stride == 0

    world_size = dist.get_world_size(process_group)
    kv_buffer = torch.empty(
        (2, total_k * world_size, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    )

    kv_buffer_copy = torch.empty_like(kv_buffer)

    k_0 = k[:, :heads_k_stride].contiguous()
    v_0 = v[:, :heads_k_stride].contiguous()
    comm = AllGatherComm(process_group)

    comm.all_gather(kv_buffer_copy[0], k_0)
    comm.all_gather(kv_buffer_copy[1], v_0)

    for i in range(0, nheads_k, heads_k_stride):
        comm.wait()
        kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

        if i < nheads_k - heads_k_stride:
            # all_gather the next kv slice while this one computes
            kv_slice_left = i + heads_k_stride
            kv_slice_right = kv_slice_left + heads_k_stride
            send_k = k[:, kv_slice_left:kv_slice_right].contiguous()
            send_v = v[:, kv_slice_left:kv_slice_right].contiguous()
            comm.all_gather(kv_buffer_copy[0], send_k)
            comm.all_gather(kv_buffer_copy[1], send_v)

        q_i = q[:, i * nheads // nheads_k : (i + heads_k_stride) * nheads // nheads_k]
        k_i = kv_buffer[0][local_k_slice]
        v_i = kv_buffer[1][local_k_slice]

        out, lse = backend.forward(
            q_i,
            k_i,
            v_i,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
        )
        out_list.append(out)
        lse_list.append(lse)

    out = torch.cat(out_list, dim=1)
    lse = torch.cat(lse_list, dim=-2)
    return out, lse


def llama3_flash_attn_varlen_backward(  # pylint: disable=too-many-arguments
    backend: FlashAttnBackend,
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    heads_k_stride,
    local_k_slice,
    softmax_scale,
    dropout_p=0.0,
    causal=True,
    window_size=NO_WINDOW,
    softcap=0.0,
    deterministic=False,
):
    nheads = q.shape[1]
    total_k, nheads_k, head_dim = k.shape
    assert nheads_k % heads_k_stride == 0

    world_size = dist.get_world_size(process_group)
    kv_buffer = torch.empty(
        (2, total_k * world_size, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    )
    kv_buffer_copy = torch.empty_like(kv_buffer)

    dkv_buffer = torch.empty(
        (2, total_k * world_size, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    )

    strided_heads = heads_k_stride != nheads_k
    if strided_heads:
        kv_contiguous_buffer = torch.empty(
            (2, total_k, heads_k_stride, head_dim),
            dtype=k.dtype,
            device=k.device,
        )

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    comm = AllGatherComm(process_group)

    k_0 = k[:, :heads_k_stride].contiguous()
    v_0 = v[:, :heads_k_stride].contiguous()
    comm.all_gather(kv_buffer_copy[0], k_0)
    comm.all_gather(kv_buffer_copy[1], v_0)

    for i in range(0, nheads_k, heads_k_stride):
        dkv_buffer.zero_()

        q_slice = slice(
            i * nheads // nheads_k, (i + heads_k_stride) * nheads // nheads_k
        )
        q_i = q[:, q_slice]
        dout_i = dout[:, q_slice]
        out_i = out[:, q_slice]
        # Kernels differ in whether they accept head-strided gradient buffers, so hand
        # them a contiguous one and scatter back.
        dq_i = (
            torch.empty(q_i.shape, dtype=q.dtype, device=q.device)
            if strided_heads
            else dq
        )
        if softmax_lse.dim() == 3:
            lse_i = softmax_lse[:, q_slice].contiguous()
        else:
            lse_i = softmax_lse[q_slice]

        comm.wait()
        kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

        if i < nheads_k - heads_k_stride:
            kv_slice_left = i + heads_k_stride
            kv_slice_right = kv_slice_left + heads_k_stride
            send_k = k[:, kv_slice_left:kv_slice_right].contiguous()
            send_v = v[:, kv_slice_left:kv_slice_right].contiguous()
            comm.all_gather(kv_buffer_copy[0], send_k)
            comm.all_gather(kv_buffer_copy[1], send_v)

        k_i = kv_buffer[0][local_k_slice]
        v_i = kv_buffer[1][local_k_slice]
        dk_i = dkv_buffer[0][local_k_slice]
        dv_i = dkv_buffer[1][local_k_slice]

        backend.backward(
            dout_i,
            q_i,
            k_i,
            v_i,
            out_i,
            lse_i,
            dq_i,
            dk_i,
            dv_i,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            deterministic=deterministic,
        )

        if strided_heads:
            dq[:, q_slice] = dq_i
            # reduce_scatter needs contiguous buffers
            dk_i = kv_contiguous_buffer[0]
            dv_i = kv_contiguous_buffer[1]
        else:
            dk_i = dk
            dv_i = dv

        dist.reduce_scatter_tensor(dk_i, dkv_buffer[0], group=process_group)
        dist.reduce_scatter_tensor(dv_i, dkv_buffer[1], group=process_group)

        if strided_heads:
            dk[:, i : i + heads_k_stride] = dk_i
            dv[:, i : i + heads_k_stride] = dv_i

    return dq, dk, dv


class Llama3FlashAttnVarlenFunc(torch.autograd.Function):
    """Autograd wrapper around the llama3 varlen ring forward/backward."""

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        heads_k_stride,
        local_k_slice,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        deterministic,
        group,
        backend,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = llama3_flash_attn_varlen_forward(
            backend,
            group,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            heads_k_stride,
            local_k_slice,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.heads_k_stride = heads_k_stride
        ctx.local_k_slice = local_k_slice
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.group = group
        ctx.backend = backend
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = llama3_flash_attn_varlen_backward(
            ctx.backend,
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.heads_k_stride,
            ctx.local_k_slice,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            deterministic=ctx.deterministic,
        )
        return (dq, dk, dv) + (None,) * 14


def llama3_flash_attn_varlen_func(  # pylint: disable=too-many-arguments
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    heads_k_stride,
    local_k_slice,
    *,
    backend: FlashAttnBackend,
    group,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=NO_WINDOW,
    softcap=0.0,
    deterministic=False,
):
    return Llama3FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        heads_k_stride,
        local_k_slice,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        deterministic,
        group,
        backend,
    )
