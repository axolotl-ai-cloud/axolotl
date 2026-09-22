"""Basic ring attention over batched (non-packed) inputs: KV blocks circulate around the
ring and each rank merges partial outputs with the online-softmax log-sum-exp trick.

Vendored from ring-flash-attention (https://github.com/zhuzilin/ring-flash-attention,
MIT License, Copyright 2024 Zilin Zhu), generalised to run on any `FlashAttnBackend`.
"""

import torch

from axolotl.monkeypatch.ring_attn.backends import NO_WINDOW, FlashAttnBackend
from axolotl.monkeypatch.ring_attn.utils import RingComm, update_out_and_lse


def ring_flash_attn_forward(  # pylint: disable=too-many-arguments
    backend: FlashAttnBackend,
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0.0,
    causal=True,
    window_size=NO_WINDOW,
    softcap=0.0,
):
    comm = RingComm(process_group)

    out = None
    lse = None

    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)

        # Under causal masking rank r only attends to KV blocks from ranks <= r, and
        # only its own block (step 0) needs the causal mask.
        if not causal or step <= comm.rank:
            block_out, block_lse = backend.forward(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal and step == 0,
                window_size=window_size,
                softcap=softcap,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def ring_flash_attn_backward(  # pylint: disable=too-many-arguments
    backend: FlashAttnBackend,
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0.0,
    causal=True,
    window_size=NO_WINDOW,
    softcap=0.0,
    deterministic=False,
):
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None

    block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    next_k, next_v = None, None

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        if step <= kv_comm.rank or not causal:
            backend.backward(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                block_dq_buffer,
                block_dk_buffer,
                block_dv_buffer,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal and step == 0,
                window_size=window_size,
                softcap=softcap,
                deterministic=deterministic,
            )

            if dq is None:
                dq = block_dq_buffer.to(torch.float32)
                dk = block_dk_buffer.to(torch.float32)
                dv = block_dv_buffer.to(torch.float32)
            else:
                dq += block_dq_buffer
                d_kv_comm.wait()
                dk = block_dk_buffer + next_dk
                dv = block_dv_buffer + next_dv
        elif step != 0:
            d_kv_comm.wait()
            dk, dv = next_dk, next_dv

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

        next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv)

    d_kv_comm.wait()

    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class RingFlashAttnFunc(torch.autograd.Function):
    """Autograd wrapper around the batch ring forward/backward."""

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments
        ctx,
        q,
        k,
        v,
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
        out, softmax_lse = ring_flash_attn_forward(
            backend,
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse)
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
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = ring_flash_attn_backward(
            ctx.backend,
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None


def ring_flash_attn_func(  # pylint: disable=too-many-arguments
    q,
    k,
    v,
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
    return RingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        deterministic,
        group,
        backend,
    )
