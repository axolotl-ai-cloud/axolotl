"""Context-parallel DSA attention for GLM-5.2.

GLM-5.2 is always multi-GPU; at long context the sequence is sharded across a context-parallel (CP)
group. The MLA compression makes CP cheap: each rank needs every key, but a key is only the 576-wide
*compressed* ``k_shared`` (kv_lora 512 + rope 64) — ~1.1 KB/token — NOT the per-head 64x256 expanded
keys (~32 KB/token). So we all-gather the compressed KV (a ~28x smaller collective than gathering
expanded K/V) and each rank attends its local queries against the global KV with the absorbed kernel
(already global-KV-aware via ``q_offset`` + global top-k indices).

The all-gather is differentiable (its backward reduce-scatters the per-rank ``dk_shared`` grads), so
training is correct. ``topk_idx`` must reference GLOBAL key positions (the indexer scores against the
same gathered KV). Sequence sharding here is contiguous (rank r owns ``[r·L, (r+1)·L)``).
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .attention_mla_absorb import absorb_query
from .dispatch import mla_attn

_COMM_STREAM = {}


def _comm_stream(dev):
    if dev not in _COMM_STREAM:
        _COMM_STREAM[dev] = torch.cuda.Stream(device=dev)
    return _COMM_STREAM[dev]


class _OverlapGather(torch.autograd.Function):
    """Differentiable seq all-gather whose data movement was already started (async) before this op,
    so it overlaps with intervening compute. Backward all-reduces the global grad and slices the
    rank-local part (== the reduce-scatter the standard differentiable all-gather does)."""

    @staticmethod
    def forward(ctx, k_local, gathered, work, group, world, rank):
        if work is not None:
            work.wait()
        ctx.group, ctx.world, ctx.rank, ctx.sl = group, world, rank, k_local.shape[1]
        return torch.cat(gathered, dim=1)

    @staticmethod
    def backward(ctx, dg):
        dg = dg.contiguous()
        dist.all_reduce(dg, op=dist.ReduceOp.SUM, group=ctx.group)
        s = slice(ctx.rank * ctx.sl, (ctx.rank + 1) * ctx.sl)
        return dg[:, s].contiguous(), None, None, None, None, None


def cp_mla_attn_overlapped(
    q_pass, q_rot, w_kb_k, k_shared, topk_idx, scale, group=None, crossover=None
):
    """Comm-overlapped CP attention: issue the compressed-KV all-gather on a side stream, compute the
    absorption GEMM (local, ~hidden behind the gather), then attend. Same result as cp_mla_attn but
    the absorption is hidden behind the collective. Takes the RAW query projections so the GEMM can
    run in the overlap window. Returns the local out_latent."""
    world = dist.get_world_size(group) if dist.is_initialized() else 1
    if world == 1:
        q_abs = absorb_query(q_pass, q_rot, w_kb_k)
        return mla_attn(
            q_abs, k_shared, topk_idx, scale, q_offset=0, crossover=crossover
        )
    rank = dist.get_rank(group)
    s_local = q_pass.shape[2]
    gathered = [torch.empty_like(k_shared) for _ in range(world)]
    stream = _comm_stream(k_shared.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        work = dist.all_gather(
            gathered, k_shared.contiguous(), group=group, async_op=True
        )
    q_abs = absorb_query(
        q_pass, q_rot, w_kb_k
    )  # overlaps with the gather on the side stream
    torch.cuda.current_stream().wait_stream(stream)
    k_global = _OverlapGather.apply(k_shared, gathered, work, group, world, rank)
    return mla_attn(
        q_abs, k_global, topk_idx, scale, q_offset=rank * s_local, crossover=crossover
    )


def all_gather_seq(x: torch.Tensor, group=None) -> torch.Tensor:
    """Differentiable all-gather along the sequence dim (1): [B,S_local,D] -> [B,S_global,D]."""
    world = dist.get_world_size(group) if dist.is_initialized() else 1
    if world == 1:
        return x
    import torch.distributed.nn as dist_nn

    parts = dist_nn.all_gather(x.contiguous(), group=group)
    return torch.cat(list(parts), dim=1)


def cp_mla_attn(q_abs, k_shared, topk_idx, scale, group=None, crossover=None):
    """Context-parallel absorbed DSA attention. ``q_abs`` [B,H,S_local,576], ``k_shared``
    [B,S_local,576], ``topk_idx`` [B,S_local,T] (GLOBAL key positions) are the rank-local shards.
    All-gathers the compressed KV (cheap), then runs the local queries against the global KV with the
    correct ``q_offset``. Returns the local out_latent [B,H,S_local,kv_lora]."""
    rank = dist.get_rank(group) if dist.is_initialized() else 0
    s_local = q_abs.shape[2]
    k_global = all_gather_seq(
        k_shared, group
    )  # differentiable; backward reduce-scatters dk
    return mla_attn(
        q_abs, k_global, topk_idx, scale, q_offset=rank * s_local, crossover=crossover
    )
