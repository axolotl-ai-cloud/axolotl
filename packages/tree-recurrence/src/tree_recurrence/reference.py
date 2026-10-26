# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""PyTorch references of the kernels the components run, with the kernels' APIs.

Each takes the boundaries its kernel takes (``cu_seqlens`` or ``seq_idx``) and
per-sequence initial states, and runs token by token in float32, so tree outputs and
gradients can be checked against unpacked runs on CPU. They are slow by design.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _wide(x: torch.Tensor) -> torch.Tensor:
    """At least float32: float64 stays, for checks in double precision."""
    return x if x.dtype == torch.float64 else x.float()


def _spans(seq_idx: torch.Tensor | None, total: int) -> list[tuple[int, int]]:
    if seq_idx is None:
        return [(0, total)]
    ids = seq_idx.tolist()
    starts = [t for t in range(total) if t == 0 or ids[t] != ids[t - 1]]
    return list(zip(starts, starts[1:] + [total], strict=True))


def _sequences(
    batch: int, total: int, cu_seqlens: torch.Tensor | None
) -> list[tuple[int, int, int]]:
    if cu_seqlens is None:
        return [(row, 0, total) for row in range(batch)]
    if batch != 1:
        raise ValueError("cu_seqlens needs a batch of one")
    bounds = cu_seqlens.tolist()
    return [(0, start, end) for start, end in zip(bounds[:-1], bounds[1:], strict=True)]


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    seq_idx: torch.Tensor | None = None,
    activation: str | None = None,
    **kwargs,
) -> torch.Tensor:
    """``causal_conv1d_fn``: a depthwise causal conv over ``x`` ``[B, D, T]`` with
    ``weight`` ``[D, W]``, restarting at ``seq_idx`` boundaries."""
    width, channels = weight.shape[-1], x.shape[1]
    rows = []
    for row in range(x.shape[0]):
        spans = _spans(None if seq_idx is None else seq_idx[row], x.shape[-1])
        rows.append(
            torch.cat(
                [
                    F.conv1d(
                        F.pad(
                            x[row : row + 1, :, start:end].to(weight.dtype),
                            (width - 1, 0),
                        ),
                        weight[:, None],
                        bias,
                        groups=channels,
                    )
                    for start, end in spans
                ],
                dim=-1,
            )
        )
    out = torch.cat(rows)
    if activation in ("silu", "swish"):
        out = F.silu(out)
    return out.to(x.dtype)


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _delta_rule(q, k, v, g, beta, scale, initial_state, l2norm, cu_seqlens, per_key):
    dtype = v.dtype
    q, k, v, g, beta = (_wide(t) for t in (q, k, v, g, beta))
    if l2norm:
        q, k = _l2norm(q), _l2norm(k)
    heads = v.shape[2]
    if q.shape[2] != heads:
        q = q.repeat_interleave(heads // q.shape[2], dim=2)
        k = k.repeat_interleave(heads // k.shape[2], dim=2)
    q = q * (q.shape[-1] ** -0.5 if scale is None else scale)
    outs, finals = [], []
    for i, (row, start, end) in enumerate(
        _sequences(v.shape[0], v.shape[1], cu_seqlens)
    ):
        if initial_state is None:
            state = v.new_zeros(heads, k.shape[-1], v.shape[-1])
        else:
            state = _wide(initial_state[i])
        steps = []
        for t in range(start, end):
            decay = g[row, t].exp()
            state = state * (decay[..., None] if per_key else decay[:, None, None])
            remembered = (state * k[row, t][..., None]).sum(-2)
            delta = (v[row, t] - remembered) * beta[row, t][..., None]
            state = state + k[row, t][..., None] * delta[..., None, :]
            steps.append((state * q[row, t][..., None]).sum(-2))
        outs.append(torch.stack(steps))
        finals.append(state)
    out = torch.cat(outs)[None] if cu_seqlens is not None else torch.stack(outs)
    return out.to(dtype), torch.stack(finals)


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """FLA's ``chunk_gated_delta_rule``: ``q``/``k`` ``[B, T, H, K]``, ``v``
    ``[B, T, H, V]``, ``g``/``beta`` ``[B, T, H]``."""
    out, final = _delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        False,
    )
    return out, final if output_final_state else None


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """FLA's ``chunk_kda``: the delta rule with a decay per key channel, ``g``
    ``[B, T, H, K]``."""
    out, final = _delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        True,
    )
    return out, final if output_final_state else None


def mamba_chunk_scan_combined(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    seq_idx: torch.Tensor | None = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    return_final_states: bool = False,
    **kwargs,
):
    """mamba-ssm's SSD scan: ``x`` ``[B, T, H, P]``, ``dt`` ``[B, T, H]``, ``B``/``C``
    ``[B, T, G, N]``, restarting at ``seq_idx`` boundaries."""
    batch, total, heads, head_dim = x.shape
    dt = _wide(dt) + (0 if dt_bias is None else _wide(dt_bias))
    if dt_softplus:
        dt = F.softplus(dt)
    dt = dt.clamp(*dt_limit)
    B = _wide(B).repeat_interleave(heads // B.shape[2], dim=2)
    C = _wide(C).repeat_interleave(heads // C.shape[2], dim=2)
    skip = None if D is None else (_wide(D) if D.dim() == 2 else _wide(D)[:, None])
    rows, finals = [], []
    for row in range(batch):
        steps = []
        for start, end in _spans(None if seq_idx is None else seq_idx[row], total):
            state = B.new_zeros(heads, head_dim, B.shape[-1])
            if initial_states is not None and start == 0:
                state = _wide(initial_states[row])
            for t in range(start, end):
                xt = _wide(x[row, t])
                decay = torch.exp(dt[row, t] * _wide(A))[:, None, None]
                state = (
                    state * decay
                    + (dt[row, t, :, None] * xt)[..., None] * B[row, t, :, None]
                )
                y = (state * C[row, t, :, None]).sum(-1)
                steps.append(y if skip is None else y + skip * xt)
        rows.append(torch.stack(steps))
        finals.append(state)
    out = torch.stack(rows)
    if z is not None:
        out = out * F.silu(_wide(z))
    out = out.to(x.dtype)
    return (out, torch.stack(finals)) if return_final_states else out


def _gated_rmsnorm(x, z, weight, eps, group_size, norm_before_gate):
    dtype = x.dtype
    x, z = _wide(x), _wide(z)
    if not norm_before_gate:
        x = x * F.silu(z)
    grouped = x.reshape(*x.shape[:-1], -1, group_size)
    grouped = grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + eps)
    x = grouped.reshape(x.shape) * _wide(weight)
    if norm_before_gate:
        x = x * F.silu(z)
    return x.to(dtype)


def mamba_split_conv1d_scan_combined(
    zxbcdt: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor | None,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int,
    initial_states: torch.Tensor | None = None,
    seq_idx: torch.Tensor | None = None,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    return_final_states: bool = False,
    activation: str = "silu",
    rmsnorm_weight: torch.Tensor | None = None,
    rmsnorm_eps: float = 1e-6,
    outproj_weight: torch.Tensor | None = None,
    outproj_bias: torch.Tensor | None = None,
    headdim: int | None = None,
    ngroups: int = 1,
    norm_before_gate: bool = True,
    **kwargs,
):
    """mamba-ssm's fused Mamba2 mixer: split the input projection, conv, SSD scan,
    gated RMSNorm and output projection, restarting at ``seq_idx`` boundaries."""
    heads = D.shape[0]
    head_dim = D.shape[1] if D.dim() == 2 else headdim
    assert head_dim is not None, "headdim is needed with a per-head D"
    d_inner = heads * head_dim
    state_size = (conv1d_weight.shape[0] - d_inner) // ngroups // 2
    groups_state = ngroups * state_size
    if zxbcdt.shape[-1] != 2 * d_inner + 2 * groups_state + heads:
        raise NotImplementedError("the reference has no gated MLP channels")
    batch, total, _ = zxbcdt.shape
    z, xBC, dt = zxbcdt.split([d_inner, d_inner + 2 * groups_state, heads], dim=-1)
    xBC = causal_conv1d(
        xBC.transpose(1, 2), conv1d_weight, conv1d_bias, seq_idx, activation
    ).transpose(1, 2)
    x, B, C = xBC.split([d_inner, groups_state, groups_state], dim=-1)
    scanned = mamba_chunk_scan_combined(
        x.reshape(batch, total, heads, head_dim),
        dt,
        A,
        B.reshape(batch, total, ngroups, state_size),
        C.reshape(batch, total, ngroups, state_size),
        chunk_size,
        D=D,
        dt_bias=dt_bias,
        initial_states=initial_states,
        seq_idx=seq_idx,
        dt_softplus=True,
        dt_limit=dt_limit,
        return_final_states=return_final_states,
    )
    y, final = scanned if return_final_states else (scanned, None)
    y = y.reshape(batch, total, d_inner)
    if rmsnorm_weight is None:
        y = y * F.silu(z)
    else:
        y = _gated_rmsnorm(
            y, z, rmsnorm_weight, rmsnorm_eps, d_inner // ngroups, norm_before_gate
        )
    if outproj_weight is not None:
        y = F.linear(y, outproj_weight, outproj_bias)
    return (y, final) if return_final_states else y
