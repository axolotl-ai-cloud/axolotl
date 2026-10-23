# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""SonicMoE expert LoRA without materializing ``W + s * B @ A`` ("LoRA split").

The frozen base expert weights stay in sonic-moe's fused quack GEMMs. The LoRA term enters them
as a low-rank addend ``C``: ``s1 * (x P1) Q1`` before the gated epilogue of the gate/up GEMM and
``s2 * (a P2) Q2`` in the down GEMM. In backward, ``gemm_dgated`` (which takes no addend) produces
the base part of the gate/up gradient and the down LoRA correction is added through the GLU
backward, which is linear in its upstream gradient. LoRA gradients come from rank-r GEMMs only, so
no merged weight and no full-rank weight gradient is ever built.

Per routed pair ``p`` (row vectors, expert ``e``, router score ``sigma``), with the base linears
``M1 [H, 2I]`` / ``M2 [I, H]`` and LoRA ``P1 [H, r] Q1 [r, 2I]`` / ``P2 [I, r] Q2 [r, H]``::

    h = x M1 + s1 (x P1) Q1            a = glu(h)            y = a M2 + s2 (a P2) Q2
    o_t = sum_p sigma_p y_p

    g = sigma G                        w = G Q2^T            v = a P2
    dsigma = <G M2^T, a> + s2 <w, v>
    dh = glu'(h)[sigma G M2^T] + glu'(h)[s2 sigma w P2^T]
    dx = dh M1^T + s1 (dh Q1^T) P1^T
    dQ2 = s2 sum (sigma v)^T G         dP2 = s2 sum (sigma a)^T w
    dQ1 = s1 sum (x P1)^T dh           dP1 = s1 sum x^T (dh Q1^T)

``TorchOps`` implements every primitive in plain torch with the kernels' semantics, so the whole
autograd function runs on CPU; ``SonicOps`` binds the same primitives to the loaded hub kernel.
"""

from __future__ import annotations

import importlib
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F

_LORA_SPLIT: dict = {"enabled": False}
_LORA_SPLIT_WARNED: set[str] = set()

GLU_EPILOGUES = {"silu": "swiglu", "gelu": "geglu"}


def configure_lora_split(enabled: bool = False) -> None:
    _LORA_SPLIT["enabled"] = bool(enabled)
    _LORA_SPLIT_WARNED.clear()


def lora_split_unsupported_reason(
    module, hidden_states: torch.Tensor, w1, b1, w2, b2
) -> str | None:
    """Why the split path cannot run this (LoRA-carrying) experts call, or ``None``."""
    from .nvfp4 import resolve_gated_activation

    if not getattr(module, "has_gate", True):
        return "non-gated experts"
    if any(t.dtype != torch.bfloat16 for t in (hidden_states, w1, w2)):
        return "non-bf16 experts"
    if w1.requires_grad or w2.requires_grad:
        return "trainable base expert weights"
    if b1 is not None or b2 is not None:
        return "expert bias"
    if getattr(module, "has_post_expert_norm", False):
        return "post-expert norm"
    act = resolve_gated_activation(module.config)
    act = {"gelu_pytorch_tanh": "gelu", "gelu_tanh": "gelu"}.get(act, act)
    if act not in GLU_EPILOGUES:
        return f"{act!r} activation"
    # quack reads a concat layout along the non-contiguous dim, which is H for [E, H, 2I]
    if getattr(module, "is_transposed", False) and getattr(
        module, "is_concatenated", True
    ):
        return "transposed concatenated gate_up layout"
    return None


def lora_split_applies(module, hidden_states, w1, b1, w2, b2) -> bool:
    if not _LORA_SPLIT["enabled"]:
        return False
    reason = lora_split_unsupported_reason(module, hidden_states, w1, b1, w2, b2)
    if reason is None:
        return True
    if reason not in _LORA_SPLIT_WARNED:
        _LORA_SPLIT_WARNED.add(reason)
        from axolotl.utils.logging import get_logger

        get_logger(__name__).warning(
            "sonicmoe_lora_split: %s, materializing the expert LoRA instead", reason
        )
    return False


def lora_split_factors(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    num_experts: int,
    *,
    input_dim_last: bool,
    interleave: bool,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert ``(P [E, in, r], Q [E, r, out])`` with ``x @ (W_eff_e - W_e) == s (x P_e) Q_e``.

    PEFT's layout is ``lora_A [r*E, dim2]`` (expert-major rows) and ``lora_B [dim1, r*E]``
    (expert-minor columns), ``W_eff_e = W_e + s B_e A_e`` on a ``[E, dim1, dim2]`` base.
    ``input_dim_last``: the base is ``[E, out, in]`` (applied as ``x @ W_e.T``) rather than
    ``[E, in, out]``. ``interleave`` reorders a concatenated ``[gate; up]`` output into the
    interleaved columns the gated kernels produce. Differentiable, so autograd maps the
    gradients back to PEFT's layout.
    """
    r = lora_A.shape[0] // num_experts
    a = lora_A.reshape(num_experts, r, lora_A.shape[1])
    b = lora_B.reshape(lora_B.shape[0], r, num_experts).permute(2, 0, 1)
    p, q = (a.transpose(1, 2), b.transpose(1, 2)) if input_dim_last else (b, a)
    if interleave:
        q = q.unflatten(-1, (2, q.shape[-1] // 2)).transpose(-1, -2).flatten(-2)
    return p.to(dtype).contiguous(), q.to(dtype).contiguous()


class RoutingMetadata(NamedTuple):
    """sonic-moe's general-routing metadata. ``s_reverse_scatter_idx`` is ``TK`` for a sentinel."""

    cu_seqlens: torch.Tensor
    x_gather_idx: torch.Tensor
    s_scatter_idx: torch.Tensor
    s_reverse_scatter_idx: torch.Tensor
    token_offset: torch.Tensor


def glu_forward(gate: torch.Tensor, up: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "swiglu":
        return F.silu(gate) * up
    return F.gelu(gate, approximate="tanh") * up


def glu_backward(
    gate: torch.Tensor, up: torch.Tensor, d: torch.Tensor, activation: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(dgate, dup)`` for upstream ``d``, as quack's ``dswiglu`` / ``dgeglu`` compute them."""
    if activation == "swiglu":
        sig = torch.sigmoid(gate)
        act = gate * sig
        dact = sig + act * (1 - sig)
    else:
        c1 = math.sqrt(2 / math.pi)
        c2 = 0.044715 * c1
        gate_sq = gate * gate
        t = torch.tanh(gate * (c1 + c2 * gate_sq))
        half = 0.5 + 0.5 * t
        act = gate * half
        dact = half + 0.5 * gate * (1 - t * t) * (c1 + 3 * c2 * gate_sq)
    return d * up * dact, d * act


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.promote_types(dtype, torch.float32)


class TorchOps:
    """Plain-torch versions of the kernel primitives the split path uses, same semantics.

    A varlen output keeps the kernels' contract that rows past the last expert segment (EP
    sentinel pairs) are never written; they are set to NaN here so any read of them shows.
    """

    @staticmethod
    def _segments(cu: torch.Tensor):
        bounds = cu.tolist()
        return list(enumerate(zip(bounds[:-1], bounds[1:], strict=True)))

    @staticmethod
    def _poison_tail(out: torch.Tensor, cu: torch.Tensor) -> None:
        out[int(cu[-1]) :] = float("nan")

    @staticmethod
    def _interleave(t: torch.Tensor, dim: int) -> torch.Tensor:
        n = t.shape[dim]
        return (
            t.unflatten(dim, (2, n // 2)).transpose(dim - 1, dim).flatten(dim - 1, dim)
        )

    def gemm(self, a, b, out, cu, a_idx=None, alpha=1.0, c=None, concat_b=False):
        if concat_b:
            b = self._interleave(b, -2)
        ct = _compute_dtype(out.dtype)
        for e, (lo, hi) in self._segments(cu):
            rows = a[a_idx[lo:hi].long()] if a_idx is not None else a[lo:hi]
            res = alpha * (rows.to(ct) @ b[e].to(ct))
            if c is not None:
                res = res + c[lo:hi].to(ct)
            out[lo:hi] = res.to(out.dtype)
        self._poison_tail(out, cu)
        return out

    def gemm_k(self, a_t, b, out, cu, a_idx=None, alpha=1.0):
        ct = _compute_dtype(out.dtype)
        for e, (lo, hi) in self._segments(cu):
            cols = a_t[:, a_idx[lo:hi].long()] if a_idx is not None else a_t[:, lo:hi]
            out[e] = (alpha * (cols.to(ct) @ b[lo:hi].to(ct))).to(out.dtype)
        return out

    def gemm_gated(self, x, b, c, preact, postact, cu, a_idx, activation, concat_b):
        if concat_b:
            b = self._interleave(b, -1)
        ct = _compute_dtype(preact.dtype)
        for e, (lo, hi) in self._segments(cu):
            pre = x[a_idx[lo:hi].long()].to(ct) @ b[e].to(ct)
            if c is not None:
                pre = pre + c[lo:hi].to(ct)
            preact[lo:hi] = pre.to(preact.dtype)
            postact[lo:hi] = glu_forward(pre[:, ::2], pre[:, 1::2], activation).to(
                postact.dtype
            )
        self._poison_tail(preact, cu)
        self._poison_tail(postact, cu)

    def gemm_dgated(
        self, dout, b, preact, dpreact, postact, scale, cu, a_idx, activation
    ):
        ct = _compute_dtype(preact.dtype)
        colvec = torch.empty(preact.shape[0], dtype=ct, device=dout.device)
        for e, (lo, hi) in self._segments(cu):
            d = dout[a_idx[lo:hi].long()].to(ct) @ b[e].to(ct)
            gate = preact[lo:hi, ::2].to(ct)
            up = preact[lo:hi, 1::2].to(ct)
            sigma = scale[lo:hi, None].to(ct)
            act = glu_forward(gate, up, activation)
            dgate, dup = glu_backward(gate, up, d * sigma, activation)
            dpreact[lo:hi] = (
                torch.stack([dgate, dup], dim=-1).flatten(-2).to(dpreact.dtype)
            )
            postact[lo:hi] = (act * sigma).to(postact.dtype)
            colvec[lo:hi] = (act * d).sum(-1)
        self._poison_tail(dpreact, cu)
        self._poison_tail(postact, cu)
        self._poison_tail(colvec, cu)
        return colvec

    @staticmethod
    def _slot_tokens(token_offset: torch.Tensor, num_slots: int) -> torch.Tensor:
        counts = token_offset.diff().long()
        tokens = torch.arange(counts.numel(), device=token_offset.device)
        return tokens.repeat_interleave(counts, output_size=num_slots)

    def router_forward(self, y, scores, rev, token_offset, num_tokens, num_experts):
        num_slots = rev.numel()
        valid = rev < num_slots
        tokens = self._slot_tokens(token_offset, num_slots)[valid]
        ct = _compute_dtype(y.dtype)
        rows = y[rev[valid].long()].to(ct) * scores[valid, None].to(ct)
        out = torch.zeros(num_tokens, y.shape[1], dtype=ct, device=y.device)
        return out.index_add_(0, tokens, rows).to(y.dtype)

    def token_reduce(self, dx_expanded, rev, token_offset, num_tokens, num_experts):
        num_slots = rev.numel()
        valid = rev < num_slots
        tokens = self._slot_tokens(token_offset, num_slots)[valid]
        ct = _compute_dtype(dx_expanded.dtype)
        out = torch.zeros(num_tokens, dx_expanded.shape[1], dtype=ct, device=rev.device)
        out.index_add_(0, tokens, dx_expanded[rev[valid].long()].to(ct))
        return out.to(dx_expanded.dtype)

    def gather_valid_rows(self, src, rev):
        valid = rev < rev.numel()
        return torch.where(valid, src[torch.where(valid, rev, 0).long()], 0.0)

    def glu_bwd_add_(self, dpreact, preact, d, cu, activation):
        n = int(cu[-1])
        ct = _compute_dtype(preact.dtype)
        gate = preact[:n, ::2].to(ct)
        up = preact[:n, 1::2].to(ct)
        dgate, dup = glu_backward(gate, up, d[:n].to(ct), activation)
        dpreact[:n, ::2] = (dpreact[:n, ::2].to(ct) + dgate).to(dpreact.dtype)
        dpreact[:n, 1::2] = (dpreact[:n, 1::2].to(ct) + dup).to(dpreact.dtype)

    def routing_metadata(self, token_indices, expert_indices, num_tokens, num_experts):
        device = expert_indices.device
        num_slots = expert_indices.numel()
        valid = expert_indices < num_experts
        key = torch.where(valid, expert_indices, num_experts).long()
        order = torch.sort(key, stable=True).indices
        n_valid = int(valid.sum())
        freq = torch.bincount(key[valid], minlength=num_experts)
        cu = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
        cu[1:] = torch.cumsum(freq, 0)
        s_scatter = torch.zeros(num_slots, dtype=torch.int32, device=device)
        s_scatter[:n_valid] = order[:n_valid].int()
        x_gather = torch.zeros(num_slots, dtype=torch.int32, device=device)
        x_gather[:n_valid] = token_indices[order[:n_valid]].int()
        rev = torch.full((num_slots,), num_slots, dtype=torch.int32, device=device)
        rev[order[:n_valid]] = torch.arange(n_valid, dtype=torch.int32, device=device)
        token_offset = torch.searchsorted(
            token_indices.long(), torch.arange(num_tokens + 1, device=device)
        ).int()
        return RoutingMetadata(cu, x_gather, s_scatter, rev, token_offset)


class SonicOps:
    """The split path's primitives on the loaded sonic-moe hub kernel and its vendored quack."""

    def __init__(self):
        from transformers.integrations.hub_kernels import lazy_load_kernel
        from transformers.integrations.sonicmoe import load_sonicmoe_kernel

        load_sonicmoe_kernel()
        root = lazy_load_kernel("sonic-moe").__name__
        self.gemm_interface = importlib.import_module(f"{root}.quack.gemm_interface")
        self.functional = importlib.import_module(f"{root}.functional")
        self.forward_ops = importlib.import_module(f"{root}.functional.forward")
        self.backward_ops = importlib.import_module(f"{root}.functional.backward")

    def gemm(self, a, b, out, cu, a_idx=None, alpha=1.0, c=None, concat_b=False):
        concat = ("B",) if concat_b else None
        if c is None:
            self.gemm_interface.gemm(
                a,
                b,
                out=out,
                alpha=float(alpha),
                cu_seqlens_m=cu,
                A_idx=a_idx,
                dynamic_scheduler=False,
                concat_layout=concat,
            )
        else:
            self.gemm_interface.gemm_add(
                a,
                b,
                C=c,
                out=out,
                alpha=float(alpha),
                cu_seqlens_m=cu,
                A_idx=a_idx,
                dynamic_scheduler=False,
                concat_layout=concat,
            )
        return out

    def gemm_k(self, a_t, b, out, cu, a_idx=None, alpha=1.0):
        self.gemm_interface.gemm(
            a_t,
            b,
            out=out,
            alpha=float(alpha),
            cu_seqlens_k=cu,
            A_idx=a_idx,
            batch_idx_permute=None,
            dynamic_scheduler=False,
        )
        return out

    def gemm_gated(self, x, b, c, preact, postact, cu, a_idx, activation, concat_b):
        self.gemm_interface.gemm_gated(
            x,
            b,
            C=c,
            activation=activation,
            cu_seqlens_m=cu,
            A_idx=a_idx,
            preact_out=preact,
            postact_out=postact,
            store_preact=True,
            concat_layout=("B",) if concat_b else None,
        )

    def gemm_dgated(
        self, dout, b, preact, dpreact, postact, scale, cu, a_idx, activation
    ):
        _, _, colvec = self.gemm_interface.gemm_dgated(
            dout,
            b,
            PreAct=preact,
            activation=activation,
            dx_out=dpreact,
            postact_out=postact,
            colvec_scale=scale,
            colvec_reduce=True,
            cu_seqlens_m=cu,
            A_idx=a_idx,
            dynamic_scheduler=False,
        )
        return colvec

    def router_forward(self, y, scores, rev, token_offset, num_tokens, num_experts):
        out = torch.empty(num_tokens, y.shape[1], dtype=y.dtype, device=y.device)
        self.forward_ops._router_forward(
            y=y,
            o=out,
            topk_scores=scores,
            s_reverse_scatter_idx=rev,
            num_activated_expert_per_token_offset=token_offset,
            varlen_K_max=num_experts,
            H=y.shape[1],
            is_varlen_K=True,
        )
        return out

    def token_reduce(self, dx_expanded, rev, token_offset, num_tokens, num_experts):
        out = torch.empty(
            num_tokens,
            dx_expanded.shape[1],
            dtype=dx_expanded.dtype,
            device=dx_expanded.device,
        )
        self.backward_ops._token_broadcast_backward(
            dx_reduced=out,
            dx_expanded=dx_expanded,
            s_reverse_scatter_idx=rev,
            num_activated_expert_per_token_offset=token_offset,
            varlen_K_max=num_experts,
            H=dx_expanded.shape[1],
            is_varlen_K=True,
        )
        return out

    def gather_valid_rows(self, src, rev):
        out = torch.empty(rev.numel(), dtype=src.dtype, device=src.device)
        self.backward_ops._gather_valid_rows(src, rev, out)
        return out

    def glu_bwd_add_(self, dpreact, preact, d, cu, activation):
        from .glu_bwd_triton import glu_bwd_add_

        glu_bwd_add_(dpreact, preact, d, cu, activation)

    def routing_metadata(self, token_indices, expert_indices, num_tokens, num_experts):
        device = expert_indices.device
        num_slots = expert_indices.numel()
        # zero-init like moe_general_routing_inputs: the kernel leaves sentinel/tail lanes alone
        s_scatter = torch.zeros(num_slots, dtype=torch.int32, device=device)
        rev = torch.zeros(num_slots, dtype=torch.int32, device=device)
        freq = torch.empty(num_experts, dtype=torch.int32, device=device)
        cu = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        x_gather = torch.zeros(num_slots, dtype=torch.int32, device=device)
        token_offset = torch.empty(num_tokens + 1, dtype=torch.int32, device=device)
        self.functional.general_routing_router_metadata_triton(
            token_indices,
            expert_indices,
            num_tokens,
            num_experts,
            freq,
            cu,
            x_gather,
            s_scatter,
            rev,
            token_offset,
        )
        return RoutingMetadata(cu, x_gather, s_scatter, rev, token_offset)


_SONIC_OPS: SonicOps | None = None


def sonic_ops() -> SonicOps:
    global _SONIC_OPS
    if _SONIC_OPS is None:
        _SONIC_OPS = SonicOps()
    return _SONIC_OPS


def _addend_buffer(out: torch.Tensor) -> torch.Tensor:
    # Writing C into the output buffer is safe for the kernels (each tile reads its C before
    # storing D there) but not something a traced graph may assume of a mutated input.
    return torch.empty_like(out) if torch.compiler.is_compiling() else out


class LoRASplitExperts(torch.autograd.Function):
    """Gated MoE experts on flat routing with expert LoRA as low-rank addends (base frozen).

    ``w1`` / ``w2`` are the quack B operands sonic-moe's forward uses, ``[E, H, 2I]`` and
    ``[E, I, H]``; ``p1, q1`` / ``p2, q2`` are from ``lora_split_factors`` (``q1`` in the gated
    kernels' interleaved column order) or ``None`` for a linear without LoRA.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        scores,
        p1,
        q1,
        p2,
        q2,
        w1,
        w2,
        cu,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        token_offset,
        scaling1: float,
        scaling2: float,
        activation: str,
        concat: bool,
        ops,
    ):
        num_tokens, hidden = x.shape
        num_slots = scores.numel()
        num_experts = cu.numel() - 1
        two_i = w1.shape[-1]
        kw = {"dtype": x.dtype, "device": x.device}

        h = torch.empty(num_slots, two_i, **kw)
        a = torch.empty(num_slots, two_i // 2, **kw)
        u = c1 = None
        if p1 is not None:
            u = ops.gemm(
                x, p1, torch.empty(num_slots, p1.shape[-1], **kw), cu, x_gather_idx
            )
            c1 = ops.gemm(u, q1, _addend_buffer(h), cu, alpha=scaling1)
        ops.gemm_gated(x, w1, c1, h, a, cu, x_gather_idx, activation, concat)

        y = torch.empty(num_slots, hidden, **kw)
        v = c2 = None
        if p2 is not None:
            v = ops.gemm(a, p2, torch.empty(num_slots, p2.shape[-1], **kw), cu)
            c2 = ops.gemm(v, q2, _addend_buffer(y), cu, alpha=scaling2)
        ops.gemm(a, w2, y, cu, c=c2)
        del a, c1, c2
        out = ops.router_forward(
            y, scores, s_reverse_scatter_idx, token_offset, num_tokens, num_experts
        )

        ctx.save_for_backward(
            x,
            h,
            u,
            v,
            scores,
            p1,
            q1,
            p2,
            q2,
            w1,
            w2,
            cu,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            token_offset,
        )
        ctx.scaling = (scaling1, scaling2)
        ctx.activation = activation
        ctx.concat = concat
        ctx.ops = ops
        return out

    @staticmethod
    def backward(ctx, dout):
        (
            x,
            h,
            u,
            v,
            scores,
            p1,
            q1,
            p2,
            q2,
            w1,
            w2,
            cu,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            token_offset,
        ) = ctx.saved_tensors
        ops = ctx.ops
        scaling1, scaling2 = ctx.scaling
        activation = ctx.activation
        dout = dout.contiguous()
        num_tokens, hidden = x.shape
        num_slots = scores.numel()
        num_experts = cu.numel() - 1
        inter = h.shape[1] // 2
        kw = {"dtype": x.dtype, "device": x.device}
        acc = _compute_dtype(x.dtype)
        sigma = scores[s_scatter_idx.long()]

        dh = torch.empty_like(h)
        a_prime = torch.empty(num_slots, inter, **kw)
        ds_grouped = ops.gemm_dgated(
            dout,
            w2.transpose(1, 2),
            h,
            dh,
            a_prime,
            sigma,
            cu,
            x_gather_idx,
            activation,
        )

        dp2 = dq2 = None
        if p2 is not None:
            r2 = p2.shape[-1]
            w = ops.gemm(
                dout,
                q2.transpose(1, 2),
                torch.empty(num_slots, r2, **kw),
                cu,
                x_gather_idx,
            )
            ds_grouped = ds_grouped + scaling2 * (w.to(acc) * v.to(acc)).sum(-1)
            v_scaled = (v.to(acc) * sigma[:, None]).to(v.dtype)
            dq2 = ops.gemm_k(
                dout.t(),
                v_scaled,
                torch.empty(num_experts, hidden, r2, **kw),
                cu,
                x_gather_idx,
                alpha=scaling2,
            ).transpose(1, 2)
            dp2 = ops.gemm_k(
                a_prime.t(),
                w,
                torch.empty(num_experts, inter, r2, **kw),
                cu,
                alpha=scaling2,
            )
            w_scaled = (w.to(acc) * (scaling2 * sigma)[:, None]).to(w.dtype)
            # the down-LoRA part of da reuses a' (consumed by dP2 above)
            d2 = ops.gemm(w_scaled, p2.transpose(1, 2), a_prime, cu)
            ops.glu_bwd_add_(dh, h, d2, cu, activation)
            del w, v_scaled, w_scaled, d2
        del a_prime
        dscores = ops.gather_valid_rows(ds_grouped, s_reverse_scatter_idx)

        dp1 = dq1 = None
        dx_expanded = torch.empty(num_slots, hidden, **kw)
        c = None
        if p1 is not None:
            r1 = p1.shape[-1]
            z1 = ops.gemm(dh, q1.transpose(1, 2), torch.empty(num_slots, r1, **kw), cu)
            dq1 = ops.gemm_k(
                u.t(),
                dh,
                torch.empty(num_experts, r1, h.shape[1], **kw),
                cu,
                alpha=scaling1,
            )
            dp1 = ops.gemm_k(
                x.t(),
                z1,
                torch.empty(num_experts, hidden, r1, **kw),
                cu,
                x_gather_idx,
                alpha=scaling1,
            )
            c = ops.gemm(
                z1, p1.transpose(1, 2), _addend_buffer(dx_expanded), cu, alpha=scaling1
            )
            del z1
        ops.gemm(dh, w1.transpose(1, 2), dx_expanded, cu, c=c, concat_b=ctx.concat)
        del dh, c
        dx = ops.token_reduce(
            dx_expanded, s_reverse_scatter_idx, token_offset, num_tokens, num_experts
        )
        return (dx, dscores, dp1, dq1, dp2, dq2) + (None,) * 12


def sonicmoe_lora_split_forward(
    facade,
    hidden_states: torch.Tensor,
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
    router_scores: torch.Tensor,
    lora_w1,
    lora_w2,
    ops=None,
) -> torch.Tensor:
    """The split path on flat routing (``moe_general_routing_inputs`` arguments).

    ``facade`` exposes the base weights in HF layout plus the layout flags and activation, as
    ``experts._LoRAExpertsFacade`` does; ``lora_w*`` are ``(lora_A, lora_B, scaling)`` or ``None``.
    """
    ops = ops if ops is not None else sonic_ops()
    num_experts = facade.num_experts
    transposed = facade.is_transposed
    concat = facade.is_concatenated
    activation = GLU_EPILOGUES[facade.config.hidden_act]
    w1 = facade.gate_up_proj if transposed else facade.gate_up_proj.transpose(1, 2)
    w2 = facade.down_proj if transposed else facade.down_proj.transpose(1, 2)
    dtype = hidden_states.dtype

    p1 = q1 = p2 = q2 = None
    scaling1 = scaling2 = 1.0
    if lora_w1 is not None:
        p1, q1 = lora_split_factors(
            lora_w1[0],
            lora_w1[1],
            num_experts,
            input_dim_last=not transposed,
            interleave=concat,
            dtype=dtype,
        )
        scaling1 = float(lora_w1[2])
    if lora_w2 is not None:
        p2, q2 = lora_split_factors(
            lora_w2[0],
            lora_w2[1],
            num_experts,
            input_dim_last=not transposed,
            interleave=False,
            dtype=dtype,
        )
        scaling2 = float(lora_w2[2])

    expert_indices = torch.where(
        (expert_indices >= 0) & (expert_indices < num_experts),
        expert_indices,
        num_experts,
    ).int()
    meta = ops.routing_metadata(
        token_indices.int(), expert_indices, hidden_states.shape[0], num_experts
    )
    return LoRASplitExperts.apply(
        hidden_states.contiguous(),
        router_scores.to(_compute_dtype(router_scores.dtype)),
        p1,
        q1,
        p2,
        q2,
        w1,
        w2,
        *meta,
        scaling1,
        scaling2,
        activation,
        concat,
        ops,
    )
