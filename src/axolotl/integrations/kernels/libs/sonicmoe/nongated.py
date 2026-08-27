# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI

"""Non-gated (relu²) grouped experts forward for the sonicmoe backend.

The sonic-moe op layer currently asserts gated epilogues (swiglu/geglu) even
though quack ships reglu/relu_sq, so non-gated experts (nemotron_h) route here:
a single-launch ``torch._grouped_mm`` MLP with a hand-written backward
(``_grouped_mm`` has no usable autograd). Differentiable to ``x``, ``w1``,
``w2`` — full-param training gets real weight grads, and LoRA composes by
passing the ``MoELoRAMaterialize``-built ``W_eff`` in as the weights. The
per-expert ``F.linear`` loop keeps CPU/float64 correctness for tests.

``AXOLOTL_SONICMOE_NONGATED_FUSED=1`` instead takes the REGLU-duplication path
through the CUTLASS kernel (``relu²(h) == relu(h)·h``); it requires a
sonic-moe build whose op layer allows the reglu epilogue.
"""

from __future__ import annotations

import torch


def _use_gmm(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and hasattr(torch, "_grouped_mm")
        and torch.cuda.get_device_capability(x.device)[0] >= 9
    )


def _seg_linear(x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Per-expert ``x_e @ w[e].T`` loop fallback; ``offs`` = [E] segment ends."""
    outs, start = [], 0
    for e, end in enumerate(offs.tolist()):
        outs.append(torch.nn.functional.linear(x[start:end], w[e]))
        start = end
    return torch.cat(outs, dim=0)


def _seg_outer(g: torch.Tensor, x: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Per-expert ``g_e^T @ x_e`` loop fallback → [E, dim_g, dim_x]."""
    E = offs.numel()
    out = g.new_zeros(E, g.shape[-1], x.shape[-1])
    start = 0
    for e, end in enumerate(offs.tolist()):
        if end > start:
            out[e] = g[start:end].transpose(0, 1) @ x[start:end]
        start = end
    return out


class _NonGatedGroupedMLP(torch.autograd.Function):
    """``y = act(x_g @ w1^T) @ w2^T`` per expert group; act = relu².

    ``x_g`` [T, L] expert-sorted, ``w1`` [E, I, L], ``w2`` [E, L, I],
    ``offs`` [E] int32 segment ends. Grads to ``x_g``, ``w1``, ``w2``.
    """

    @staticmethod
    def forward(ctx, x_g, w1, w2, offs):
        if _use_gmm(x_g):
            h = torch._grouped_mm(x_g, w1.transpose(-2, -1), offs=offs)
        else:
            h = _seg_linear(x_g, w1, offs)
        a = torch.relu(h).square()
        if _use_gmm(x_g):
            y = torch._grouped_mm(a, w2.transpose(-2, -1), offs=offs)
        else:
            y = _seg_linear(a, w2, offs)
        ctx.save_for_backward(x_g, w1, w2, offs, h)
        return y

    @staticmethod
    def backward(ctx, g):
        x_g, w1, w2, offs, h = ctx.saved_tensors
        g = g.contiguous()
        a = torch.relu(h).square()
        gmm = _use_gmm(g)

        if gmm:
            da = torch._grouped_mm(g, w2, offs=offs)
        else:
            start, parts = 0, []
            for e, end in enumerate(offs.tolist()):
                parts.append(g[start:end] @ w2[e])
                start = end
            da = torch.cat(parts, dim=0)
        dh = da * (2.0 * torch.relu(h))

        dw1 = dw2 = None
        if ctx.needs_input_grad[1]:
            dw1 = (
                torch._grouped_mm(dh.transpose(0, 1), x_g, offs=offs)
                if gmm
                else _seg_outer(dh, x_g, offs)
            )
        if ctx.needs_input_grad[2]:
            dw2 = (
                torch._grouped_mm(g.transpose(0, 1), a, offs=offs)
                if gmm
                else _seg_outer(g, a, offs)
            )

        if gmm:
            dx = torch._grouped_mm(dh, w1, offs=offs)
        else:
            start, parts = 0, []
            for e, end in enumerate(offs.tolist()):
                parts.append(dh[start:end] @ w1[e])
                start = end
            dx = torch.cat(parts, dim=0)
        return dx, dw1, dw2, None


def sonicmoe_nongated_forward(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Route → grouped non-gated relu² MLP → weighted combine.

    ``w1`` / ``w2`` may be ``MoELoRAMaterialize`` outputs (LoRA) or the raw
    params (full-param); both receive gradients through the Function.
    """
    from .nvfp4_lora import combine_expert_outputs, route_and_group

    x_g, expert_offsets, gather_idx, w_g = route_and_group(
        hidden_states, top_k_index, top_k_weights, num_experts
    )
    offs = expert_offsets[1:].to(device=x_g.device, dtype=torch.int32)
    y_g = _NonGatedGroupedMLP.apply(x_g, w1.contiguous(), w2.contiguous(), offs)
    return combine_expert_outputs(y_g, gather_idx, w_g, hidden_states.shape[0])
