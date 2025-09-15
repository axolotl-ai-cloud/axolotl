"""
PyTorch 2.8+ grouped GEMM MoE path (cuBLASLt-backed).
This is a cautious first pass that probes available ops and only runs when supported.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def available() -> bool:
    try:
        ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        if ver < (2, 8):
            return False
        # Check for aten grouped mm ops
        return hasattr(torch.ops, "aten") and (
            hasattr(torch.ops.aten, "_grouped_mm")
            or hasattr(torch.ops.aten, "_scaled_grouped_mm")
        )
    except Exception:
        return False


LAST_ERROR: Optional[str] = None


def _call_grouped_mm(
    As: List[torch.Tensor], Bs: List[torch.Tensor]
) -> Optional[List[torch.Tensor]]:
    """
    Try calling the appropriate grouped mm op available in this torch build.
    Returns list of outputs or None on failure.
    """
    global LAST_ERROR
    try:
        # Ensure 2D contiguous inputs
        As = [a.contiguous().view(a.shape[0], a.shape[1]) for a in As]
        Bs = [b.contiguous().view(b.shape[0], b.shape[1]) for b in Bs]

        if hasattr(torch.ops.aten, "_grouped_mm"):
            try:
                # Some builds expect tuples rather than lists
                return torch.ops.aten._grouped_mm(tuple(As), tuple(Bs))  # type: ignore[attr-defined]
            except Exception as e:
                LAST_ERROR = f"_grouped_mm failed: {e}"
        # Avoid _scaled_grouped_mm for now; its signature requires packed inputs.
    except Exception as e:
        LAST_ERROR = f"call error: {e}"
        return None
    return None


def moe_ffn_forward_grouped(
    hidden_states, gate_linear, experts_module, top_k: int
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Attempt a grouped GEMM fast path using PyTorch 2.8+.
    If unavailable or fails, returns (None, None) so caller can fallback.
    """
    try:
        bsz, seqlen, hdim = hidden_states.shape
        x = hidden_states.view(-1, hdim)
        router_logits = gate_linear(x)

        # topk routing in torch (keep simple to avoid dependency cycles)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        topk_weight, topk_idx = torch.topk(routing_weights, top_k, dim=-1, sorted=False)
        topk_weight = (topk_weight / topk_weight.sum(dim=-1, keepdim=True)).to(x.dtype)

        # Build per-expert input lists
        flat_idx = topk_idx.view(-1)
        x_rep = x.repeat_interleave(top_k, dim=0)

        # Cache stacked weights on experts
        E = experts_module.num_experts
        dev, dt = x.device, x.dtype
        if (
            not hasattr(experts_module, "_stacked_w1")
            or experts_module._stacked_w1.device != dev
            or experts_module._stacked_w1.dtype != dt
        ):
            w1 = [experts_module[i].w1.weight.t() for i in range(E)]
            w3 = [experts_module[i].w3.weight.t() for i in range(E)]
            w2 = [experts_module[i].w2.weight.t() for i in range(E)]
            experts_module._stacked_w1 = (
                torch.stack(w1, dim=0)
                .to(device=dev, dtype=dt, non_blocking=True)
                .contiguous()
            )
            experts_module._stacked_w3 = (
                torch.stack(w3, dim=0)
                .to(device=dev, dtype=dt, non_blocking=True)
                .contiguous()
            )
            experts_module._stacked_w2 = (
                torch.stack(w2, dim=0)
                .to(device=dev, dtype=dt, non_blocking=True)
                .contiguous()
            )
            experts_module._stacked_w13 = torch.cat(
                [experts_module._stacked_w1, experts_module._stacked_w3], dim=-1
            ).contiguous()
        W13 = experts_module._stacked_w13
        W2 = experts_module._stacked_w2

        # Grouped GEMM for up+gate
        As: List[torch.Tensor] = []
        Bs: List[torch.Tensor] = []
        expert_slices = []
        for i in range(E):
            sel = flat_idx == i
            if sel.any():
                Xi = x_rep[sel]
                As.append(Xi)
                Bs.append(W13[i])
                expert_slices.append((i, sel))

        if not As:
            # no tokens routed — edge case
            out = torch.zeros_like(x)
            return out.view(bsz, seqlen, hdim), router_logits

        Y_list = _call_grouped_mm(As, Bs)
        if Y_list is None:
            return None, None

        # SwiGLU on each expert block and prepare for down projection
        As2: List[torch.Tensor] = []
        Bs2: List[torch.Tensor] = []
        y_buf = torch.empty_like(x_rep)
        # split Y into (I, I)
        for (i, sel), Yi in zip(expert_slices, Y_list):
            I2 = Yi.shape[-1] // 2
            Yi_hidden = F.silu(Yi[:, :I2]) * Yi[:, I2:]
            As2.append(Yi_hidden)
            Bs2.append(W2[i])

        Y2_list = _call_grouped_mm(As2, Bs2)
        if Y2_list is None:
            return None, None

        # Write back, apply per-token weighting, and reduce over top_k
        for (i, sel), Out_i in zip(expert_slices, Y2_list):
            y_buf[sel] = Out_i
        y = (y_buf.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        return y.view(bsz, seqlen, hdim), router_logits
    except Exception:
        return None, None
