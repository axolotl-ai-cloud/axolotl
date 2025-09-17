"""
PyTorch 2.8+ grouped GEMM MoE path (cuBLASLt-backed).
This is a cautious first pass that probes available ops and only runs when supported.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def available() -> bool:
    try:
        ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        if ver < (2, 8):
            return False
        # Require Hopper+ (SM90) per torch error message and check op presence
        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability()
        if major < 9:
            return False
        return hasattr(torch.ops, "aten") and hasattr(torch.ops.aten, "_grouped_mm")
    except Exception:
        return False


LAST_ERROR: Optional[str] = None
_LOGGER = logging.getLogger("axolotl.moe.grouped")


def _is_mixtral_layout(mod: torch.nn.Module) -> bool:
    return all(hasattr(mod, attr) for attr in ("w1", "w3", "w2"))


def _is_qwen_layout(mod: torch.nn.Module) -> bool:
    has_fused = hasattr(mod, "gate_up_proj")
    has_split = hasattr(mod, "up_proj") and hasattr(mod, "gate_proj")
    return (has_fused or has_split) and hasattr(mod, "down_proj")


def _call_grouped_mm(
    As: List[torch.Tensor], Bs: List[torch.Tensor]
) -> Optional[List[torch.Tensor]]:
    """
    Call grouped mm using aten._grouped_mm with packed representation.

    - A_cat: concat As along rows -> [sum_i Mi, K]
    - B_stk: stack Bs per group -> [G, K, N]
    - offs: lengths per group Mi -> [G] int32
    Returns list of per-group outputs split from concatenated result.
    """
    global LAST_ERROR
    try:
        # Ensure 2D contiguous inputs
        As2 = [a.contiguous().view(a.shape[0], a.shape[1]) for a in As]
        Bs2 = [b.contiguous().view(b.shape[0], b.shape[1]) for b in Bs]

        if not As2:
            return []
        device = As2[0].device
        A_cat = torch.cat(As2, dim=0)
        B_stk = torch.stack(Bs2, dim=0)
        offs = torch.tensor([a.shape[0] for a in As2], device=device, dtype=torch.int32)

        if hasattr(torch.ops.aten, "_grouped_mm"):
            try:
                Y_cat = torch.ops.aten._grouped_mm(A_cat, B_stk, offs)  # type: ignore[attr-defined]
                outs: List[torch.Tensor] = []
                start = 0
                for m in offs.tolist():
                    outs.append(Y_cat[start : start + m, :])
                    start += m
                return outs
            except Exception as e:
                LAST_ERROR = f"_grouped_mm failed: {e}"
                return None
        LAST_ERROR = "aten._grouped_mm not present"
        return None
    except Exception as e:
        LAST_ERROR = f"call error: {e}"
        return None


def moe_ffn_forward_grouped(
    hidden_states, gate_linear, experts_module, top_k: int
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Attempt grouped GEMM fast path using PyTorch 2.8+."""
    global LAST_ERROR
    LAST_ERROR = None
    bsz, seqlen, hdim = hidden_states.shape
    x = hidden_states.view(-1, hdim)
    router_logits = gate_linear(x)

    # top-k routing executed in torch to avoid extra dependencies
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    topk_weight, topk_idx = torch.topk(routing_weights, top_k, dim=-1, sorted=False)
    topk_weight = (topk_weight / topk_weight.sum(dim=-1, keepdim=True)).to(x.dtype)

    flat_idx = topk_idx.view(-1)
    x_rep = x.repeat_interleave(top_k, dim=0)

    E = experts_module.num_experts
    dev, dt = x.device, x.dtype
    first = experts_module[0]

    is_mixtral = _is_mixtral_layout(first)
    is_qwen2 = _is_qwen_layout(first)
    nested_attr: Optional[str] = None
    if not (is_mixtral or is_qwen2):
        for candidate in ("mlp", "ffn"):
            nested = getattr(first, candidate, None)
            if nested is None:
                continue
            if _is_mixtral_layout(nested):
                is_mixtral = True
                nested_attr = candidate
                break
            if _is_qwen_layout(nested):
                is_qwen2 = True
                nested_attr = candidate
                break
    if not (is_mixtral or is_qwen2):
        if not getattr(experts_module, "_ax_grouped_logged_fail", False):
            _LOGGER.warning(
                "torch_grouped: unsupported expert layout; falling back to naive"
            )
            experts_module._ax_grouped_logged_fail = True
        LAST_ERROR = "unsupported expert layout"
        return None, None

    def _resolve_expert(idx: int):
        expert = experts_module[idx]
        if nested_attr is None:
            return expert
        nested_mod = getattr(expert, nested_attr, None)
        if nested_mod is None:
            raise AttributeError(f"expert {idx} missing nested module '{nested_attr}'")
        return nested_mod

    try:
        if is_mixtral:
            if (
                not hasattr(experts_module, "_stacked_w1")
                or experts_module._stacked_w1.device != dev
                or experts_module._stacked_w1.dtype != dt
            ):
                mods = [_resolve_expert(i) for i in range(E)]
                w1 = [mod.w1.weight.t() for mod in mods]
                w3 = [mod.w3.weight.t() for mod in mods]
                w2 = [mod.w2.weight.t() for mod in mods]
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
        else:
            if (
                not hasattr(experts_module, "_stacked_w13")
                or experts_module._stacked_w13.device != dev
                or experts_module._stacked_w13.dtype != dt
            ):
                w13 = []
                w2 = []
                for i in range(E):
                    mod = _resolve_expert(i)
                    if hasattr(mod, "gate_up_proj"):
                        w13.append(mod.gate_up_proj.weight.t())
                    elif hasattr(mod, "up_proj") and hasattr(mod, "gate_proj"):
                        w13.append(
                            torch.cat(
                                [mod.up_proj.weight.t(), mod.gate_proj.weight.t()],
                                dim=-1,
                            )
                        )
                    else:
                        LAST_ERROR = "unrecognized Qwen MoE expert weight layout"
                        if not getattr(
                            experts_module, "_ax_grouped_logged_fail", False
                        ):
                            _LOGGER.warning(
                                "torch_grouped: could not resolve Qwen MoE expert weights; fallback to naive"
                            )
                            experts_module._ax_grouped_logged_fail = True
                        return None, None
                    w2.append(mod.down_proj.weight.t())
                experts_module._stacked_w13 = (
                    torch.stack(w13, dim=0)
                    .to(device=dev, dtype=dt, non_blocking=True)
                    .contiguous()
                )
                experts_module._stacked_w2 = (
                    torch.stack(w2, dim=0)
                    .to(device=dev, dtype=dt, non_blocking=True)
                    .contiguous()
                )
            W13 = experts_module._stacked_w13
            W2 = experts_module._stacked_w2
    except AttributeError as err:
        LAST_ERROR = str(err)
        if not getattr(experts_module, "_ax_grouped_logged_fail", False):
            _LOGGER.warning(
                "torch_grouped: expert weights missing expected attributes; falling back to naive"
            )
            experts_module._ax_grouped_logged_fail = True
        return None, None

    As: List[torch.Tensor] = []
    Bs: List[torch.Tensor] = []
    expert_slices: List[Tuple[int, torch.Tensor]] = []
    for i in range(E):
        sel = flat_idx == i
        if sel.any():
            Xi = x_rep[sel]
            As.append(Xi)
            Bs.append(W13[i])
            expert_slices.append((i, sel))

    if not As:
        out = torch.zeros_like(x)
        return out.view(bsz, seqlen, hdim), router_logits

    Y_list = _call_grouped_mm(As, Bs)
    if Y_list is None:
        if not getattr(experts_module, "_ax_grouped_logged_fail", False):
            _LOGGER.warning(
                f"torch_grouped: grouped_mm up+gate failed; falling back to naive. Reason: {LAST_ERROR}"
            )
            experts_module._ax_grouped_logged_fail = True
        return None, None

    As2: List[torch.Tensor] = []
    Bs2: List[torch.Tensor] = []
    y_buf = torch.empty_like(x_rep)
    for (i, _sel), Yi in zip(expert_slices, Y_list, strict=False):
        I2 = Yi.shape[-1] // 2
        Yi_hidden = F.silu(Yi[:, :I2]) * Yi[:, I2:]
        As2.append(Yi_hidden)
        Bs2.append(W2[i])

    Y2_list = _call_grouped_mm(As2, Bs2)
    if Y2_list is None:
        if not getattr(experts_module, "_ax_grouped_logged_fail", False):
            _LOGGER.warning(
                f"torch_grouped: grouped_mm down failed; falling back to naive. Reason: {LAST_ERROR}"
            )
            experts_module._ax_grouped_logged_fail = True
        return None, None

    for (_i, sel), Out_i in zip(expert_slices, Y2_list, strict=False):
        y_buf[sel] = Out_i
    y = (y_buf.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
    if not getattr(experts_module, "_ax_grouped_logged_ok", False):
        _LOGGER.info(
            f"torch_grouped: engaged grouped GEMM path (experts={E}, top_k={top_k})"
        )
        experts_module._ax_grouped_logged_ok = True
    return y.view(bsz, seqlen, hdim), router_logits
