import os
import sys

import torch
import torch.nn.functional as F


def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    # Convert to python list for splitting
    num_tokens = num_tokens_per_expert.tolist()

    # Account for potential padding tokens if any
    num_padding = x.shape[0] - sum(num_tokens)

    # Split along expert groups and process each expert in a loop
    x_splits = torch.split(
        x[: sum(num_tokens)], split_size_or_sections=num_tokens, dim=0
    )
    out_splits = []
    for expert_idx, x_chunk in enumerate(x_splits):
        h = F.silu(torch.matmul(x_chunk, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_chunk, w3[expert_idx].transpose(-2, -1))
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        out_splits.append(h)
    out = torch.cat(out_splits, dim=0)

    if num_padding > 0:
        out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))

    return out


def _try_import_cg_grouped_gemm():
    """Try to import torchtitan's Triton contiguous grouped GEMM.

    Returns the callable `cg_grouped_gemm(inputs, expert_weights, expert_indices, group_size_m=128)`
    if importable from either an installed `torchtitan` package or a sibling checkout at
    ../torchtitan/torchtitan/experiments/kernels/triton_contiguous_group_gemm.
    """
    # Prefer vendored kernel if present
    try:
        from axolotl.kernels.vendor.tt_cg_gemm import cg_grouped_gemm  # type: ignore

        return cg_grouped_gemm
    except Exception:
        pass
    # Attempt import from an installed torchtitan package
    try:
        from torchtitan.experiments.kernels.triton_contiguous_group_gemm.cg_backward import (
            cg_grouped_gemm,  # type: ignore
        )

        return cg_grouped_gemm
    except Exception:
        pass

    # Attempt import from a sibling checkout: ../torchtitan/torchtitan/experiments/kernels/triton_contiguous_group_gemm
    try:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(this_dir, "../../../.."))
        tt_kernel_dir = os.path.abspath(
            os.path.join(
                repo_root,
                "../torchtitan/torchtitan/experiments/kernels/triton_contiguous_group_gemm",
            )
        )
        if os.path.isdir(tt_kernel_dir) and tt_kernel_dir not in sys.path:
            sys.path.insert(0, tt_kernel_dir)
        # The module does local imports (cg_forward), so we import by filename module name
        import importlib

        mod = importlib.import_module("cg_backward")
        if hasattr(mod, "cg_grouped_gemm"):
            return mod.cg_grouped_gemm
    except Exception:
        pass

    return None


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    # Prefer torchtitan Triton CG-GEMM if available; otherwise fallback to Python loop.
    assert x.dim() == 2

    cg_grouped_gemm = _try_import_cg_grouped_gemm()
    if cg_grouped_gemm is None:
        return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)

    # Build expert index vector corresponding to `x` rows. We rely on caller to have
    # grouped tokens by expert in ascending expert order; `num_tokens_per_expert` conveys sizes.
    num_experts = w1.shape[0]
    counts_i32 = num_tokens_per_expert.to(dtype=torch.int32, device=x.device)

    # CG-GEMM expects contiguous groups of fixed size (group_size_m). We'll pad each expert's
    # block up to the next multiple of `group_size_m` with zeros, run the kernel, then unpad.
    group_size_m = 128
    # total real tokens
    total_real = int(counts_i32.sum().item())
    # compute per-expert padded counts
    padded_counts = ((counts_i32 + (group_size_m - 1)) // group_size_m) * group_size_m

    # Create expert indices for padded layout
    expert_ids = torch.arange(num_experts, device=x.device, dtype=torch.int32)
    expert_indices = torch.repeat_interleave(expert_ids, padded_counts)

    # Pad inputs with zeros at the tail of each expert block
    x_padded_parts = []
    start = 0
    for e in range(num_experts):
        cnt = int(counts_i32[e].item())
        pad = int(padded_counts[e].item() - cnt)
        if cnt > 0:
            x_part = x[start : start + cnt]
            start += cnt
        else:
            x_part = x.new_zeros((0, x.shape[-1]))
        if pad > 0:
            x_part = torch.vstack((x_part, x.new_zeros((pad, x.shape[-1]))))
        x_padded_parts.append(x_part)
    x_padded = torch.cat(x_padded_parts, dim=0).contiguous()

    # Weights expected as [E, N, K]; inputs as [M_total, K]
    # w1,w3 are [E, hidden, dim]; w2 is [E, dim, hidden]
    h1 = F.silu(
        cg_grouped_gemm(x_padded, w1, expert_indices, group_size_m=group_size_m)
    )
    h3 = cg_grouped_gemm(x_padded, w3, expert_indices, group_size_m=group_size_m)
    h = h1 * h3
    out_padded = cg_grouped_gemm(
        h, w2, expert_indices, group_size_m=group_size_m
    ).type_as(x)
    # Drop padded rows
    out = out_padded[:total_real]
    return out


def _compute_routing(
    x: torch.Tensor,
    gate: torch.nn.Module,
    num_experts: int,
    top_k: int,
    *,
    score_func: str = "sigmoid",
    route_norm: bool = True,
    route_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute routing scores and selections directly from provided `gate` for correct gradients."""
    scores = gate(x)
    if score_func == "sigmoid":
        scores = torch.sigmoid(scores.to(torch.float32))
    elif score_func == "softmax":
        scores = F.softmax(scores.to(torch.float32), dim=1)
    else:
        raise NotImplementedError(f"Unknown score_func {score_func}")

    top_scores, top_idx = torch.topk(scores, k=top_k, dim=1)
    if score_func == "sigmoid" and route_norm:
        denom = top_scores.sum(dim=-1, keepdim=True) + 1e-20
        top_scores = top_scores / denom
    top_scores = top_scores * route_scale

    # Integer counts per expert for routing; prefer bincount over histc to avoid floats
    flat_idx = top_idx.reshape(-1)
    num_tokens_per_expert = torch.bincount(flat_idx, minlength=num_experts).to(
        dtype=torch.int32
    )
    return top_scores, top_idx, num_tokens_per_expert


def _extract_linear_from_router(router: torch.nn.Module) -> torch.nn.Module | None:
    """Try to find an inner Linear used by router modules (common names)."""
    for name in ("gate", "router", "proj", "linear"):
        inner = getattr(router, name, None)
        if isinstance(inner, torch.nn.Linear):
            return inner
    return None


def _router_forward_topk(
    x: torch.Tensor,
    gate: torch.nn.Module,
    num_experts: int,
    top_k: int,
    *,
    score_func: str = "sigmoid",
    route_norm: bool = True,
    route_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Attempt to get (top_scores, top_idx) directly from router forward.

    Supports routers that return:
      - logits tensor -> we compute topk
      - (top_scores, top_idx)
      - dict with keys like 'topk_scores'/'topk_indices' or 'scores'/'indices'
    Falls back to None if shapes aren't recognizable.
    """
    try:
        out = gate(x)
    except Exception:
        return None

    # Tensor logits -> compute
    if isinstance(out, torch.Tensor):
        scores = out
        if score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        else:
            return None
        top_scores, top_idx = torch.topk(scores, k=top_k, dim=1)
        if score_func == "sigmoid" and route_norm:
            denom = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denom
        top_scores = top_scores * route_scale
        return top_scores, top_idx

    # Tuple/list
    if isinstance(out, (tuple, list)):
        tensors = [t for t in out if isinstance(t, torch.Tensor)]
        if len(tensors) >= 2:
            a, b = tensors[0], tensors[1]
            # Identify which is indices
            if b.dtype in (torch.int32, torch.int64, torch.long) and b.dim() == 2:
                top_idx = b
                top_scores = a
            elif a.dtype in (torch.int32, torch.int64, torch.long) and a.dim() == 2:
                top_idx = a
                top_scores = b
            else:
                # Not obvious; try treating first as logits
                scores = tensors[0]
                if scores.dim() == 2 and scores.shape[1] in (num_experts,):
                    return _router_forward_topk(
                        x,
                        lambda y: scores,
                        num_experts,
                        top_k,
                        score_func=score_func,
                        route_norm=route_norm,
                        route_scale=route_scale,
                    )
                return None

            # If scores look like probabilities/logits for top-k (N, top_k), accept
            if top_scores.dim() == 2 and top_scores.shape[1] == top_k:
                return top_scores, top_idx
            # If scores are full logits (N, E), compute topk
            if top_scores.dim() == 2 and top_scores.shape[1] == num_experts:
                return _router_forward_topk(
                    x,
                    lambda y: top_scores,
                    num_experts,
                    top_k,
                    score_func=score_func,
                    route_norm=route_norm,
                    route_scale=route_scale,
                )

    # Dict
    if isinstance(out, dict):
        # common key patterns
        score_keys = [
            "topk_scores",
            "scores",
            "route_scores",
            "routing_scores",
            "dispatch_weights",
            "combine_weights",
        ]
        index_keys = [
            "topk_indices",
            "indices",
            "route_indices",
            "routing_indices",
            "expert_indices",
        ]
        s = next(
            (
                out[k]
                for k in score_keys
                if k in out and isinstance(out[k], torch.Tensor)
            ),
            None,
        )
        idx = next(
            (
                out[k]
                for k in index_keys
                if k in out and isinstance(out[k], torch.Tensor)
            ),
            None,
        )
        if s is not None and idx is not None:
            if s.dim() == 2 and idx.dim() == 2:
                return s, idx
        if s is not None and s.dim() == 2 and s.shape[1] in (top_k, num_experts):
            return _router_forward_topk(
                x,
                lambda y: s,
                num_experts,
                top_k,
                score_func=score_func,
                route_norm=route_norm,
                route_scale=route_scale,
            )

    return None


class TokenReorderer(torch.nn.Module):
    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self, top_scores: torch.Tensor, selected_experts_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # counts per expert as int32
        num_tokens_per_expert = torch.bincount(
            selected_experts_indices.reshape(-1), minlength=self.num_experts
        ).to(dtype=torch.int32)

        token_indices_sorted = torch.argsort(
            selected_experts_indices.reshape(-1), stable=True
        )
        top_scores_sorted = top_scores.view(-1)[token_indices_sorted]
        token_indices_sorted = token_indices_sorted // self.top_k

        return top_scores_sorted, token_indices_sorted, num_tokens_per_expert


def moe_forward_kernel(
    *,
    hidden_states: torch.Tensor,
    gate: torch.nn.Module,
    experts: list,
    shared_expert: object | None,
    top_k: int,
    score_func: str = "sigmoid",
    route_norm: bool = True,
    route_scale: float = 1.0,
) -> torch.Tensor:
    """Execute MoE forward using grouped-expert kernels on HF DeepSeek-V3 MLP-like module.

    Args:
        hidden_states: (bs, seqlen, dim)
        gate: nn.Linear mapping dim->num_experts
        experts: iterable of expert modules exposing gate_proj, up_proj, down_proj
        shared_expert: optional dense expert module with gate_proj/up_proj/down_proj
        top_k: number of experts per token
        score_func: 'sigmoid' or 'softmax'
        route_norm: normalize sigmoid top-k scores
        route_scale: scaling of routing scores
    """
    bs, slen, dim = hidden_states.shape
    x = hidden_states.view(-1, dim)

    num_experts = len(experts)

    # Compute routing using router if possible, else fall back to inner Linear or generic path
    routed = _router_forward_topk(
        x,
        gate,
        num_experts,
        top_k,
        score_func=score_func,
        route_norm=route_norm,
        route_scale=route_scale,
    )
    if routed is not None:
        top_scores, top_idx = routed
    else:
        inner = _extract_linear_from_router(gate)
        if inner is None:
            # Last resort: assume gate(x) returned logits-like tensor
            top_scores, top_idx = _router_forward_topk(
                x,
                gate,
                num_experts,
                top_k,
                score_func=score_func,
                route_norm=route_norm,
                route_scale=route_scale,
            ) or (None, None)
            if top_scores is None:
                raise RuntimeError("Unable to derive routing from router output format")
        else:
            top_scores, top_idx, _ = _compute_routing(
                x,
                inner,
                num_experts,
                top_k,
                score_func=score_func,
                route_norm=route_norm,
                route_scale=route_scale,
            )

    reorderer = TokenReorderer(num_experts=num_experts, top_k=top_k)
    top_scores_sorted, token_indices_sorted, num_tokens_per_expert = reorderer(
        top_scores, top_idx
    )

    # Gather routed inputs according to sorted token indices
    gather_index = token_indices_sorted.reshape(-1, 1).expand(-1, dim)
    routed_input = torch.gather(x, dim=0, index=gather_index)
    # Weight inputs by routing scores (score-before-experts policy)
    routed_input = (
        routed_input.to(torch.float32) * top_scores_sorted.reshape(-1, 1)
    ).to(x.dtype)

    # Collect expert weights and stack per tensor: w1, w2, w3
    # Shapes: w1: (E, hidden, dim); w3: (E, hidden, dim); w2: (E, dim, hidden)
    w1 = torch.stack([exp.gate_proj.weight for exp in experts], dim=0)
    w3 = torch.stack([exp.up_proj.weight for exp in experts], dim=0)
    w2 = torch.stack([exp.down_proj.weight for exp in experts], dim=0)

    # Compute expert outputs via grouped mm
    routed_output = _run_experts_grouped_mm(
        w1, w2, w3, routed_input, num_tokens_per_expert
    )

    # Add shared expert path if present
    if shared_expert is not None:
        try:
            out3 = shared_expert(hidden_states)
            out = out3.view(bs * slen, dim)
        except Exception:
            out = shared_expert(hidden_states.view(-1, dim)).view(bs * slen, dim)
    else:
        out = torch.zeros_like(x)

    # Scatter back to original token positions and reshape
    out = out.scatter_add(dim=0, index=gather_index, src=routed_output)
    return out.view(bs, slen, dim)
