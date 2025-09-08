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


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    # grouped mm between a 2D tensor and a 3D tensor
    assert x.dim() == 2

    # Offsets are cumsum of token counts per expert
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    # Use bfloat16 for grouped mm to match torch internal API contracts
    try:
        h = F.silu(
            torch._grouped_mm(
                x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets
            )
        )
        h = h * torch._grouped_mm(
            x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
        )
        out = torch._grouped_mm(
            h, w2.bfloat16().transpose(-2, -1), offs=offsets
        ).type_as(x)
    except AttributeError:
        # Fallback if torch._grouped_mm is unavailable
        out = _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)
    return out


def _compute_routing(
    x: torch.Tensor,
    gate: torch.nn.Linear,
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

    num_tokens_per_expert = torch.histc(
        top_idx.view(-1), bins=num_experts, min=0, max=num_experts
    )
    return top_scores, top_idx, num_tokens_per_expert


class TokenReorderer(torch.nn.Module):
    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self, top_scores: torch.Tensor, selected_experts_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        token_indices_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )
        top_scores_sorted = top_scores.view(-1)[token_indices_sorted]
        token_indices_sorted = token_indices_sorted // self.top_k

        return top_scores_sorted, token_indices_sorted, num_tokens_per_expert


def moe_forward_kernel(
    *,
    hidden_states: torch.Tensor,
    gate: torch.nn.Linear,
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

    # Compute routing directly from `gate` so gradients flow to gate.weight
    top_scores, top_idx, _ = _compute_routing(
        x,
        gate,
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
