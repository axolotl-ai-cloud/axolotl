import torch
import torch.nn as nn
from types import SimpleNamespace

from axolotl.monkeypatch.models.bailing_moe_v2.modeling import (
    BailingMoeV2GroupedExperts,
)


class DummyExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.moe_intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.moe_intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.moe_intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(hidden)


def test_grouped_experts_matches_python_loop():
    torch.manual_seed(0)

    config = SimpleNamespace(
        hidden_size=16,
        moe_intermediate_size=32,
        hidden_act="silu",
        num_experts=4,
        num_experts_per_tok=2,
    )

    experts = nn.ModuleList([DummyExpert(config) for _ in range(config.num_experts)])
    grouped = BailingMoeV2GroupedExperts(config, experts, backend_impl="grouped")

    batch, seq = 2, 3
    hidden_states = torch.randn(batch, seq, config.hidden_size)

    topk_idx = torch.stack(
        [
            torch.randperm(config.num_experts)[: config.num_experts_per_tok]
            for _ in range(batch * seq)
        ],
        dim=0,
    )
    topk_weight = torch.rand(batch * seq, config.num_experts_per_tok)
    topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
    topk_weight = topk_weight.to(hidden_states.dtype)

    # Baseline Python loop (mirrors original forward implementation)
    hidden_flat = hidden_states.view(-1, config.hidden_size)
    repeated_hidden = hidden_flat.repeat_interleave(config.num_experts_per_tok, dim=0)
    flat_idx = topk_idx.view(-1)

    loop_outputs = torch.zeros_like(repeated_hidden)
    for expert_id, expert in enumerate(experts):
        mask = flat_idx == expert_id
        if mask.any():
            loop_outputs[mask] = expert(repeated_hidden[mask])

    loop_outputs = loop_outputs.view(-1, config.num_experts_per_tok, config.hidden_size)
    weighted = loop_outputs * topk_weight.unsqueeze(-1)
    loop_result = weighted.sum(dim=1).view(batch, seq, config.hidden_size)

    grouped_result = grouped(hidden_states, topk_idx, topk_weight)

    torch.testing.assert_close(grouped_result, loop_result, atol=1e-5, rtol=1e-5)
