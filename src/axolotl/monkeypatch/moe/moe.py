import torch
import torch.nn as nn
import torch.nn.functional as F
from axolotl.monkeypatch.moe.mlp import FusedExperts

class SparseMoeBlock(nn.Module):
    def __init__(self, experts, hidden_dim, ffn_dim, num_experts, top_k):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts: FusedExperts = experts

    def _post_training(self, model, name):
        # get original weights back: reverse the concat + stack in the fused experts
        w1s, w3s = torch.split(torch.unbind(self.experts.experts.weight, dim=0), 2, dim=1)
        w2s = torch.unbind(self.experts.output_experts.weight, dim=0)

        # TODO: recreate MoE class with original weights
        experts = []
        for i in range(self.num_experts):
            pass

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Fused expert forward
        final_hidden_states = self.experts(hidden_states, routing_weights, selected_experts)

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
