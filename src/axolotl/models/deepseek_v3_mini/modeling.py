from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig


class DeepseekV3MiniConfig(PretrainedConfig):
    model_type = "deepseek_v3"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 1024,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        max_position_embeddings: int = 4096,
        dropout: float = 0.0,
        # MoE
        num_experts: int = 8,
        top_k: int = 2,
        num_shared_experts: int = 0,
        router_score_fn: str = "sigmoid",
        route_norm: bool = True,
        route_scale: float = 1.0,
        # Misc
        tie_word_embeddings: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts
        self.router_score_fn = router_score_fn
        self.route_norm = route_norm
        self.route_scale = route_scale
        self.tie_word_embeddings = tie_word_embeddings


class ExpertMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # names chosen to match our patcher expectations
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gate_proj(x))
        h = h * self.up_proj(x)
        return self.down_proj(h)


class DeepseekV3MiniMoEMLP(nn.Module):
    """MoE MLP with attributes the patcher looks for: gate (Linear) + experts list.

    The actual routing/compute will be monkeypatched by our DeepSeek‑V3 kernel.
    """

    def __init__(self, dim: int, hidden_dim: int, cfg: DeepseekV3MiniConfig):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.top_k = cfg.top_k
        self.num_experts_per_tok = cfg.top_k  # alias used by some code
        self.router_score_fn = cfg.router_score_fn
        self.route_norm = cfg.route_norm
        self.route_scale = cfg.route_scale

        self.gate = nn.Linear(dim, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [ExpertMLP(dim, hidden_dim) for _ in range(cfg.num_experts)]
        )
        self.shared_experts: Optional[nn.Module] = None
        if cfg.num_shared_experts > 0:
            # simple dense FFN as shared expert(s); combine count via widened hidden_dim
            self.shared_experts = ExpertMLP(dim, hidden_dim * cfg.num_shared_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unpatched forward: simple dense average of top-k experts for safety.
        # At runtime, our patch will replace this with Triton CG‑GEMM kernel.
        bsz, seqlen, dim = x.shape
        flat = x.view(-1, dim)
        logits = self.gate(flat)
        scores = (
            torch.sigmoid(logits)
            if self.router_score_fn == "sigmoid"
            else F.softmax(logits, 1)
        )
        top_val, top_idx = torch.topk(scores, k=self.top_k, dim=1)
        top_val = top_val / (top_val.sum(-1, keepdim=True) + 1e-9)
        out = torch.zeros_like(flat)
        for k in range(self.top_k):
            idx = top_idx[:, k]
            val = top_val[:, k : k + 1]
            parts = []
            for e in range(len(self.experts)):
                mask = idx == e
                if mask.any():
                    parts.append((mask, self.experts[e](flat[mask]) * val[mask]))
            if parts:
                for mask, contrib in parts:
                    out[mask] += contrib
        if self.shared_experts is not None:
            out = out + self.shared_experts(flat)
        return out.view(bsz, seqlen, dim)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        h = self.num_heads
        q = self.q_proj(x).view(b, s, h, -1).transpose(1, 2)
        k = self.k_proj(x).view(b, s, h, -1).transpose(1, 2)
        v = self.v_proj(x).view(b, s, h, -1).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.o_proj(out)


class Block(nn.Module):
    def __init__(self, cfg: DeepseekV3MiniConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = Attention(cfg.hidden_size, cfg.num_attention_heads, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.hidden_size)
        self.mlp = DeepseekV3MiniMoEMLP(cfg.hidden_size, cfg.intermediate_size, cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DeepseekV3MiniModel(PreTrainedModel):
    config_class = DeepseekV3MiniConfig

    def __init__(self, config: DeepseekV3MiniConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [Block(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor]:
        bsz, seqlen = input_ids.shape
        device = input_ids.device
        pos = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
        x = self.embed_tokens(input_ids) + self.embed_positions(pos)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return (x,)


class DeepseekV3MiniForCausalLM(PreTrainedModel):
    config_class = DeepseekV3MiniConfig

    def __init__(self, config: DeepseekV3MiniConfig):
        super().__init__(config)
        self.model = DeepseekV3MiniModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        (hidden_states,) = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        return {"logits": logits, "loss": loss}
