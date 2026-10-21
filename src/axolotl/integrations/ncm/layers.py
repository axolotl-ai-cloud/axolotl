"""Transformer building blocks shared by the NCM tokenizer and generator."""

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        rms = x_f.pow(2).mean(dim=-1, keepdim=True).add_(self.eps).rsqrt_()
        return (x_f * rms).to(x.dtype) * self.weight


def rope_cos_sin(
    positions: torch.Tensor, head_dim: int, base: float
) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, half, dtype=torch.float32, device=positions.device) / half)
    )
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos.to(x.dtype) + rotated * sin.to(x.dtype)


class SelfAttention(nn.Module):
    def __init__(
        self, n_embd: int, n_head: int, bias: bool, dropout: float, qk_norm: bool
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.dropout = dropout
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(n_embd // n_head)
            self.k_norm = RMSNorm(n_embd // n_head)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = (
            t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            for t in self.c_attn(x).split(C, dim=2)
        )
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = apply_rotary_emb(q, *rope), apply_rotary_emb(k, *rope)
        if attn_mask is not None:
            attn_mask = attn_mask.to(q.dtype)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd: int, bias: bool, dropout: float):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-LN transformer block; attention is bidirectional unless a mask is given."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        bias: bool = True,
        dropout: float = 0.0,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, eps=1e-6, bias=bias)
        self.attn = SelfAttention(n_embd, n_head, bias, dropout, qk_norm)
        self.ln_2 = nn.LayerNorm(n_embd, eps=1e-6, bias=bias)
        self.mlp = MLP(n_embd, bias, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attn_mask, rope)
        return x + self.mlp(self.ln_2(x))
