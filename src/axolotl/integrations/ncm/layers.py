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


class Segments:
    """Prefix-LM over level blocks: the prefix attends to itself, each block to the
    prefix and itself. One mask-free SDPA call per block skips the masked-out work
    and lets SDPA pick flash/cuDNN kernels."""

    def __init__(self, prefix_len: int, sizes: list[int], prefix_keep=None):
        self.prefix_len = prefix_len
        self.sizes = list(sizes)
        # (B, P) bool, False at padded prefix tokens.
        self.prefix_keep = prefix_keep

    def _mask(self, n: int) -> torch.Tensor | None:
        if self.prefix_keep is None:
            return None
        return F.pad(self.prefix_keep, (0, n), value=True)[:, None, None, :]

    def attend(self, q, k, v, dropout_p: float) -> torch.Tensor:
        P = self.prefix_len
        qs, ks, vs = (t.split([P, *self.sizes], dim=2) for t in (q, k, v))
        out = [
            F.scaled_dot_product_attention(
                qs[0], ks[0], vs[0], attn_mask=self._mask(0), dropout_p=dropout_p
            )
        ]
        for i, n in enumerate(self.sizes, start=1):
            out.append(
                F.scaled_dot_product_attention(
                    qs[i],
                    torch.cat([ks[0], ks[i]], dim=2),
                    torch.cat([vs[0], vs[i]], dim=2),
                    attn_mask=self._mask(n),
                    dropout_p=dropout_p,
                )
            )
        return torch.cat(out, dim=2)


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
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        segments: Segments | None = None,
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
        p = self.dropout if self.training else 0.0
        if segments is None:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        else:
            y = segments.attend(q, k, v, p)
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
    """Pre-LN transformer block; full bidirectional attention unless segmented."""

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
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        segments: Segments | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), rope, segments)
        return x + self.mlp(self.ln_2(x))
