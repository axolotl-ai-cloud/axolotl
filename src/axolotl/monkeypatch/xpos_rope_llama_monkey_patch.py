# pylint: skip-file
"""
Copied from https://github.com/kaiokendev/cutoff-len-is-context-len/blob/main/util/xpos_rope_llama_monkey_patch.py
"""
import torch
import transformers
import transformers.models.llama.modeling_llama
from einops import rearrange


class XposRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scale_base=2048,
        use_xpos=True,
    ):
        super().__init__()
        self.max_seq_len_cached = max_position_embeddings
        self.scale_base = scale_base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(self.max_seq_len_cached, device=device).type_as(inv_freq)
        freqs = torch.einsum("i , j -> i j", t, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("freqs_cached", freqs, persistent=False)

        if not use_xpos:
            self.register_buffer("scale", None)
            self.register_buffer("scale_cached", torch.ones(1))
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        power = (t - (self.max_seq_len_cached // 2)) / self.scale_base
        scale_cached = scale ** rearrange(power, "n -> n 1")
        scale_cached = torch.cat((scale_cached, scale_cached), dim=-1)

        self.register_buffer("scale", scale, persistent=False)
        self.register_buffer("scale_cached", scale_cached, persistent=False)

    def forward(
        self,
        x,
        seq_len,
    ):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device).type_as(
                self.inv_freq
            )
            freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
            freqs = torch.cat((freqs, freqs), dim=-1).to(dtype=x.dtype)

            self.register_buffer("freqs_cached", freqs)

            if self.scale is None:
                self.register_buffer(
                    "scale_cached", torch.ones(1, device=x.device).to(dtype=x.dtype)
                )

                return self.freqs_cached.to(dtype=x.dtype), self.scale_cached

            power = (t - (seq_len // 2)) / self.scale_base
            scale = self.scale ** rearrange(power, "n -> n 1")
            scale = torch.cat((scale, scale), dim=-1).to(dtype=x.dtype)
            self.register_buffer("scale_cached", scale)

        return self.freqs_cached.to(dtype=x.dtype), self.scale_cached.to(dtype=x.dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, freqs, scale=1, position_ids=None):
    freqs = freqs[position_ids, :]
    if scale.shape[-1] != 1:
        scale = scale[position_ids, :]

    q_embed = (q * freqs.cos() * scale) + (rotate_half(q) * freqs.sin() * scale)
    k_embed = (k * freqs.cos() * 1 / scale) + (rotate_half(k) * freqs.sin() * 1 / scale)

    return q_embed, k_embed


def replace_llama_rope_with_xpos_rope():
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = XposRotaryEmbedding
    transformers.models.llama.modeling_llama.apply_rotary_pos_emb = apply_rotary_pos_emb
