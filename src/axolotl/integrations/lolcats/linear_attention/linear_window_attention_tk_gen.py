"""
LoLCATs + ThunderKittens linear attention + sliding window for generation
"""

import logging
from typing import Any, Callable, List, Optional

import torch
import torch.nn.functional as F

from .linear_attention import LinearAttentionState
from .linear_window_attention_tk_long import LolcatsTKWindowLongAttention

LOG = logging.getLogger(
    "axolotl.integrations.lolcats.linear_attention.linear_attention_tk_gen"
)

try:
    from thunderkittens import hedgehog as tk_window_hedgehog_attention

    LOG.debug("Successfully imported ThunderKittens for TK window attention")
except ImportError:
    LOG.debug("Failed to import ThunderKittens for TK window attention")


class LolcatsWindowAttentionTKGen(LolcatsTKWindowLongAttention):
    def __init__(self, *args, window_size: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_attention = False
        self.base_inference = False
        self.window_size = 64  # hard-coded support for TK kernel
        self.decode_window_size = 64

        b, h, l, d = 1, 32, 8192, 128
        self.y_true = torch.zeros(b, h, l, d, dtype=torch.bfloat16, device="cuda")
        self.kv_state = torch.zeros(b, h, d, d, dtype=torch.float32, device="cuda")
        self.k_state = torch.zeros(b, h, d, dtype=torch.float32, device="cuda")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,  # “legacy” cache approach
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        """
        Forward pass with the option to compute attention weights multiple ways
        if self.train_attention is True
        -> Consistent with HuggingFace Transformers for easy use with their pretrained models
        """
        b, l, _ = hidden_states.size()
        assert (
            past_key_value is not None
        ), "past_key_value must be provided for generation"
        assert (
            self.train_attention is False
        ), "train_attention is not supported for generation"
        assert (
            self.base_inference is False
        ), "base_inference is not supported for generation"
        assert use_cache is True, "use_cache must be True for generation"
        past_key_value.window_size = self.decode_window_size
        q, k, v, kv_seq_len = self.process_qkv(
            hidden_states, attention_mask, position_ids, past_key_value
        )
        if q.shape[2] == 1 and kv_seq_len > 1:  # Generating after prefill
            f_q = self.feature_map_q(q)
            _kv = past_key_value.update_for_decoding(
                k, v, self.layer_idx, self.feature_map_k
            )
            k_cache, v_cache, kv_state, k_state = _kv
            # Sliding window + linear attention decode
            window_factors = F.sigmoid(self.window_factors)
            linear_factors = 1 - window_factors if self.affine_attention_factors else 1

            # Softmax attention terms
            a_sm = torch.einsum("bhmd,bhnd->bhmn", q.float(), k_cache.float()) * (
                k.shape[-1] ** -0.5
            )
            a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
            a_sm = window_factors * torch.exp(a_sm - a_sm_max)
            sum_sm = a_sm.sum(dim=-1, keepdim=True)

            # Combine with linear attention terms
            y_true = torch.einsum(
                "bhmn,bhnd->bhmd", a_sm, v_cache.float()
            ) + linear_factors * torch.einsum(
                "bhld,bhdf->bhlf", f_q.float(), kv_state.float()
            )
            sum_ln = (
                linear_factors
                * torch.einsum("bhld,bhnd->bhl", f_q.float(), k_state.float())[
                    ..., None
                ]
            )
            self.y_true = (y_true / (sum_sm + sum_ln)).to(q.dtype)

        else:  # Process prefill
            # Use TK-implemented linear + terrace window attention
            b, h, l, d = q.shape
            device = q.device
            # tk.hedgehog arguments
            # y_true   = torch.zeros(b, h, l, d, dtype=torch.bfloat16, device=device)
            # kv_state = torch.zeros(b, h, d, d, dtype=torch.float32, device=device)
            # k_state  = torch.zeros(b, h, d, dtype=torch.float32, device=device)
            betas = F.sigmoid(self.window_factors[0, :, 0, 0].to(dtype=torch.float32))
            alphas = (
                1 - betas
                if self.affine_attention_factors
                else torch.ones(betas.shape, dtype=torch.float32, device=device)
            )
            q_map = self.feature_map_q.mlp.layer
            k_map = self.feature_map_k.mlp.layer
            # Saves outputs to y_pred, k_state, kv_state, where we fuse:
            # 1. f_q, f_k = self.feature_map_q(q), self.feature_map_k(k)
            # 2. y_pred = attention(q, k, f_q, f_k, v)  # b, h, l, d
            # 3. kv_state = torch.einsum(‘bhlf,bhld->bhfd’,
            #                            f_k[:, :, :-self.window_size],
            #                            v[:, :, :-self.window_size])  # b, h, f, d
            # 4. k_state = f_k[:, :, :-self.window_size].sum(dim=-2)   # b, h, d

            tk_window_hedgehog_attention(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                self.y_true,
                self.k_state,
                self.kv_state,
                q_map,
                k_map,
                alphas,
                betas,
            )

            past_key_value.update_with_kv(
                self.kv_state, self.k_state.unsqueeze(-2), k, v, self.layer_idx
            )

        # Concatenate heads and apply output projection
        y_true = self.y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
        y_true = self.o_proj(y_true)
        return y_true, None, past_key_value


class LinearAttentionTKWindowGenerationCache(LinearAttentionState):
    """
    Class for `past_key_values`
    -> Alternative to KV cache; here we only maintain a “KV state” and “K state”
    -> Modified from transformers.cache_utils.DynamicCache (v4.36)
    """

    def __init__(self, window_size: int = 64) -> None:
        super().__init__()
        self._seen_tokens = 0  # should be `self.seen_tokens` in Transformers v4.36
        self._seen_tokens_by_layer: List[int] = []
        self.window_size = window_size

        self.decode_kv_states: List[torch.Tensor] = []
        self.decode_k_states: List[torch.Tensor] = []
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []

    def update_with_kv(
        self,
        kv_state: torch.Tensor,
        k_state: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
    ):
        """
        Update the cache with new KV and K states
        """
        if layer_idx == 0:
            self._seen_tokens += k.shape[2]
        self._seen_tokens_by_layer.append(k.shape[2])

        # Initialize KV and K states
        if len(self.decode_k_states) <= layer_idx:
            self.decode_kv_states.append(kv_state)
            self.decode_k_states.append(k_state)
        else:  # Update KV and K states
            self.decode_kv_states[layer_idx] = (
                self.decode_kv_states[layer_idx] + kv_state
            )
            self.decode_k_states[layer_idx] = self.decode_k_states[layer_idx] + k_state

        self.k_cache.append(k[:, :, -self.window_size :, :])
        self.v_cache.append(v[:, :, -self.window_size :, :])

    def update_for_decoding(
        self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, feature_map_k: Callable
    ):
        """
        Update the cache for decoding
        """
        k_cache = self.k_cache[layer_idx]
        v_cache = self.v_cache[layer_idx]
        k_state = feature_map_k(k_cache[:, :, :1, :])
        v_state = v_cache[:, :, :1, :]
        kv_state = torch.einsum("bhlf,bhld->bhfd", k_state.float(), v_state.float()).to(
            k.dtype
        )

        self.decode_kv_states[layer_idx] += kv_state
        self.decode_k_states[layer_idx] += k_state

        self.k_cache[layer_idx] = torch.cat([k_cache[:, :, 1:, :], k], dim=-2)
        self.v_cache[layer_idx] = torch.cat([v_cache[:, :, 1:, :], v], dim=-2)
        if layer_idx == 0:
            self._seen_tokens += k.shape[-2]
        self._seen_tokens_by_layer[layer_idx] += k.shape[-2]
        return (
            self.k_cache[layer_idx],
            self.v_cache[layer_idx],
            self.decode_kv_states[layer_idx],
            self.decode_k_states[layer_idx],
        )
