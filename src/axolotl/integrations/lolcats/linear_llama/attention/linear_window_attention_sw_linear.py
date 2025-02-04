"""
Subquadratic attention combining sliding window and linear attentions
- Using "standard" sliding windows
- Didactically computes outputs with n^2 attention weights for now
- Copied + adapted from linear_window_attention_tk.py for single-file reference

For each layer:
- We first compute (softmax) attention over sliding windows
- We then compute standard linear attention to "fill in" the earlier parts
- We combine to model the entire sequence
"""

import logging
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache

try:
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
except ModuleNotFoundError:
    _flash_attention_forward = None  # Transformers v4.36

from ..model.rotary import apply_rotary_pos_emb

# Causal linear attention dot product CUDA kernel from fast-transformers
from .linear_attention import (
    LinearAttentionState,
    LolcatsLinearAttention,
    causal_dot_product,
)

LOG = logging.getLogger(
    "axolotl.integrations.lolcats.linear_attention.linear_window_attention_sw_long"
)


# ----------------------
# Sliding window helpers
# ----------------------
def get_masks(
    window_size: int, q_len: int, k_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return masks for softmax and linear attention terms
    -> 1 is include, 0 is ignore
    """
    causal_mask = torch.ones((q_len, k_len), device=device, dtype=torch.int).tril(
        max(k_len - q_len, 0)
    )
    linear_mask = torch.ones((q_len, k_len), device=device, dtype=torch.int).tril(
        max(k_len - q_len, 0) - window_size
    )
    window_mask = causal_mask - linear_mask
    # Return softmax mask (window), linear attention mask
    # -> shapes broadcast over (b, h, q_len, k_len)
    return window_mask[None, None, ...], linear_mask[None, None, ...]


def hybrid_attention_quadratic(
    q: torch.Tensor,
    k: torch.Tensor,
    f_q: torch.Tensor,
    f_k: torch.Tensor,
    v: torch.Tensor,
    window_factor: torch.Tensor,
    linear_factor: torch.Tensor,
    window_size: int,
    kv_state: Optional[torch.Tensor] = None,
    k_state: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
    mask_value: float = -1e8,
):
    """
    Hybrid attention combining sliding window and linear attentions
    """

    mask_window, mask_linear = get_masks(
        window_size, q.shape[-2], k.shape[-2], q.device
    )

    # 1. Sliding window (softmax attention)
    a_sm = torch.einsum("bhmd,bhnd->bhmn", q.float(), k.float()) * (k.shape[-1] ** -0.5)
    a_sm = a_sm.masked_fill(~mask_window.bool(), mask_value)
    # torch.softmax(a_sm, dim=-1), but we account for the max when combining
    a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
    a_sm = window_factor * torch.exp(a_sm - a_sm_max)
    sum_sm = a_sm.sum(dim=-1, keepdim=True)

    # 2. Under window (linear attention)
    a_ln = torch.einsum("bhmd,bhnd->bhmn", f_q.float(), f_k.float())
    a_ln = linear_factor * a_ln.masked_fill(~mask_linear.bool(), 0)
    sum_ln = a_ln.sum(dim=-1, keepdim=True)

    # 3. Combine
    a = ((a_sm + a_ln) / (sum_sm + sum_ln)).to(q.dtype)  # Save attention weights
    # Allow outputs to also depend on prior kv_state and k_state
    y = torch.einsum("bhmn,bhnd->bhmd", a_sm + a_ln, v.float())
    if (
        kv_state is not None and k_state is not None
    ):  # Combine with prior kv_state and k_state
        y += linear_factor * torch.einsum(
            "bhld,bhdf->bhlf", f_q.float(), kv_state.float()
        )
        sum_ln += (
            linear_factor
            * torch.einsum("bhld,bhnd->bhl", f_q.float(), k_state.float())[..., None]
        )
    y = (y / (sum_sm + sum_ln)).to(q.dtype)
    return y, a  # attention weights only for the last chunk


# ------------------------------
# Hybrid window attention linear
# ------------------------------
def under_window_linear_attention(
    f_q: torch.Tensor,
    f_k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    linear_factor: torch.Tensor,
    eps: float = 1e-12,
):
    """Compute hybrid window attention dot product with linear complexity in q_len"""
    dtype = f_q.dtype
    w = window_size
    f_k = F.pad(f_k, (0, 0, w, 0), value=0)[:, :, :-w, :]
    v = F.pad(v, (0, 0, w, 0), value=0)[:, :, :-w, :]
    qkv = linear_factor * causal_dot_product(
        f_q.contiguous().to(dtype=torch.float32),
        f_k.contiguous().to(dtype=torch.float32),
        v.contiguous().to(dtype=torch.float32),
    ).to(dtype=dtype)
    sum_f_k = f_k.float().cumsum(dim=2).to(dtype=dtype)
    sum_qk = linear_factor * torch.einsum("bhld,bhld->bhl", f_q, sum_f_k)[..., None]
    sum_qk[sum_qk == 0] += eps
    return qkv, sum_qk


def sliding_window_softmax_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    window_factor: torch.Tensor,
    mask_value: float = -1e8,
):
    """
    Compute sliding window softmax attention without materializing
    O(seq_len^2) attention weights
    """
    d = q.shape[-1]
    # Compute windows for keys
    window_kwargs = {"dimension": 2, "size": window_size, "step": 1}
    k = F.pad(k, (0, 0, window_size - 1, 0), value=0).unfold(**window_kwargs)
    v = F.pad(v, (0, 0, window_size - 1, 0), value=0).unfold(**window_kwargs)

    # Compute windowed_softmax(qk); causal in its construction
    a_sm = torch.einsum("bhld,bhldw->bhlw", q, k) * (d**-0.5)
    a_sm[a_sm == 0] = -torch.finfo(
        q.dtype
    ).max  # heuristic for zeroing out padding above
    a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
    a_sm = window_factor * torch.exp(a_sm - a_sm_max)
    sum_sm = a_sm.sum(dim=-1, keepdim=True)
    return torch.einsum("bhlw,bhldw->bhld", a_sm, v), sum_sm
    # return torch.einsum('bhlw,bhldw->bhld', torch.softmax(qk, dim=-1), v)


def hybrid_attention_linear(
    q: torch.Tensor,
    k: torch.Tensor,
    f_q: torch.Tensor,
    f_k: torch.Tensor,
    v: torch.Tensor,
    window_factor: Optional[torch.Tensor] = None,
    linear_factor: Optional[torch.Tensor] = None,
    window_size: int = 64,
    kv_state: Optional[torch.Tensor] = None,
    k_state: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
    mask_value: float = -1e8,
):
    """
    Alternative hybrid attention combining sliding window and linear attentions
    -> Uses O(n) memory if n is sequence length by padding and unfolding windows
    """
    # window_kwargs = {"dimension": 2, "size": window_size, "step": 1}
    if window_factor is None:
        raise ValueError("window_factor must be provided")

    if linear_factor is None:
        raise ValueError("linear_factor must be provided")

    # 1. Sliding window (softmax attention)
    with torch.no_grad():
        qkv_sm, sum_qk_sm = sliding_window_softmax_attention(
            q, k, v, window_size, window_factor, mask_value
        )

    # 2. Under window (linear attention)
    qkv_ln, sum_qk_ln = under_window_linear_attention(
        f_q, f_k, v, window_size, linear_factor, eps
    )

    # 3. Combine
    y = (qkv_sm + qkv_ln) / (sum_qk_sm + sum_qk_ln)
    return y, None


# ---------------------
# Attention layer class
# ---------------------
class LolcatsLinearSlidingWindowAttention(LolcatsLinearAttention):
    """
    Lolcats attention combining sliding window and linear attention
    """

    def __init__(
        self,
        window_size: int = 64,
        decode_window_size: Optional[int] = None,
        affine_attention_factors: bool = False,
        init_window_factor: float = 0,
        train_window_factor: bool = True,
        state_grad_enabled: bool = False,
        **kwargs,
    ):
        self.window_size = window_size
        self.decode_window_size = (
            decode_window_size if decode_window_size is not None else window_size
        )
        self.window_kwargs = {"dimension": 2, "size": window_size, "step": 1}
        super().__init__(**kwargs)
        # Determine how we compute attentions
        self.linear_attention = hybrid_attention_linear
        self.attention_type = "lolcats_llama_window_sw"
        # Learnable factor for combining attentions
        self.affine_attention_factors = affine_attention_factors
        device, dtype = self.q_proj.weight.device, self.q_proj.weight.dtype
        if train_window_factor:
            self.window_factors = nn.Parameter(
                init_window_factor
                * torch.ones(1, self.num_heads, 1, 1, device=device, dtype=dtype)
            )
        else:
            self.register_buffer(
                "window_factors",
                init_window_factor
                * torch.ones(1, self.num_heads, 1, 1, device=device, dtype=dtype),
            )
        # Whether we use original flash attention 2 inference (use during attention transfer)
        self.base_inference = False
        self.state_grad_enabled = state_grad_enabled

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
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

        if self.train_attention and self.base_inference:
            with torch.no_grad():
                _y_true = flash_attention_2(
                    self,  # self.base_attn,
                    hidden_states=hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )[0]
                # _y_true.shape is (batch_size, seq_len, num_heads, head_dim)
                y_true = _y_true.reshape(b, l, -1).contiguous()
                y_true = self.o_proj(y_true)
                # layer_io = (hidden_states, _y_true)  # hack
                layer_io = (hidden_states.cpu(), _y_true.cpu())  # hack
                return y_true, layer_io, None

        else:
            q, k, v, kv_seq_len = self.process_qkv(
                hidden_states, attention_mask, position_ids, past_key_value
            )
            f_q, f_k = self.feature_map_q(q), self.feature_map_k(
                k
            )  # Have to do after repeat for grouped-query attn if we use same fmap

            attn_weights = None
            # attention_mask = None  # For now this is always True
            if past_key_value is None:  # Regular training
                window_factors = F.sigmoid(self.window_factors)
                linear_factors = (
                    1 - window_factors if self.affine_attention_factors else 1
                )
                y_true, a_pred = self.linear_attention(
                    q,
                    k,
                    f_q,
                    f_k,
                    v,
                    window_factors,
                    linear_factors,
                    window_size=self.window_size,
                )
                attn_weights = a_pred
            else:
                past_key_value.window_size = self.decode_window_size
                if (
                    f_q.shape[2] == 1 and kv_seq_len > 1 and not self.training
                ):  # Generating
                    assert use_cache is True
                    _kv = past_key_value.update_for_decoding(
                        k, v, self.layer_idx, self.feature_map_k, dtype=q.dtype
                    )
                    k_cache, v_cache, f_kv_state, f_k_state = _kv

                    # Sliding window + linear attention decode
                    window_factors = F.sigmoid(self.window_factors)
                    linear_factors = (
                        1 - window_factors if self.affine_attention_factors else 1
                    )

                    # Softmax attention terms
                    a_sm = torch.einsum(
                        "bhmd,bhnd->bhmn", q.float(), k_cache.float()
                    ) * (k.shape[-1] ** -0.5)
                    a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
                    a_sm = window_factors * torch.exp(a_sm - a_sm_max)
                    sum_sm = a_sm.sum(dim=-1, keepdim=True)

                    # Combine with linear attention terms
                    y_true = torch.einsum(
                        "bhmn,bhnd->bhmd", a_sm, v_cache.float()
                    ) + linear_factors * torch.einsum(
                        "bhlf,bhfd->bhld", f_q.float(), f_kv_state.float()
                    )
                    sum_ln = (
                        linear_factors
                        * torch.einsum(
                            "bhlf,bhnf->bhl", f_q.float(), f_k_state.float()
                        )[..., None]
                    )
                    y_true = (y_true / (sum_sm + sum_ln)).to(q.dtype)

                else:  # Stateful training
                    try:
                        kv_state = past_key_value.kv_states[self.layer_idx]
                        k_state = past_key_value.k_states[self.layer_idx]
                    except IndexError:
                        kv_state, k_state = None, None
                    window_factors = F.sigmoid(self.window_factors)
                    linear_factors = (
                        1 - window_factors if self.affine_attention_factors else 1
                    )
                    y_true, _ = self.linear_attention(
                        q,
                        k,
                        f_q,
                        f_k,
                        v,
                        window_factors,
                        linear_factors,
                        window_size=self.window_size,
                        kv_state=kv_state,
                        k_state=k_state,
                    )
                    # Save and update KV cache and states
                    # past_key_value.update(k, v.detach(), self.layer_idx,
                    #                       fmap_key_states=f_k.detach(),
                    #                       accumulate_in_fp32=True)
                    past_key_value.update(
                        k,
                        v,
                        self.layer_idx,
                        fmap_key_states=f_k,
                        accumulate_in_fp32=True,
                    )
            # Concatenate heads and apply output projection
            _y_true = y_true.transpose(1, 2).contiguous()
            y_true = self.o_proj(_y_true.view(b, l, self.hidden_size))

            if self.train_attention:
                attn_weights = _y_true  # flash_attn outputs are shape (b, l, h, d)
        return y_true, attn_weights, past_key_value


class LinearAttentionSlidingWindowCache(LinearAttentionState):
    """
    Class for `past_key_values`
    -> Alternative to KV cache; here we only maintain a "KV state" and "K state"
    -> Modified from transformers.cache_utils.DynamicCache (v4.36)
    """

    def __init__(self, window_size: int = 64) -> None:
        super().__init__()
        self._seen_tokens = 0  # should be `self.seen_tokens` in Transformers v4.36
        self._seen_tokens_by_layer: List[int] = []
        self.kv_states: List[torch.Tensor] = []
        self.k_states: List[torch.Tensor] = []

        # Account for sliding windows
        self.decode_kv_states: List[torch.Tensor] = []
        self.decode_k_states: List[torch.Tensor] = []
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []
        self.window_size = window_size

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: Optional[int] = None,
        cache_kwargs: Optional[Any] = None,
        accumulate_in_fp32: bool = False,
        fmap_key_states: Optional[torch.Tensor] = None,  # should not be None
        grad_enabled: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV, K states; and KV cache during training
        - For decoding, use `self.decode_kv_states` to keep track of KV states
          up to sliding window terms
        - For (chunked) training, use `self.kv_states` to keep track of KV states
          up to end of sequence
        - Likewise for `self.decode_k_states` and `self.k_states`
        """
        if fmap_key_states is None:
            raise ValueError("fmap_key_states must not be None")

        if layer_idx is None:
            raise ValueError("Layer index must not be None")

        with torch.set_grad_enabled(grad_enabled):
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]

            dtype = key_states.dtype
            if accumulate_in_fp32:
                # key_states = key_states.float()
                fmap_key_states = fmap_key_states.float()
                value_states = value_states.float()

            # Decoding KV state (KV terms up to last window_size)
            decode_kv_state = torch.einsum(
                "bhlf,bhld->bhfd",
                fmap_key_states[:, :, : -self.window_size],
                value_states[:, :, : -self.window_size],
            )
            # KV state
            kv_state = decode_kv_state + torch.einsum(
                "bhlf,bhld->bhfd",
                fmap_key_states[:, :, -self.window_size :],
                value_states[:, :, -self.window_size :],
            )
            # shape is b, h, 1, f; note the 1
            decode_k_state = fmap_key_states[:, :, : -self.window_size].sum(
                dim=-2, keepdim=True
            )
            k_state = decode_k_state + fmap_key_states[:, :, -self.window_size :].sum(
                dim=-2, keepdim=True
            )

            # Update the cache
            if len(self.k_states) <= layer_idx:  # Initializing kv and k states
                self.kv_states.append(kv_state.to(dtype))
                self.k_states.append(k_state.to(dtype))

                self.decode_kv_states.append(decode_kv_state.to(dtype))
                self.decode_k_states.append(decode_k_state.to(dtype))

                self.k_cache.append(key_states[:, :, -self.window_size :, :])
                self.v_cache.append(
                    value_states[:, :, -self.window_size :, :].to(dtype)
                )
                # self._seen_tokens_by_layer[layer_idx].append(key_states.shape[-2])
            else:
                # Update kv and k states recurrently
                kv_state = (self.kv_states[layer_idx].to(kv_state.dtype) + kv_state).to(
                    dtype
                )
                k_state = (self.k_states[layer_idx].to(kv_state.dtype) + k_state).to(
                    dtype
                )
                self.kv_states[layer_idx] = kv_state
                self.k_states[layer_idx] = k_state

                decode_kv_state = (
                    self.decode_kv_states[layer_idx].to(kv_state.dtype)
                    + decode_kv_state
                ).to(dtype)
                decode_k_state = (
                    self.decode_k_states[layer_idx].to(kv_state.dtype) + decode_k_state
                ).to(dtype)
                self.decode_kv_states[layer_idx] = decode_kv_state
                self.decode_k_states[layer_idx] = decode_k_state

                self.k_cache[layer_idx] = key_states[:, :, -self.window_size :, :]
                self.v_cache[layer_idx] = value_states[:, :, -self.window_size :, :]
            self._seen_tokens_by_layer[layer_idx] += key_states.shape[-2]

        return self.kv_states[layer_idx], self.k_states[layer_idx]

    def update_for_decoding(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
        feature_map_k: Callable,
        dtype: torch.dtype,
    ):
        """
        Update the decoding KV and K states, and KV cache, during decodeing
        """
        with torch.no_grad():
            k_cache = self.k_cache[layer_idx]
            v_cache = self.v_cache[layer_idx]

            if k_cache.shape[-2] < self.window_size:  # build window-size cache
                self.k_cache[layer_idx] = torch.cat([k_cache, keys], dim=-2)
                self.v_cache[layer_idx] = torch.cat([v_cache, values], dim=-2)
            else:
                # MZ 6/3: handle short inputs; zero-out padding when initial k.shape[2] < self.window_size
                # if k_cache[:, :, :1, :].sum() == 0:   # heuristic for zeroing out padding in cache
                #     f_k_state = torch.zeros(k_cache[:, :, :1, :].shape, dtype=dtype, device=k_cache.device)
                # else:
                #     f_k_state = feature_map_k(k_cache[:, :, :1, :])
                # -> MZ (later): above only relevant if we zero-pad in our hybrid attention computation
                k_state = feature_map_k(k_cache[:, :, :1, :])
                v_state = v_cache[:, :, :1, :]
                kv_state = torch.einsum(
                    "bhlf,bhld->bhfd", k_state.float(), v_state.float()
                ).to(
                    dtype
                )  # b, h, f, d
                self.decode_kv_states[layer_idx] += kv_state
                self.decode_k_states[layer_idx] += k_state

                self.k_cache[layer_idx] = torch.cat(
                    [k_cache[:, :, 1:, :], keys], dim=-2
                )
                self.v_cache[layer_idx] = torch.cat(
                    [v_cache[:, :, 1:, :], values], dim=-2
                )

            if layer_idx == 0:
                self._seen_tokens += keys.shape[-2]
            self._seen_tokens_by_layer[layer_idx] += keys.shape[-2]
            return (
                self.k_cache[layer_idx],
                self.v_cache[layer_idx],
                self.decode_kv_states[layer_idx],
                self.decode_k_states[layer_idx],
            )


# -----------------
# Flash Attention 2
# -----------------


def flash_attention_2(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
):
    """
    Wrapper for LlamaFlashAttention2
    Copied and modified from HF Transformers v4.36 and v4.43 implementations
    - (4.43) https://github.com/huggingface/transformers/blob/868d36d29ec132deeaaf8571b25b6a1b911d0145/src/transformers/models/llama/modeling_llama.py#L402
    - (4.36) https://github.com/huggingface/transformers/blob/a7cab3c283312b8d4de5df3bbe719971e24f4281/src/transformers/models/llama/modeling_llama.py#L456
    """

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    try:  # As in Transformers v4.36
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(key_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
    except Exception:  # As in Transformers v4.39
        cos, sin = self.rotary_emb(key_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        LOG.debug(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    if getattr(self, "_flash_attention_forward", False):
        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            is_causal=True,
        )
    else:
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=0,  # dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=True,
        )
    return attn_output, past_key_value
