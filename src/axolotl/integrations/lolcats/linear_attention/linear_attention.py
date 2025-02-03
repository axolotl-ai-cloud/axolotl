"""
Linear attention classes
"""

import copy
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import Cache

# Causal linear attention dot product CUDA kernel from fast-transformers
try:
    from csrc import causal_dot_product as fast_causal_dot_product
except ImportError:
    fast_causal_dot_product = None

from ..model.feature_map import init_feature_map, init_learned_kernel
from ..model.rotary import apply_rotary_pos_emb, get_rotary_embeddings
from .utils import repeat_kv

# -------------------
# Attention functions
# -------------------


def causal_dot_product(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    Causal linear attention dot product
    - If available, use CUDA kernel from fast-transformers
    """
    if fast_causal_dot_product is None:
        kv = torch.einsum("bhlf,bhld->bhlfd", k, v)
        return torch.einsum("bhlf,bhlfd->bhld", q, kv.cumsum(dim=2))
    return fast_causal_dot_product(q, k, v)


def linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    fp32_attention: bool = False,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Compute linear attention with CUDA kernel implementation from fast-transformers
    - https://github.com/idiap/fast-transformers
    - Assume q, k are shape (batch_size, num_heads, seq_len, feature_dim);
      v is shape (b, h, l, head_dim)
    """
    dtype = q.dtype
    # Causal mask already applied
    y = causal_dot_product(
        q.contiguous().to(dtype=torch.float32),
        k.contiguous().to(dtype=torch.float32),
        v.contiguous().to(dtype=torch.float32),
    )
    if fp32_attention:
        y = (
            y
            / (
                torch.einsum("bhld,bhld->bhl", q.float(), k.float().cumsum(dim=2)) + eps
            )[..., None]
        ).to(dtype=dtype)
    else:
        y = y.to(dtype=dtype)
        k = k.float().cumsum(dim=2).to(dtype=dtype)
        y = y / (torch.einsum("bhld,bhld->bhl", q, k) + eps)[..., None]
    return y, None, None


def softmax_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: Optional[torch.Tensor] = None,
    causal: bool = True,
    fp32_attention: bool = True,
):
    """
    Standard softmax attention; only compute outputs if v is not None
    -> Assume q, k, v are shape (batch_size, num_heads, seq_len, head_dim)
    """
    y = None
    a = torch.einsum("bhmd,bhnd->bhmn", q, k) * (k.shape[-1] ** -0.5)
    if causal:  # Apply causal mask
        m, n = a.shape[-2:]
        causal_mask = torch.ones((m, n), device=a.device, dtype=torch.bool).triu(
            n - m + 1
        )
        a = a.masked_fill(causal_mask, -torch.finfo(a.dtype).max)
    if fp32_attention:
        a = torch.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
    else:
        a = torch.softmax(a, dim=-1)
    if v is not None:
        y = torch.einsum("bhmn,bhnd->bhmd", a, v)
    return y, a, None


def quadratic_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: Optional[torch.Tensor] = None,
    causal: bool = True,
    fp32_attention: bool = False,
    eps: float = 1e-12,
):
    """
    Compute attention with feature maps by instantiating L x L matrix of attention weights
    -> Use for attention distillation
    -> Assume q, k are shape (batch_size, num_heads, seq_len, feature_dim); v is shape (b, h, l, head_dim)
    """
    y = None
    dtype = q.dtype
    if fp32_attention:
        q, k = q.float(), k.float()
    a = torch.einsum("bhmd,bhnd->bhmn", q, k)  # note we don't scale, tho we could
    if causal:  # Apply causal mask
        m, n = a.shape[-2:]
        causal_mask = torch.ones((m, n), device=a.device, dtype=torch.bool).triu(
            n - m + 1
        )
        a = a.masked_fill(causal_mask, 0)
    # Normalize to compute attention
    a = a / (a.sum(dim=-1, keepdim=True) + eps)
    a = a.to(dtype=dtype) if fp32_attention else a
    if torch.isnan(a).sum() > 0:
        breakpoint()
    if v is not None:
        y = torch.einsum("bhmn,bhnd->bhmd", a, v)
    return y, a, None


# ---------------------
# Attention layer class
# ---------------------


class LolcatsLinearAttention(nn.Module):
    """
    LoLCATs attention implementation initialized from a
    `LlamaAttention` or `MistralAttention` object (base_attn)

    Most of the arguments are directly tied to argparse args
    - For now we don't support padding.
    """

    def __init__(
        self,
        base_attn: nn.Module,  # like LlamaAttention
        feature_map: str,
        feature_map_kwargs: dict,
        layer_idx: Optional[int] = None,
        max_layer_idx: Optional[int] = None,
        learned_kernel: Optional[str] = None,
        learned_kernel_kwargs: Optional[dict] = None,
        tie_qk_kernels: Optional[bool] = False,
        rotary_config: Optional[dict] = None,
        train_attention: Optional[bool] = False,
        remove_base_attn: bool = True,
        attention_type: Optional[str] = "lolcats_llama",
        mask_value: int = 0,
        eps: float = 1e-12,
        fp32_attention: bool = False,
        track_state_grads: bool = False,
        rank: Optional[int] = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.base_config = getattr(base_attn, "config", None)
        if self.base_config is not None:
            self.base_config = self.base_config.to_dict()
        self.attention_type = attention_type
        self.mask_value = mask_value
        self.eps = eps
        self.layer_idx = layer_idx if layer_idx is not None else base_attn.layer_idx
        self.max_layer_idx = max_layer_idx
        self.tie_qk_kernels = tie_qk_kernels
        self.train_attention = train_attention
        self.base_inference = False
        self.fp32_attention = fp32_attention
        self.track_state_grads = track_state_grads
        if rank == 0:  # multi-gpu
            if fp32_attention and layer_idx == 0:
                print(f"-> fp32_attention is {fp32_attention}")
            if layer_idx == 0 and feature_map_kwargs is not None:
                for k, v in feature_map_kwargs.items():
                    print(f"-> {k}: {v}")
            if layer_idx == 0 and learned_kernel_kwargs is not None:
                for k, v in learned_kernel_kwargs.items():
                    print(f"-> {k}: {v}")

        self.remove_base_attn = remove_base_attn

        self.init_weights_(base_attn, remove_base_attn)
        self.init_feature_map_(
            feature_map, feature_map_kwargs, learned_kernel, learned_kernel_kwargs
        )

    def init_feature_map_(
        self,
        feature_map: str,
        feature_map_kwargs: dict,
        learned_kernel: Optional[str] = None,
        learned_kernel_kwargs: Optional[dict] = None,
    ):
        """
        Initialize MLP-based feature map
        """
        self.fmap_gqa = False  # Turn True if specified below
        if learned_kernel is not None and learned_kernel_kwargs is not None:
            # Ensure dict
            learned_kernel_kwargs = {k: v for k, v in learned_kernel_kwargs.items()}
            learned_kernel_kwargs["num_heads"] = self.num_heads
            learned_kernel_kwargs["head_dim"] = self.head_dim
            learned_kernel_kwargs["dtype"] = self.q_proj.weight.dtype
            learned_kernel_kwargs["device"] = self.q_proj.weight.device
            # Create MLP
            mlp_learned_kernel = init_learned_kernel(
                learned_kernel, **learned_kernel_kwargs
            )
        # Add "activation"; see src.models.feature_map.py
        self.feature_map_q = init_feature_map(
            name=feature_map, mlp=mlp_learned_kernel, **feature_map_kwargs
        )
        if self.tie_qk_kernels:  # tie mlp weights for query and key feature maps
            self.feature_map_k = self.feature_map_q
        else:
            self.feature_map_k = copy.deepcopy(self.feature_map_q)

    def init_weights_(self, base_attn: nn.Module, remove_base_attn: bool = True):
        """
        Initialize module layers, weights, positional dependencies, etc.
        from original softmax attention layer (base_attn)
        """
        # Make other attributes accessible
        self.attention_dropout = 0  # We don't use dropout
        self.hidden_size = base_attn.config.hidden_size
        self.num_heads = base_attn.config.num_attention_heads
        self.head_dim = base_attn.head_dim
        self.num_key_value_heads = base_attn.config.num_key_value_heads
        self.num_key_value_groups = base_attn.num_key_value_groups

        self.q_shape = [self.num_heads, self.head_dim]
        self.k_shape = [self.num_key_value_heads, self.head_dim]
        self.v_shape = [self.num_key_value_heads, self.head_dim]

        # Copy original model projection layers
        self.q_proj = base_attn.q_proj
        self.k_proj = base_attn.k_proj
        self.v_proj = base_attn.v_proj
        self.o_proj = base_attn.o_proj
        try:  # If wanting to use FA2 for ground-truth inference
            self._flash_attn_uses_top_left_mask = (
                base_attn._flash_attn_uses_top_left_mask
            )
        except AttributeError:
            pass

        if self.remove_base_attn or remove_base_attn:
            del base_attn  # We don't need to keep these around
        else:
            self.base_attn = base_attn  # For some training runs helpful to just call

    def process_qkv(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Any] = None,
    ):
        """
        Compute queries, keys, and values
        """
        b, l, _ = hidden_states.size()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        kv_seq_len = k.shape[-2]

        # Shape is (batch_size, seq_len, num_heads, head_dim)
        q = q.view(b, l, *self.q_shape).transpose(1, 2)
        k = k.view(b, l, *self.k_shape).transpose(1, 2)
        v = v.view(b, l, *self.v_shape).transpose(1, 2)

        if (
            past_key_value is not None
        ):  # and k.shape[2] > q.shape[2]:  # e.g., when generating
            past_key_value.window_size = getattr(
                self, "decode_window_size", None
            )  # self.decode_window_size
            if isinstance(
                past_key_value, Cache
            ):  # In Transformers v4.36+ this is a DynamicCache object
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx
                )
            else:
                kv_seq_len += past_key_value[0].shape[-2]

        # Apply rotary embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        return q, k, v, kv_seq_len

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Any] = None,  # "legacy" cache approach
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        """
        Forward pass modified from transformers.models.mistral.modeling_mistral (v4.36)
        - Consistent with HuggingFace Transformers for easy use with their pretrained models
        """
        b, l, _ = hidden_states.size()
        q, k, v, kv_seq_len = self.process_qkv(
            hidden_states, attention_mask, position_embeddings, past_key_value
        )

        if self.base_inference:
            with torch.no_grad():
                # 1. Compute "ground-truth" attention output and weights
                y_true, _, _ = softmax_attention(q, k, v, causal=True)
                y_true = (
                    y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
                )
                y_true = self.o_proj(y_true)
                attn_weights = (None, None)

        elif self.train_attention:  # Distilling / learning attentions
            # Note for now we assume no padding when distilling; attention masks only enforce causality
            assert (
                output_attentions is True
            ), f"When training feature maps, output_attentions should be True but is {output_attentions}"
            with torch.no_grad():
                # 1. Compute "ground-truth" attention output and weights
                _y_true, attn_true, _ = softmax_attention(q, k, v, causal=True)
                y_true = (
                    _y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
                )
                y_true = self.o_proj(y_true)

            # 2. Compute "predicted" attention (just weights)
            q, k = self.feature_map_q.q_map(q), self.feature_map_k.k_map(k)
            y_pred, attn_pred, _ = quadratic_attention(q, k, v, causal=True)
            attn_weights = (  # type: ignore
                (attn_pred, attn_true),
                (y_pred, _y_true),
            )  # Save both attention weights so we can supervise.

        else:  # Finetuning
            q, k = self.feature_map_q(q), self.feature_map_k(k)
            # Apply prefill mask
            if attention_mask is not None and q.shape[2] > 1:
                if len(attention_mask.shape) == 4:
                    lin_attn_mask = (attention_mask == 0)[:, :1, -1, :l][
                        ..., None
                    ]  # b, 1, k_len, 1
                else:
                    lin_attn_mask = attention_mask[:, None, :, None]  # b, 1, k_len, 1
                k = k.masked_fill(~lin_attn_mask, 0)

            if past_key_value is not None:  # Initialize states
                if len(past_key_value.kv_states) == self.layer_idx:
                    b, h, _, f = k.shape
                    past_key_value.kv_states.append(
                        torch.zeros(
                            b, h, f, self.head_dim, dtype=q.dtype, device=q.device
                        )
                    )
                    past_key_value.k_states.append(
                        torch.zeros(b, h, 1, f, dtype=q.dtype, device=q.device)
                    )
                # Generating
                if q.shape[2] == 1 and kv_seq_len > 1 and past_key_value is not None:
                    assert use_cache is True
                    kv_state, k_state = past_key_value.update(
                        k, v, self.layer_idx, accumulate_in_fp32=self.fp32_attention
                    )
                    if self.fp32_attention:
                        q = q.float()
                        y_true = (
                            torch.einsum("bhlf,bhfd->bhld", q, kv_state.float())
                            / torch.einsum("bhlf,bhlf->bhl", q, k_state.float())[
                                ..., None
                            ]
                        ).to(dtype=k.dtype)
                    else:
                        y_true = (
                            torch.einsum("bhlf,bhfd->bhld", q, kv_state)
                            / torch.einsum("bhlf,bhlf->bhl", q, k_state)[..., None]
                        )
                else:
                    kv_state = past_key_value.kv_states[self.layer_idx]
                    k_state = past_key_value.k_states[self.layer_idx]
                    y_true, _, _ = linear_attention(
                        q, k, v, self.fp32_attention, self.eps
                    )  # Ordinarily the states are ignored
                    past_key_value.update(
                        k.detach(),
                        v.detach(),
                        self.layer_idx,
                        accumulate_in_fp32=self.fp32_attention,
                    )
                    # doing some unnecessary recomputation here
            else:
                y_true, _, _ = linear_attention(q, k, v, self.fp32_attention, self.eps)

            # Concatenate heads and apply output projection
            y_true = y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
            y_true = self.o_proj(y_true)
            attn_weights = None

        return y_true, attn_weights


class LinearAttentionState(Cache):
    """
    Handle the KV and K states for linear attention
    - Adopts HF Transformers `past_key_values` convention
    - Inherits from `Cache` class
    - Modified from transformers.cache_utils.DynamicCache (v4.36)
    """

    def __init__(self) -> None:
        self._seen_tokens = 0  # should be `self.seen_tokens` in Transformers v4.36
        self._seen_tokens_by_layer: List[int] = []
        self.kv_states: List[torch.Tensor] = []
        self.k_states: List[torch.Tensor] = []

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Returns the sequence length of the cached states. A layer index can be optionally passed.
        """
        if layer_idx is None:
            raise ValueError("Layer index must not be None")

        if len(self._seen_tokens_by_layer) <= layer_idx:  # Initializing kv and k states
            self._seen_tokens_by_layer.append(0)
        return self._seen_tokens_by_layer[layer_idx]

    def get_max_length(self) -> Optional[int]:
        """
        Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length.
        """
        return None

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: Optional[int] = None,
        cache_kwargs: Optional[Any] = None,
        accumulate_in_fp32: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx is None:
            raise ValueError("Layer index must not be None")

        with torch.no_grad():
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]
            dtype = key_states.dtype
            if accumulate_in_fp32:
                key_states, value_states = key_states.float(), value_states.float()

            kv_state = torch.einsum(
                "bhlf,bhld->bhfd", key_states, value_states
            ).detach()
            k_state = key_states.sum(
                dim=-2, keepdim=True
            ).detach()  # b, h, 1, f; note the 1
            # Update the cache
            if len(self.k_states) <= layer_idx:  # Initializing kv and k states
                print(
                    "if len(self.k_states) <= layer_idx:  # Initializing kv and k states"
                )
                self.kv_states.append(kv_state.to(dtype))
                self.k_states.append(k_state.to(dtype))
            else:
                kv_state = (self.kv_states[layer_idx].to(kv_state.dtype) + kv_state).to(
                    dtype
                )
                k_state = (self.k_states[layer_idx].to(kv_state.dtype) + k_state).to(
                    dtype
                )
                self.kv_states[layer_idx] = kv_state
                self.k_states[layer_idx] = k_state
            self._seen_tokens_by_layer[layer_idx] += key_states.shape[-2]
        return self.kv_states[layer_idx], self.k_states[layer_idx]

    def to_legacy_cache(self):
        """Hack, but just return self"""
        return self

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """
        Reorders the cache for beam search, given the selected beam indices.
        -> Copied from transformers/src/transformers/cache_utils.py
        """
        raise NotImplementedError(
            "Reordering cache not implemented for LinearAttentionState"
        )
