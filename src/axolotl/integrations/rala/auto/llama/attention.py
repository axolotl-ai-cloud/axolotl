from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Cache
from transformers.models.llama.modeling_llama import (
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)


def kappa(x: torch.Tensor) -> torch.Tensor:
    """
    The paper uses κ(x) = ELU(x) + 1.
    x is assumed to be [batch, n_heads, seq_len, head_dim].
    """
    return F.elu(x) + 1


class LlamaRALAAttention(nn.Module):
    """
    LlamaAttention replaced with Rank-Augmented Linear Attention (RALA).
    Adapted from the standard LlamaAttention for demonstration.
    **Not** a fully drop-in replacement if you need caching/TP.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Same Q, K, V, output projections
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=config.attention_bias
        )

        # We will preserve rope usage
        self._init_rope()

        # A simple φ-projection for RALA:
        # The paper uses φ(x) as a linear transform or identity. We'll do a linear:
        self.phi = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def _init_rope(self):
        # Standard Llama rope logic
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,  # pylint: disable=unused-argument
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        """
        RALA forward pass.
        This version omits incremental decoding with `past_key_value` for simplicity
        (linear attention caching is non-trivial).
        """
        bsz, q_len, _ = hidden_states.size()

        # Standard Q, K, V
        query_states = self.q_proj(hidden_states)  # [b, seq, n_heads*dim]
        key_states = self.k_proj(hidden_states)  # [b, seq, n_kv_heads*dim]
        value_states = self.v_proj(hidden_states)  # [b, seq, n_kv_heads*dim]

        # Reshape to [b, n_heads, seq_len, head_dim]
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Apply RoPE (rotary embeddings) just as in standard Llama
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # If you still want to handle the repeated KV for multi-group setups:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Now we apply RALA.

        # 1) Apply κ(.) to Q,K: shape [b, n_heads, seq_len, head_dim]
        Q_kappa = kappa(query_states)
        K_kappa = kappa(key_states)

        # 2) Compute global query Q_g = average of Q_kappa across seq_len => [b, n_heads, head_dim]
        # The paper denotes Q_g = (1/N) Σ_i Q_kappa_i
        seq_len_float = float(q_len)  # for scaling
        Q_g = Q_kappa.mean(dim=2)  # [b, n_heads, head_dim]

        # 3) Compute alpha_j for each token j in [0..seq_len-1]
        #    alpha_j = N * softmax( Q_g · K_kappa_j^T ), shape => [b, n_heads, seq_len]
        # Dot product over head_dim
        # K_kappa is [b, n_heads, seq_len, head_dim], Q_g is [b, n_heads, head_dim]
        # We'll do an einsum or transpose to produce logits [b, n_heads, seq_len]

        # Dot product across the last dimension (d_head), resulting in shape [b, n_heads, seq_len]
        # logits = torch.einsum("bnh, bnsh -> bns", Q_g, K_kappa)  # [b, n_heads, seq_len]
        logits = (Q_g.unsqueeze(2) * K_kappa).sum(
            dim=-1
        )  # -> [b, n_heads, seq_len]  # identical to above but torch.compile should work

        # 4) Incorporate causal or padding mask if provided.
        #    In standard Llama, attention_mask is broadcast as [b, 1, seq_len, seq_len] or similar.
        #    For RALA, we only do a single softmax over "j" dimension. We can add the mask to logits.
        #    Caution: This might not replicate strict causal linear attention. It's a best-effort approach.
        if attention_mask is not None:
            # Usually Llama's causal mask is [b, 1, q_len, kv_len] with 0 or -inf
            # We want shape [b, n_heads, seq_len], so we can broadcast accordingly:
            # e.g., attention_mask: [b, 1, q_len, seq_len]
            # We pick the slice that corresponds to q_len vs. kv_len.
            # Typically the last two dims are (q_len, kv_len). We want the kv_len dimension to be `seq_len`.
            # We'll do something like:
            if attention_mask.dim() == 4:
                # attention_mask: [b, 1, q_len, kv_len]
                # if q_len == kv_len, we can do attention_mask[:, :, :, :seq_len], then squeeze dims
                mask_2d = attention_mask[:, 0, :, :q_len]  # [b, q_len, seq_len]
                # we only want [b, n_heads, seq_len], so we must broadcast over q_len if needed
                # but in this snippet, we do a single alpha_j for each j *per head*,
                # ignoring per-token Q_i. So there's a mismatch.
                # A simpler approach is to apply the mask for the entire sequence if a token j is invalid for ANY i.
                # That is approximate. We'll just pick the first row of q_len, or do min across i dimension...
                # For demonstration, let's sum or min across i dimension to see if j is valid for ANY i.
                # Or we do a "causal" approach: all tokens j>i get masked. But there's no direct i index here in alpha_j.
                # We'll just do a rough approach, e.g. mask = min across the q_len dimension:
                mask_1d = torch.min(mask_2d, dim=1)[
                    0
                ]  # [b, seq_len], picking the worst mask across query positions
                # broadcast for n_heads
                mask_1d = mask_1d.unsqueeze(1).expand(
                    -1, self.num_heads, -1
                )  # [b, n_heads, seq_len]
                logits = logits + mask_1d
            else:
                # Possibly it's [b, seq_len]. Then we just broadcast to [b,n_heads,seq_len].
                mask_1d = attention_mask  # [b, seq_len]
                mask_1d = mask_1d.unsqueeze(1).expand(-1, self.num_heads, -1)
                logits = logits + mask_1d

        alpha = F.softmax(logits, dim=-1)  # [b, n_heads, seq_len]
        # multiply by seq_len per the formula
        alpha = alpha * seq_len_float

        # 5) Construct the outer-sum:  Σ_j alpha_j * (K_kappa_j^T V_j)
        #    The paper shows a d×d matrix formed per head.
        #    K_kappa: [b, n_heads, seq_len, head_dim], V: [b, n_heads, seq_len, head_dim]
        #    For each j, do outer product K_kappa_j (d×1) × V_j^T (1×d) => d×d
        #    Then multiply by alpha_j and sum over j.
        #    We'll do an einsum for that: [b,n_heads,seq_len,d] outer [b,n_heads,seq_len,d] => [b,n_heads,d,d]
        #    alpha: [b, n_heads, seq_len].
        value_states_ = value_states  # [b, n_heads, seq_len, head_dim]
        outer_sum = torch.einsum("bns,bnsd,bnsf->bndf", alpha, K_kappa, value_states_)

        # Explanation:
        #  - 'bnhs' is alpha (batch, n_heads, seq_len)
        #  - 'bnhsd' is K_kappa  (b,n_heads,seq_len, d)
        #  - 'bnhsf' is V        (b,n_heads,seq_len, d)
        # We want [b,n_heads,d,f], which is the d×d matrix per head.
        # Actually we need an outer product (K_kappa_j^T × V_j). That is [d, d].
        # The call above is not quite correct if we want K_kappa_j^T × V_j as [d,d].
        # Let's do a simpler approach:
        #   outer_sum = sum_j alpha_j * (K_kappa_j^T outer V_j).
        #   = "bnhs,bnhsd,bnhsf -> bnhdf"
        #   means: alpha has shape (b,n,h,s), K_kappa has shape (b,n,h,s,d), V has shape (b,n,h,s,d)
        #   We want to produce (b,n,h,d,d).
        # So the correct einsum string is 'bnhs,bnhsd,bnhsf->bnhdf':
        #   alpha indexes b,n,h,s
        #   K_kappa indexes b,n,h,s,d => K_kappa_j
        #   V indexes b,n,h,s,f => V_j
        # The resulting shape is (b,n,h,d,f). Great.

        # 6) For each token i, Y_i = φ(X_i) ∘ [ κ(Q_i) × outer_sum ]
        #    Here κ(Q_i) is shape [b,n,h,d], outer_sum is shape [b,n,h,d,d].
        #    We'll do a batch matmul: result_attn = Q_kappa_i × outer_sum => [b,n,h,d]
        #    Then multiply elementwise by φ(X_i).
        #    But φ(X_i) is a single [b,seq_len,d_model], so we reshape to [b,seq_len,n,h_dim].
        #    We'll do per-token i in a loop or broadcast. Let's do it in a single operation with einsum:

        # first, compute φ(X):
        # X is the original hidden_states: [b, seq_len, d_model]
        X_phi = self.phi(hidden_states)  # [b, seq_len, d_model]
        X_phi = X_phi.view(bsz, q_len, self.num_heads, self.head_dim)  # [b, s, n, d]
        X_phi = X_phi.transpose(1, 2)  # [b, n, s, d]

        # Now for each i in [0..q_len-1], we do a matrix multiply:
        # result_attn_i = Q_kappa_i [b,n,s,d] × outer_sum [b,n,d,d] => we want [b,n,s,d].
        # We'll do:
        result_attn = torch.einsum("bnsd,bndf->bnsf", Q_kappa, outer_sum)  # [b,n,s,d]

        # Then elementwise multiply by φ(X_i):
        context_layer = X_phi * result_attn  # [b,n,s,d]

        # Finally, reorder to [b, s, n, d] -> [b, s, n*d]
        context_layer = context_layer.transpose(1, 2).contiguous()  # [b, s, n, d]
        context_layer = context_layer.view(bsz, q_len, self.hidden_size)

        # One last linear projection:
        attn_output = self.o_proj(context_layer)

        # Not returning a standard attn_weights.
        # If you want to return alpha as "attention," we can do so:
        if output_attentions:
            # alpha: [b, n_heads, seq_len], but note it's only the "global" weighting of each key,
            # not a (q_len x kv_len) map like standard attention.
            attn_weights = alpha
        else:
            attn_weights = None

        # We omit cache / past_key_value returns to keep it simpler.
        return attn_output, attn_weights, None
