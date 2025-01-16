# Copyright 2024-2025 Axolotl AI. All rights reserved.
#
# This software may be used and distributed according to
# the terms of the Apache License 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""
Custom modeling code for RALA Llama
"""

from typing import List, Optional, Tuple, Union, Unpack

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Cache, GenerationMixin, LlamaModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (
    KwargsForCausalLM,
    LlamaAttention,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .configuration_rala import LlamaRalaConfig


def kappa(x: torch.Tensor) -> torch.Tensor:  # pylint: disable=invalid-name
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
        self.phi = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

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

        # 4. If we have a past_key_value (Cache object), let it update / append
        if past_key_value is not None:
            # This is the normal Llama pattern
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # The .update() method returns updated (key_states, value_states)
            # and typically updates internal buffers. It may also store `layer_idx` data.
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # If you still want to handle the repeated KV for multi-group setups:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Now we apply RALA.

        # 1) Apply κ(.) to Q,K: shape [b, n_heads, seq_len, head_dim]
        Q_kappa = kappa(query_states)  # pylint: disable=invalid-name
        K_kappa = kappa(key_states)  # pylint: disable=invalid-name

        # 2) Compute global query Q_g = average of Q_kappa across seq_len => [b, n_heads, head_dim]
        # The paper denotes Q_g = (1/N) Σ_i Q_kappa_i
        seq_len_float = float(q_len)  # for scaling
        Q_g = Q_kappa.mean(  # pylint: disable=invalid-name
            dim=2
        )  # [b, n_heads, head_dim]

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
        X_phi = self.phi(  # pylint: disable=invalid-name
            hidden_states
        )  # [b, seq_len, d_model]
        X_phi = X_phi.view(  # pylint: disable=invalid-name
            bsz, q_len, self.num_heads, self.head_dim
        )  # [b, s, n, d]
        X_phi = X_phi.transpose(1, 2)  # [b, n, s, d]  # pylint: disable=invalid-name

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

        if output_attentions:
            # alpha => [b, n_heads, (past_len + q_len)]
            attn_weights = alpha
        else:
            attn_weights = None

        # Return 3-tuple: (attn_output, attn_weights, past_key_value)
        return attn_output, attn_weights, past_key_value


class LlamaRalaDecoderLayer(nn.Module):
    """
    LlamaDecoderLayer with RALA support
    """

    def __init__(self, config: LlamaRalaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if LlamaRalaConfig.is_layer_idx_softmax(
            config.num_hidden_layers, layer_idx, config.softmax_every
        ):
            self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        else:
            self.self_attn = LlamaRALAAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    @classmethod
    def is_layer_idx_softmax(
        cls, num_hidden_layers: int, layer_idx: int, softmax_every: int
    ) -> bool:
        inner_layers = num_hidden_layers - 2
        if 1 + softmax_every * (inner_layers // softmax_every) == inner_layers:
            softmax_start_idx = 1
        elif 1 + softmax_every * (inner_layers // softmax_every) > inner_layers:
            layer_group_size = 1 + softmax_every * ((inner_layers // softmax_every) - 1)
            softmax_start_idx = 1 + (inner_layers - layer_group_size) // 2
        elif 1 + softmax_every * (inner_layers // softmax_every) < inner_layers:
            layer_group_size = 1 + softmax_every * (inner_layers // softmax_every)
            softmax_start_idx = 1 + (inner_layers - layer_group_size) // 2

        softmax_layers = set(range(softmax_start_idx, num_hidden_layers, softmax_every))
        softmax_layers.add(0)
        softmax_layers.add(num_hidden_layers - 1)

        return layer_idx in softmax_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)  # type: ignore

        if use_cache:
            outputs += (present_key_value,)  # type: ignore

        return outputs  # type: ignore


class LlamaRalaModel(LlamaModel):
    """
    LlamaModel with RALA support
    """

    config_class = LlamaRalaConfig

    def __init__(self, config: LlamaRalaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        self.layers = nn.ModuleList(
            [
                LlamaRalaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class LlamaRalaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    """
    LlamaForCausalLM with RALA support
    """

    config_class = LlamaRalaConfig
    _no_split_modules = ["LlamaRalaDecoderLayer"]

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaRalaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],  # type: ignore
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM
        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
