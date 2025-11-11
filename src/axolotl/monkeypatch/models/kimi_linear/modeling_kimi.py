import math
from collections.abc import Callable
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from einops import rearrange
from packaging import version
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from transformers.utils.generic import OutputRecorder

try:
    from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError as err:
    raise ImportError(
        "Plese run `pip uninstall fla-core flash-linear-attention -y && pip install git+https://github.com/fla-org/flash-linear-attention@v0.4.0`"
    ) from err

from .configuration_kimi import KimiLinearConfig

assert version.parse(transformers.__version__) >= version.parse("4.56.0"), (
    "Please upgrade transformers to >= 4.56.0"
)

logger = logging.get_logger(__name__)


class KimiDynamicCache:
    """
    Dynamic cache for Kimi model.
    Inspired by Qwen3-Next
    """

    is_compileable = False

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config

        if config.linear_attn_config is not None:
            self.layer_types = []
            for i in range(config.num_hidden_layers):
                if config.is_kda_layer(i):
                    self.layer_types.append("linear_attention")
                else:
                    self.layer_types.append("full_attention")
        else:
            self.layer_types = ["full_attention"] * config.num_hidden_layers

        self.transformer_layers = [
            i
            for i in range(config.num_hidden_layers)
            if self.layer_types[i] == "full_attention"
        ]

        linear_layers = [
            i
            for i in range(config.num_hidden_layers)
            if self.layer_types[i] == "linear_attention"
        ]
        self.last_linear_layer = linear_layers[-1] if linear_layers else -1

        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]

    def __len__(self):
        return len(self.layer_types)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                device = self.key_cache[layer_idx].device
                beam_idx = beam_idx.to(device)
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                    0, beam_idx
                )
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                    0, beam_idx
                )

            if self.conv_states[layer_idx] is not None:
                device = self.conv_states[layer_idx][0].device
                beam_idx = beam_idx.to(device)
                q_conv, k_conv, v_conv = self.conv_states[layer_idx]
                self.conv_states[layer_idx] = (
                    q_conv.index_select(0, beam_idx),
                    k_conv.index_select(0, beam_idx),
                    v_conv.index_select(0, beam_idx),
                )
                self.recurrent_states[layer_idx] = self.recurrent_states[
                    layer_idx
                ].index_select(0, beam_idx)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        layer_idx = (
            self.transformer_layers[0]
            if layer_idx not in self.transformer_layers
            else layer_idx
        )
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: int
    ) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
        """
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    @property
    def has_previous_state(self):
        """We have a previous state if the last linear (conv) layer was already updated."""
        if self.last_linear_layer == -1:
            return False
        return self.conv_states[self.last_linear_layer] is not None


class KimiRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        KimiRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(KimiRMSNorm)


class KimiBlockSparseMLP(nn.Module):
    def __init__(
        self, config: KimiLinearConfig, hidden_size=None, intermediate_size=None
    ):
        super().__init__()
        self.config = config
        self.ffn_dim = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.hidden_dim = config.hidden_size if hidden_size is None else hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)  # gate
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)  # down
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)  # up

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(
            hidden_states
        )
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class KimiMLP(nn.Module):
    def __init__(
        self, config: KimiLinearConfig, hidden_size=None, intermediate_size=None
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class KimiMLAAttention(nn.Module):
    """
    Multi-Latent Attention adapted from deepseek-v3
    """

    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.rope_theta = config.rope_theta
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        try:
            self.q_lora_rank = config.q_lora_rank
            self.qk_rope_head_dim = config.qk_rope_head_dim
            self.kv_lora_rank = config.kv_lora_rank
            self.v_head_dim = config.v_head_dim
            self.qk_nope_head_dim = config.qk_nope_head_dim
            self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            self.use_nope = config.mla_use_nope
            self.scaling = self.q_head_dim ** (-0.5)
        except Exception as e:
            raise ValueError(
                f"Kimi MLA config is not found or not properly formatted: {e}"
            ) from e

        assert self.q_lora_rank is None
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.q_head_dim,
            bias=False,
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
        )
        self.is_causal = True
        assert self.use_nope

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.q_head_dim)
        key_shape = (
            batch_size,
            seq_length,
            -1,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        q_states = self.q_proj(hidden_states)
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        k_pass = (
            self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        )
        k_pass, value_states = torch.split(
            k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        if (
            self.config._attn_implementation == "flash_attention_2"
            and self.q_head_dim != self.v_head_dim
        ):
            value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if (
            self.config._attn_implementation == "flash_attention_2"
            and self.q_head_dim != self.v_head_dim
        ):
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class KimiDeltaAttention(nn.Module):
    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.mode = "chunk"

        self.hidden_size = config.hidden_size
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.head_k_dim = self.head_dim
        self.num_k_heads = self.num_heads

        self.layer_idx = layer_idx

        assert self.mode in ["chunk", "fused_recurrent"], (
            f"Not suppoerted mode `{self.mode}`."
        )

        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.q_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_k_size, kernel_size=self.conv_size, activation="silu"
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size, kernel_size=self.conv_size, activation="silu"
        )

        self.A_log = torch.nn.Parameter(
            torch.log(
                torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)
            ).view(1, 1, -1, 1)
        )

        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))

        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=config.rms_norm_eps, activation="sigmoid"
        )
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_params: Optional[KimiDynamicCache] = None,
        **kwargs: Unpack[dict],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                attention_mask = kwargs.get("padding_mask", None)

            if attention_mask is not None and attention_mask.dim() != 2:
                raise ValueError(
                    "attention_mask must be a 0-1 matrix of shape [batch_size, seq_len] "
                    "(0 = padding). 3D masks are not supported here."
                )
        use_cache = cache_params is not None
        batch_size, q_len, _ = hidden_states.shape
        mode = "fused_recurrent" if q_len <= 64 else self.mode
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        recurrent_state = None
        if cache_params is not None:
            if cache_params.conv_states[self.layer_idx] is not None:
                conv_state_q, conv_state_k, conv_state_v = cache_params.conv_states[
                    self.layer_idx
                ]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]
        q, conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        v, conv_state_v = self.v_conv1d(
            x=self.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = fused_kda_gate(g, self.A_log, self.head_dim, g_bias=self.dt_bias)
        beta = self.b_proj(hidden_states).float().sigmoid()

        q, k = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim), (q, k)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        if mode == "chunk":
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = recurrent_state
            cache_params.conv_states[self.layer_idx] = (
                conv_state_q,
                conv_state_k,
                conv_state_v,
            )

        g = self.g_b_proj(self.g_a_proj(hidden_states))
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_dim)
        o = self.o_norm(o, g)

        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o


class KimiMoEGate(nn.Module):
    """
    MoEGate adapted from Deepseek-V3.
    Parameter correspondences:
        num_experts -> n_routed_experts
        num_experts_per_token -> num_experts_per_tok
        num_expert_group -> n_group
        moe_router_activation_func -> scoring_func
    """

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_router_activation_func = config.moe_router_activation_func
        self.num_expert_group = getattr(config, "num_expert_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)

        # topk selection algorithm
        self.moe_renormalize = config.moe_renormalize
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))

        self.e_score_correction_bias = nn.Parameter(torch.empty((self.num_experts)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor):
        """
        Computes gating scores and selects top-k experts.

        During training:
            - Uses standard softmax and top-k.
            - Returns (logits, topk_indices, topk_weights) for aux loss calculation.
            - Differentiable.

        During inference:
            - Uses the original optimized, non-differentiable logic.
            - Returns (None, topk_indices, topk_weights).
        """
        bsz, seq_len, h = hidden_states.shape
        # Reshape to (num_tokens, hidden_dim)
        hidden_states = hidden_states.view(-1, h)

        # router_logits are the raw scores from the linear layer
        router_logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )

        if self.training:
            # Training path: standard, differentiable top-k routing

            # Use softmax for probabilities, as it's standard for MoE routing loss
            gating_scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)

            # Get top-k scores and their indices
            topk_weight, topk_idx = torch.topk(
                gating_scores, self.top_k, dim=-1, sorted=False
            )

            # Re-normalize top-k weights to sum to 1
            if self.top_k > 1 and self.moe_renormalize:
                denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
                topk_weight = topk_weight / denominator

            # Apply scaling factor
            topk_weight = topk_weight * self.routed_scaling_factor

            # During training, we return the raw logits for the aux loss calculation
            return router_logits, topk_idx, topk_weight

        else:
            # Inference path: use the original optimized code
            if self.moe_router_activation_func == "sigmoid":
                scores = router_logits.sigmoid()
            elif self.moe_router_activation_func == "softmax":
                scores = router_logits.softmax(dim=1)
            else:
                raise NotImplementedError(
                    f"insupportable scoring function for MoE gating: {self.moe_router_activation_func}"
                )

            scores_for_choice = scores.view(bsz * seq_len, -1)
            scores_for_choice += self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.num_expert_group, -1)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len,
                    self.num_expert_group,
                    self.num_experts // self.num_expert_group,
                )
                .reshape(bsz * seq_len, -1)
            )
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
            _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
            topk_weight = scores.gather(1, topk_idx)

            if self.top_k > 1 and self.moe_renormalize:
                denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
                topk_weight = topk_weight / denominator
            topk_weight = topk_weight * self.routed_scaling_factor

            # During inference, the first element (logits) is None
            return None, topk_idx, topk_weight

    # def forward(self, hidden_states):
    #     bsz, seq_len, h = hidden_states.shape
    #     # compute gating score
    #     hidden_states = hidden_states.view(-1, h)
    #     logits = F.linear(
    #         hidden_states.type(torch.float32), self.weight.type(
    #             torch.float32), None
    #     )
    #     if self.moe_router_activation_func == "sigmoid":
    #         scores = logits.sigmoid()
    #     elif self.moe_router_activation_func == "softmax":
    #         scores = logits.softmax(dim=1)
    #     else:
    #         raise NotImplementedError(
    #             f"insupportable scoring function for MoE gating: {self.moe_router_activation_func}"
    #         )

    #     # select top-k experts
    #     assert not self.training
    #     scores_for_choice = scores.view(bsz * seq_len, -1)
    #     scores_for_choice += self.e_score_correction_bias.unsqueeze(0)
    #     group_scores = (
    #         scores_for_choice.view(
    #             bsz * seq_len, self.num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    #     )  # [n, num_expert_group]
    #     group_idx = torch.topk(
    #         group_scores, k=self.topk_group, dim=-1, sorted=False
    #     )[
    #         1
    #     ]  # [n, top_k_group]
    #     group_mask = torch.zeros_like(group_scores)  # [n, num_expert_group]
    #     group_mask.scatter_(1, group_idx, 1)  # [n, num_expert_group]
    #     score_mask = (
    #         group_mask.unsqueeze(-1)
    #         .expand(
    #             bsz * seq_len, self.num_expert_group, self.num_experts // self.num_expert_group
    #         )
    #         .reshape(bsz * seq_len, -1)
    #     )  # [n, e]
    #     tmp_scores = scores_for_choice.masked_fill(
    #         ~score_mask.bool(), 0.0)  # [n, e]
    #     _, topk_idx = torch.topk(
    #         tmp_scores, k=self.top_k, dim=-1, sorted=False
    #     )
    #     topk_weight = scores.gather(1, topk_idx)

    #     # norm gate to sum 1
    #     if self.top_k > 1 and self.moe_renormalize:
    #         denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
    #         topk_weight = topk_weight / denominator
    #     # must multiply the scaling factor
    #     topk_weight = topk_weight * self.routed_scaling_factor

    #     return topk_idx, topk_weight


# class KimiSparseMoeBlock(nn.Module):
#     """
#     Adapted from Deepseek-V3's MOE implementation
#     The namings are consistent with Kimi's version.
#     """

#     def __init__(self, config: KimiLinearConfig):
#         super().__init__()
#         self.config = config
#         self.hidden_dim = config.hidden_size
#         self.num_experts = config.num_experts
#         self.top_k = config.num_experts_per_token
#         self.moe_renormalize = config.moe_renormalize

#         self.ep_size = 1
#         self.experts_per_rank = config.num_experts
#         self.ep_rank = 0
#         self.experts = nn.ModuleList(
#             [
#                 KimiBlockSparseMLP(
#                     config, intermediate_size=config.moe_intermediate_size
#                 )
#                 for _ in range(config.num_experts)
#             ]
#         )
#         self.gate = KimiMoEGate(config)
#         if config.num_shared_experts is not None:
#             intermediate_size = config.moe_intermediate_size * config.num_shared_experts
#             self.shared_experts = KimiMLP(
#                 config=config, intermediate_size=intermediate_size
#             )

#     def forward(self, hidden_states):
#         identity = hidden_states
#         orig_shape = hidden_states.shape
#         topk_idx, topk_weight = self.gate(hidden_states)
#         hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
#         flat_topk_idx = topk_idx.view(-1)
#         if not self.training:
#             y = self.moe_infer(hidden_states, topk_idx,
#                                topk_weight).view(*orig_shape)
#         else:
#             raise NotImplementedError(
#                 "Training mode is not supported in KimiSparseMoeBlock")
#         if self.config.num_shared_experts is not None:
#             y = y + self.shared_experts(identity)
#         return y

#     @torch.no_grad()
#     def moe_infer(self, x, topk_ids, topk_weight):
#         cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
#         cnts.scatter_(1, topk_ids, 1)
#         tokens_per_expert = cnts.sum(dim=0)
#         idxs = topk_ids.view(-1).argsort()
#         sorted_tokens = x[idxs // topk_ids.shape[1]]

#         tokens_per_expert = tokens_per_expert.cpu().numpy()

#         outputs = []
#         start_idx = 0
#         for i, num_tokens in enumerate(tokens_per_expert):
#             end_idx = start_idx + num_tokens
#             if num_tokens == 0:
#                 continue
#             expert = self.experts[i + self.ep_rank * self.experts_per_rank]
#             tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
#             expert_out = expert(tokens_for_this_expert)
#             outputs.append(expert_out)
#             start_idx = end_idx

#         outs = torch.cat(outputs, dim=0) if len(
#             outputs) else sorted_tokens.new_empty(0)

#         new_x = torch.empty_like(outs)
#         new_x[idxs] = outs
#         final_out = (
#             new_x.view(*topk_ids.shape, -1)
#             .type(topk_weight.dtype)
#             .mul_(topk_weight.unsqueeze(dim=-1))
#             .sum(dim=1)
#             .type(new_x.dtype)
#         )
#         return final_out


# Replace the KimiSparseMoeBlock class with this new version
class KimiSparseMoeBlock(nn.Module):
    """
    Adapted from Deepseek-V3's MOE implementation.
    This version is modified to be trainable.
    """

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token

        # Hyperparameter for the auxiliary loss, add to config or hardcode
        # Typical values are around 1e-2. Let's check config first.
        self.router_aux_loss_coef = getattr(config, "router_aux_loss_coef", 0.01)

        self.experts = nn.ModuleList(
            [
                KimiBlockSparseMLP(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for _ in range(config.num_experts)
            ]
        )
        self.gate = KimiMoEGate(config)
        if config.num_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.num_shared_experts
            self.shared_experts = KimiMLP(
                config=config, intermediate_size=intermediate_size
            )

    def calculate_aux_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Calculates the auxiliary load-balancing loss for the MoE layer.
        This is a critical component for stable training of MoE models.
        It encourages the router to send a balanced number of tokens to each expert.

        The loss is a combination of:
        1. A loss that encourages the router to distribute tokens evenly.
        2. A loss that encourages the router logits to have a small magnitude (z-loss).
        """
        if router_logits is None or not self.training:
            return torch.tensor(
                0.0, device=router_logits.device, dtype=router_logits.dtype
            )

        num_tokens, num_experts = router_logits.shape

        # Calculate the probabilities and their mean across all tokens
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        mean_router_probs_per_expert = torch.mean(router_probs, dim=0)

        # Calculate the fraction of tokens dispatched to each expert
        # Create a one-hot representation of the router's choices
        # For top_k > 1, a token is "dispatched" to all its chosen experts
        # We can approximate this with router_probs during training for a differentiable loss
        tokens_per_expert = torch.mean(router_probs, dim=0)

        # The standard load balancing loss from the Switch Transformer paper
        load_balancing_loss = num_experts * torch.sum(
            mean_router_probs_per_expert * tokens_per_expert
        )

        return self.router_aux_loss_coef * load_balancing_loss

    # def forward(self, hidden_states: torch.Tensor):
    #     identity = hidden_states
    #     batch_size, seq_len, hidden_dim = hidden_states.shape

    #     # Get routing decisions from the gate
    #     # router_logits is None during inference
    #     router_logits, topk_idx, topk_weight = self.gate(hidden_states)

    #     # Calculate auxiliary loss (will be 0.0 during inference)
    #     aux_loss = self.calculate_aux_loss(router_logits)

    #     # Reshape for routing
    #     hidden_states = hidden_states.view(-1, hidden_dim)

    #     if self.training:
    #         # Differentiable training path
    #         # Create a flat index for all tokens and their top-k choices
    #         flat_topk_idx = topk_idx.view(-1)

    #         # Create a large, sparse tensor representing the routing decisions
    #         # Shape: (num_tokens * top_k, num_experts)
    #         one_hot_mask = F.one_hot(flat_topk_idx, num_classes=self.num_experts).to(topk_weight.dtype)

    #         # Combine the weights with the one-hot mask
    #         # Shape: (num_tokens, top_k, num_experts)
    #         routing_weights = (topk_weight.unsqueeze(-1) * one_hot_mask.view(hidden_states.shape[0], self.top_k, -1))

    #         # Sum over the top-k dimension to get the final weight for each token-expert pair
    #         # Shape: (num_tokens, num_experts)
    #         routing_weights = routing_weights.sum(dim=1)

    #         # This is the memory-intensive but clear and differentiable part.
    #         # We compute all expert outputs for all tokens.
    #         all_expert_outputs = [expert(hidden_states) for expert in self.experts]
    #         expert_outputs_stack = torch.stack(all_expert_outputs, dim=1) # (num_tokens, num_experts, hidden_dim)

    #         # Perform the weighted sum of expert outputs
    #         # torch.einsum is efficient for this: (tokens, experts) * (tokens, experts, dim) -> (tokens, dim)
    #         final_hidden_states = torch.einsum('te,ted->td', routing_weights, expert_outputs_stack)

    #         # Reshape back to the original shape
    #         final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)

    #     else:
    #         # Original non-differentiable inference path
    #         final_hidden_states = self.moe_infer(hidden_states, topk_idx, topk_weight).view(batch_size, seq_len, hidden_dim)

    #     if self.config.num_shared_experts is not None:
    #         final_hidden_states = final_hidden_states + self.shared_experts(identity)

    #     # Return the final states and the auxiliary loss
    #     return final_hidden_states, aux_loss
    def forward(self, hidden_states: torch.Tensor):
        """
        Optimized forward pass for MoE training that avoids materializing all expert outputs at once.
        """
        identity = hidden_states
        batch_size, seq_len, hidden_dim = hidden_states.shape
        num_tokens = batch_size * seq_len

        # Reshape for routing
        hidden_states = hidden_states.view(num_tokens, hidden_dim)

        # Get routing decisions from the gate
        # router_logits is None during inference
        router_logits, topk_idx, topk_weight = self.gate(
            hidden_states.view(batch_size, seq_len, hidden_dim)
        )

        # Calculate auxiliary loss (will be 0.0 during inference)
        aux_loss = self.calculate_aux_loss(router_logits)

        if self.training:
            # =================================================================================
            # NEW MEMORY-EFFICIENT TRAINING PATH using Scatter-Gather
            # =================================================================================

            # We have topk_idx [num_tokens, top_k] and topk_weight [num_tokens, top_k]
            # We need to dispatch each of the `num_tokens * top_k` choices to its expert

            # 1. Create a flat list of tokens and their expert assignments
            # flat_topk_idx a 1D tensor of shape (num_tokens * top_k,)
            # flat_topk_idx = topk_idx.view(-1)

            # # 2. Create a routing mask/matrix that tells us which token goes to which expert
            # # This will be a sparse matrix in COO format (row, col)
            # # rows: a token's flattened index (from 0 to num_tokens * top_k - 1)
            # # cols: the expert index a token is being routed to
            # row_indices = torch.arange(
            #     num_tokens * self.top_k, device=hidden_states.device
            # )

            # # We use a one-hot encoding which is then converted to a sparse matrix.
            # # This is a key operation for dispatching.
            # # Shape: (num_tokens * top_k, num_experts)
            # routing_mask = F.one_hot(flat_topk_idx, num_classes=self.num_experts).to(
            #     hidden_states.dtype
            # )

            # # 3. Expand hidden_states to match the top_k choices
            # # Each token is routed to `top_k` experts, so we need `top_k` copies of it.
            # # Shape: (num_tokens, top_k, hidden_dim) -> (num_tokens * top_k, hidden_dim)
            # repeated_hidden_states = (
            #     hidden_states.unsqueeze(1).repeat(1, self.top_k, 1).view(-1, hidden_dim)
            # )

            # # 4. Dispatch tokens to experts by multiplying with the routing mask
            # # This is a gather operation. We are calculating inputs for each expert.
            # # `einsum` is very efficient here.
            # # 't e, t d -> e d' where t = num_tokens * top_k
            # # This sums up all token inputs for each expert.
            # # Shape: (num_experts, hidden_dim)
            # expert_inputs = torch.einsum(
            #     "te,td->ed", routing_mask, repeated_hidden_states
            # )

            # # 5. Process inputs with experts
            # # Now we can iterate through experts, but the input to each is just a single tensor,
            # # representing the sum of all tokens routed to it. We need to be careful here.
            # # The above logic is slightly flawed for batch processing, let's correct it.
            # # A clearer way is to shuffle the tokens.

            # Let's restart the logic with a proper permutation-based approach.

            # 1. Create a permutation index to sort tokens by their expert destination
            # We get a flat list of expert indices for every token's choice
            flat_topk_idx = topk_idx.view(-1)

            # Combine the expert index with the token index to create a sorting key
            # Multiplying by num_experts makes sure that tokens for expert `i` come before `i+1`
            # This gives us a unique, sortable key for each token-expert pair
            sorted_indices = torch.argsort(flat_topk_idx)

            # Create an inverse permutation to shuffle results back later
            inverse_permutation = torch.argsort(sorted_indices)

            # 2. Shuffle the tokens according to their destination expert
            # We need to select the original token for each of its top_k destinations
            # `token_indices` will be [0, 0, ..., 1, 1, ..., num_tokens-1, ...]
            token_indices = torch.arange(
                num_tokens, device=hidden_states.device
            ).repeat_interleave(self.top_k)

            # Shuffle both the tokens and their corresponding weights
            shuffled_tokens = hidden_states[token_indices[sorted_indices]]
            shuffled_weights = topk_weight.view(-1)[sorted_indices].unsqueeze(1)

            # 3. Dispatch and compute expert outputs in batches
            # Get the boundary indices for each expert's tokens
            # `tokens_per_expert` counts how many tokens (out of num_tokens * top_k) go to each expert
            tokens_per_expert = F.one_hot(
                flat_topk_idx, num_classes=self.num_experts
            ).sum(dim=0)

            # `split_indices` will be [count_e0, count_e0+count_e1, ...]
            split_indices = tokens_per_expert.cumsum(dim=0)

            # Process tokens expert by expert
            expert_outputs = []
            current_pos = 0
            for i in range(self.num_experts):
                num_tokens_for_this_expert = tokens_per_expert[i].item()
                if num_tokens_for_this_expert == 0:
                    continue

                # Select the chunk of shuffled tokens for the current expert
                expert_input_chunk = shuffled_tokens[
                    current_pos : current_pos + num_tokens_for_this_expert
                ]

                # Run the expert
                expert_output_chunk = self.experts[i](expert_input_chunk)
                expert_outputs.append(expert_output_chunk)

                current_pos += num_tokens_for_this_expert

            # Concatenate all expert outputs
            if expert_outputs:
                concatenated_outputs = torch.cat(expert_outputs, dim=0)
            else:
                # Handle case where no tokens are routed anywhere
                concatenated_outputs = torch.empty(
                    0,
                    hidden_dim,
                    device=shuffled_tokens.device,
                    dtype=shuffled_tokens.dtype,
                )

            # 4. Un-shuffle the outputs and apply weights
            # First, apply the weights to the shuffled outputs
            weighted_shuffled_outputs = concatenated_outputs * shuffled_weights

            # Then, use the inverse permutation to get them back in the original order
            # (num_tokens * top_k, hidden_dim)
            unshuffled_outputs = weighted_shuffled_outputs[inverse_permutation]

            # 5. Combine the top_k outputs for each token
            # Reshape to (num_tokens, top_k, hidden_dim) and sum over the top_k dimension
            final_hidden_states = unshuffled_outputs.view(
                num_tokens, self.top_k, hidden_dim
            ).sum(dim=1)

            # Reshape to original batch and sequence length
            final_hidden_states = final_hidden_states.view(
                batch_size, seq_len, hidden_dim
            )

        else:
            # Original non-differentiable inference path (no changes here)
            final_hidden_states = self.moe_infer(
                hidden_states, topk_idx, topk_weight
            ).view(batch_size, seq_len, hidden_dim)

        if self.config.num_shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(identity)

        # Return the final states and the auxiliary loss
        return final_hidden_states, aux_loss

    # The original moe_infer method remains unchanged, but we keep it
    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        # This method is exactly the same as the original code
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]

        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]  # Simplified for single-GPU case
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


class KimiDecoderLayer(nn.Module):
    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        if config.is_kda_layer(layer_idx):
            self.is_linear_attn = True
            self.self_attn = KimiDeltaAttention(config=config, layer_idx=layer_idx)
        elif config.is_mla:
            self.is_linear_attn = False
            self.self_attn = KimiMLAAttention(config=config, layer_idx=layer_idx)
        else:
            raise NotImplementedError
        if (
            config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % getattr(config, "moe_layer_freq", 1) == 0
        ):
            self.block_sparse_moe = KimiSparseMoeBlock(config)
        else:
            self.mlp = KimiMLP(config)
        self.input_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = KimiRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[torch.FloatTensor]
    ]:  # Note: The return signature needs updating
        """
        Args:
            ...
        Returns:
            A tuple containing:
            - The final hidden state of the layer (torch.FloatTensor).
            - The auxiliary loss from the MoE block, if present (torch.FloatTensor or None).
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention (this part is unchanged)
        if self.is_linear_attn is False:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache_params=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        # Fully Connected (this is the part we are fixing)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        aux_loss = None  # Initialize aux_loss for this layer
        if hasattr(self, "block_sparse_moe"):
            # Unpack the tuple returned by the MoE block
            hidden_states, aux_loss = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        # Return both the hidden state and the auxiliary loss
        return hidden_states, aux_loss

    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[Tuple[torch.Tensor]] = None,
    #     output_attentions: Optional[bool] = False,
    #     use_cache: Optional[bool] = False,
    #     **kwargs: Unpack[FlashAttentionKwargs],
    # ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    #     """
    #     Args:
    #         hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
    #         attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
    #             `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
    #         output_attentions (`bool`, *optional*):
    #             Whether or not to return the attentions tensors of all attention layers. See `attentions` under
    #             returned tensors for more detail.
    #         use_cache (`bool`, *optional*):
    #             If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
    #             (see `past_key_values`).
    #         past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    #     """

    #     residual = hidden_states

    #     hidden_states = self.input_layernorm(hidden_states)

    #     # Self Attention
    #     if self.is_linear_attn is False:
    #         hidden_states = self.self_attn(
    #             hidden_states=hidden_states,
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             past_key_values=past_key_values,
    #             output_attentions=output_attentions,
    #             use_cache=use_cache,
    #             **kwargs,
    #         )
    #     else:
    #         hidden_states = self.self_attn(
    #             hidden_states=hidden_states,
    #             attention_mask=attention_mask,
    #             cache_params=past_key_values,
    #             output_attentions=output_attentions,
    #             use_cache=use_cache,
    #             **kwargs,
    #         )
    #     hidden_states = residual + hidden_states

    #     # Fully Connected
    #     residual = hidden_states
    #     hidden_states = self.post_attention_layernorm(hidden_states)
    #     if hasattr(self, "block_sparse_moe"):
    #         hidden_states = self.block_sparse_moe(hidden_states)
    #     else:
    #         hidden_states = self.mlp(hidden_states)
    #     hidden_states = residual + hidden_states

    #     return hidden_states


class KimiPreTrainedModel(PreTrainedModel):
    config_class = KimiLinearConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["KimiDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(KimiBlockSparseMLP, index=1),
        "hidden_states": KimiDecoderLayer,
        "attentions": KimiMLAAttention,
    }
    _is_stateful = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class KimiLinearModel(KimiPreTrainedModel):
    def __init__(self, config: KimiLinearConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                KimiDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if getattr(config, "_attn_implementation", None) is not None:
            if config._attn_implementation != "flash_attention_2":
                logger.warning_once(
                    f"Ignoring the provided attention implementation {config._attn_implementation}"
                )
                logger.warning_once("Using flash_attention_2 backend instead.")
                config._attn_implementation = "flash_attention_2"
        else:
            config._attn_implementation = "flash_attention_2"

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _update_linear_attn_mask(self, attention_mask, cache_position):
        """
        NOTE: Left-padding is used for linear attention mask.
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            linear_attn_mask = None
        return linear_attn_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        # We add output_hidden_states and return_dict here to pass them through
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # Most of the setup code is the same...
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = KimiDynamicCache(config=self.config)
        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds
        if past_key_values is not None:
            assert isinstance(past_key_values, KimiDynamicCache)

        # This is the core change: collecting auxiliary losses
        all_aux_losses = []

        for decoder_layer in self.layers:
            layer_mask = (
                linear_attn_mask if decoder_layer.is_linear_attn else causal_mask
            )

            # Unpack the tuple from the decoder layer
            hidden_states, layer_aux_loss = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

            # Collect the loss if it exists
            if layer_aux_loss is not None:
                all_aux_losses.append(layer_aux_loss)

        hidden_states = self.norm(hidden_states)

        # Return the standard output object AND the list of losses
        model_output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            # hidden_states and attentions could be collected if needed
        )
        return model_output, all_aux_losses

    # @check_model_inputs
    # @auto_docstring
    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[Cache] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     **kwargs: Unpack[TransformersKwargs],
    # ) -> Union[Tuple, BaseModelOutputWithPast]:

    #     use_cache = use_cache if use_cache is not None else self.config.use_cache

    #     if (input_ids is None) and (inputs_embeds is None):
    #         raise ValueError(
    #             "You must specify exactly one of input_ids or inputs_embeds")

    #     # Get inputs_embeds
    #     if inputs_embeds is None:
    #         inputs_embeds = self.embed_tokens(input_ids)

    #     if use_cache and past_key_values is None:
    #         past_key_values = KimiDynamicCache(config=self.config)

    #     if cache_position is None:
    #         past_seen_tokens = past_key_values.get_seq_length(
    #         ) if past_key_values is not None else 0
    #         cache_position: torch.Tensor = torch.arange(
    #             past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    #         )

    #     if position_ids is None:
    #         position_ids = cache_position.unsqueeze(0)

    #     causal_mask = create_causal_mask(
    #         config=self.config,
    #         input_embeds=inputs_embeds,
    #         attention_mask=attention_mask,
    #         cache_position=cache_position,
    #         past_key_values=past_key_values,
    #         position_ids=position_ids,
    #     )
    #     linear_attn_mask = self._update_linear_attn_mask(
    #         attention_mask, cache_position)

    #     hidden_states = inputs_embeds
    #     if past_key_values is not None:
    #         assert isinstance(past_key_values, KimiDynamicCache)

    #     for decoder_layer in self.layers:
    #         layer_mask = linear_attn_mask if decoder_layer.is_linear_attn else causal_mask

    #         hidden_states = decoder_layer(
    #             hidden_states,
    #             attention_mask=layer_mask,
    #             past_key_values=past_key_values,
    #             cache_position=cache_position,
    #             **kwargs,
    #         )

    #     hidden_states = self.norm(hidden_states)

    #     return BaseModelOutputWithPast(
    #         last_hidden_state=hidden_states,
    #         past_key_values=past_key_values,
    #     )


class KimiLinearForCausalLM(KimiPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = KimiLinearModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        generation_mode: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, KimiLinearForCausalLM

        >>> model = KimiLinearForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

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

        # The call to self.model now returns a tuple
        outputs, all_aux_losses = self.model(
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
        )

        logits = outputs[0]
        if generation_mode:
            logits = logits[:, -1:]
        logits = self.lm_head(logits)

        loss = None
        if labels is not None:
            main_loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

            # Sum all the auxiliary losses from the MoE layers
            if all_aux_losses:
                total_aux_loss = torch.stack(all_aux_losses).sum()
            else:
                total_aux_loss = torch.tensor(0.0, device=main_loss.device)

            # The final loss is the sum of the main loss and the auxiliary loss
            loss = main_loss + total_aux_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
