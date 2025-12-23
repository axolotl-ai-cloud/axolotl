"""
Adapted Kimi-Linear modeling to enable MoE differentiable.

Source: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/modeling_kimi.py
Revision: 6e163f3
"""

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
    MoeCausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    TransformersKwargs,
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

from axolotl.monkeypatch.models.kimi_linear.configuration_kimi import KimiLinearConfig

assert version.parse(transformers.__version__) >= version.parse("4.56.0"), (
    "Please upgrade transformers to >= 4.56.0"
)

logger = logging.get_logger(__name__)


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    """Standard Switch Transformer load balancing loss."""
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    # Concatenate all layer logits
    concatenated_gate_logits = torch.cat(
        [layer_gate for layer_gate in gate_logits], dim=0
    )

    routing_weights = F.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = F.one_hot(selected_experts, num_experts)

    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


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
    MoE Gate that returns router logits.
    Routing decisions are made in KimiSparseMoeBlock.
    """

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.gating_dim = config.hidden_size

        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        self.e_score_correction_bias = nn.Parameter(torch.zeros((self.num_experts,)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]

        Returns:
            router_logits: [batch_size * seq_len, num_experts]
        """
        _, _, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        router_logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        return router_logits

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
    MoE block adapted from Deepseek-V3.
    Returns only hidden_states - router_logits captured by OutputRecorder.
    """

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.moe_renormalize = config.moe_renormalize
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_expert_group = getattr(config, "num_expert_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)

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

    def route_tokens_to_experts(
        self,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing decisions from router logits.

        Args:
            router_logits: [num_tokens, num_experts]

        Returns:
            topk_idx: [num_tokens, top_k]
            topk_weight: [num_tokens, top_k]
        """
        num_tokens = router_logits.shape[0]

        if self.training:
            # Training: use softmax for standard aux loss compatibility
            scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1, sorted=False)
        else:
            # Inference: use original sigmoid + group selection
            scores = router_logits.sigmoid()
            scores_for_choice = scores + self.gate.e_score_correction_bias.unsqueeze(0)

            # Group-based selection
            group_scores = (
                scores_for_choice.view(num_tokens, self.num_expert_group, -1)
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
                    num_tokens,
                    self.num_expert_group,
                    self.num_experts // self.num_expert_group,
                )
                .reshape(num_tokens, -1)
            )
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
            _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
            topk_weight = scores.gather(1, topk_idx)

        # Normalize and scale
        if self.top_k > 1 and self.moe_renormalize:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only hidden_states.
        Router logits are captured by OutputRecorder for aux loss.
        """
        identity = hidden_states
        batch_size, seq_len, hidden_dim = hidden_states.shape
        num_tokens = batch_size * seq_len

        # Flatten for routing
        hidden_states_flat = hidden_states.view(num_tokens, hidden_dim)

        # Get router logits - OutputRecorder captures this!
        router_logits = self.gate(hidden_states)

        # Get routing decisions
        topk_idx, topk_weight = self.route_tokens_to_experts(router_logits)

        if self.training:
            final_hidden_states = self._training_forward(
                hidden_states_flat, topk_idx, topk_weight, num_tokens, hidden_dim
            )
        else:
            final_hidden_states = self._inference_forward(
                hidden_states_flat, topk_idx, topk_weight
            )

        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)

        # Add shared experts if present
        if self.config.num_shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(identity)

        return final_hidden_states

    def _training_forward(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
        num_tokens: int,
        hidden_dim: int,
    ) -> torch.Tensor:
        """
        Differentiable training forward using scatter-gather pattern.
        """
        # Flatten expert indices: [num_tokens * top_k]
        flat_topk_idx = topk_idx.view(-1)

        # Sort by expert index to group tokens going to same expert
        sorted_indices = torch.argsort(flat_topk_idx)
        inverse_permutation = torch.argsort(sorted_indices)

        # Each token appears top_k times (once per expert choice)
        token_indices = torch.arange(
            num_tokens, device=hidden_states.device
        ).repeat_interleave(self.top_k)

        # Gather tokens and weights in sorted order
        shuffled_tokens = hidden_states[token_indices[sorted_indices]]
        shuffled_weights = topk_weight.view(-1)[sorted_indices].unsqueeze(-1)

        # Count tokens per expert
        tokens_per_expert = F.one_hot(flat_topk_idx, num_classes=self.num_experts).sum(
            dim=0
        )

        # Process each expert's batch
        expert_outputs = []
        current_pos = 0
        for i in range(self.num_experts):
            num_tokens_for_expert = tokens_per_expert[i].item()
            if num_tokens_for_expert == 0:
                continue

            expert_input = shuffled_tokens[
                current_pos : current_pos + num_tokens_for_expert
            ]
            expert_output = self.experts[i](expert_input)
            expert_outputs.append(expert_output)
            current_pos += num_tokens_for_expert

        # Concatenate all outputs
        if expert_outputs:
            concatenated_outputs = torch.cat(expert_outputs, dim=0)
        else:
            concatenated_outputs = torch.zeros(
                num_tokens * self.top_k,
                hidden_dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        # Apply weights while still in sorted order
        weighted_outputs = concatenated_outputs * shuffled_weights

        # Unsort back to original token order
        unshuffled_outputs = weighted_outputs[inverse_permutation]

        # Sum contributions from all top_k experts for each token
        final_hidden_states = unshuffled_outputs.view(
            num_tokens, self.top_k, hidden_dim
        ).sum(dim=1)

        return final_hidden_states

    @torch.no_grad()
    def _inference_forward(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optimized inference forward (original implementation).
        """
        cnts = topk_idx.new_zeros((topk_idx.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = hidden_states[idxs // topk_idx.shape[1]]

        tokens_per_expert_list = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert_list):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            tokens_for_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_idx.shape, -1)
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
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
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

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class KimiPreTrainedModel(PreTrainedModel):
    config_class = KimiLinearConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["KimiDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(KimiMoEGate, index=0),
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        # Get inputs_embeds
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

        for decoder_layer in self.layers:
            layer_mask = (
                linear_attn_mask if decoder_layer.is_linear_attn else causal_mask
            )

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


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
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        generation_mode: Optional[bool] = None,
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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        logits = outputs[0]
        if generation_mode:
            logits = logits[:, -1:]
        logits = self.lm_head(logits)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if kwargs.get("output_router_logits", False):
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                num_experts=self.config.num_experts,
                top_k=self.config.num_experts_per_token,
                attention_mask=attention_mask,
            )
            if loss is not None:
                loss = loss + self.config.router_aux_loss_coef * aux_loss

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
