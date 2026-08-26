"""Bailing MoE V3 (Ling 3.0) modeling, adapted for training.

Source: https://huggingface.co/inclusionAI/Ling-3.0-flash/blob/main/modeling_bailing_moe_v3.py
Revision: 42766a8

The published file is inference-only: it hardcodes the eager attention
interface (so ``flash_attention_2`` silently trains without a causal mask),
builds masks with the pre-v5 helpers, and drops the padding mask before the
linear-attention kernels. Everything that transformers or ``fla`` already
implements is reused from there instead; what remains below is the Bailing
architecture itself.
"""

import copy

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Experts,
    DeepseekV3MLP,
    DeepseekV3MoE,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    DeepseekV3TopkRouter,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
    yarn_get_mscale,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import is_flash_attention_requested
from transformers.utils.output_capturing import capture_outputs

from axolotl.model_support.bailing_hybrid.configuration_bailing_moe_v3 import (
    BailingMoeV3Config,
)
from axolotl.monkeypatch.utils import get_unpad_data
from axolotl.utils.logging import get_logger

try:
    from fla.layers.utils import index_first_axis, pad_input
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError as err:
    raise ImportError("Please run `pip install fla-core==0.4.1`") from err

LOG = get_logger(__name__)


def cu_seqlens_from_position_ids(
    position_ids: torch.Tensor | None,
) -> torch.Tensor | None:
    """Flat ``[N + 1]`` document offsets for fla's varlen API, or None if unpacked.

    Axolotl removes the packed attention mask whenever sample packing is on
    (``sample_packing_drop_attention_mask``), so a ``position_ids`` restarting at 0
    is the only remaining signal for a document boundary.
    """
    if position_ids is None or position_ids.shape[0] != 1:
        return None
    starts = (position_ids[0] == 0).nonzero().flatten()
    # under context parallelism a chunk can begin mid-document, with no leading 0
    if starts.numel() == 0 or starts[0] != 0:
        starts = torch.cat([starts.new_zeros(1), starts])
    if starts.numel() < 2:
        return None
    total = starts.new_tensor([position_ids.shape[1]])
    return torch.cat([starts, total]).to(torch.int32)


class BailingMoeV3RotaryEmbedding(DeepseekV3RotaryEmbedding):
    def __init__(self, config: BailingMoeV3Config, device=None):
        rope_config = copy.deepcopy(config)
        # `head_dim` sizes the linear-attention heads; rope only covers the MLA rope split
        rope_config.head_dim = config.qk_rope_head_dim
        rope_config.rope_parameters = {
            **config.rope_parameters,
            "partial_rotary_factor": 1.0,
        }
        super().__init__(rope_config, device=device)


class BailingMoeV3MultiLatentAttention(nn.Module):
    """Multi-latent attention with a head-wise output gate."""

    def __init__(self, config: BailingMoeV3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.attention_dropout = config.attention_dropout
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_head_dim = config.qk_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.gate_granularity = config.gated_attention_proj_granularity_type
        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                config.hidden_size, self.num_heads * self.qk_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                config.hidden_size, self.q_lora_rank, bias=config.use_qkv_bias
            )
            self.q_a_layernorm = DeepseekV3RMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.use_qkv_bias,
        )
        self.kv_a_layernorm = DeepseekV3RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        if self.gate_granularity is None:
            self.g_proj = None
        elif self.gate_granularity == "head_wise":
            self.g_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        elif self.gate_granularity == "element_wise":
            self.g_proj = nn.Linear(
                config.hidden_size, self.num_heads * self.v_head_dim, bias=False
            )
        else:
            raise ValueError(
                f"Unknown attention gate granularity {self.gate_granularity}"
            )
        self.dense = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.use_qkv_bias,
        )

        self.scaling = self.qk_head_dim ** (-0.5)
        rope_parameters = config.rope_parameters
        if rope_parameters.get(
            "rope_type", "default"
        ) != "default" and rope_parameters.get("mscale_all_dim", 0):
            mscale = yarn_get_mscale(
                rope_parameters["factor"], rope_parameters["mscale_all_dim"]
            )
            self.scaling = self.scaling * mscale * mscale

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (
            batch_size,
            seq_length,
            -1,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
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

        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        pad_head_dim = (
            is_flash_attention_requested(self.config)
            and self.qk_head_dim != self.v_head_dim
        )
        if pad_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
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
        if pad_head_dim:
            attn_output = attn_output[..., : self.v_head_dim]

        if self.g_proj is not None:
            gate = F.sigmoid(self.g_proj(hidden_states).float()).type_as(attn_output)
            if self.gate_granularity == "head_wise":
                attn_output = attn_output * gate[:, :, :, None]
            else:
                attn_output = attn_output * gate.view(
                    batch_size, seq_length, self.num_heads, self.v_head_dim
                )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        return self.dense(attn_output)


class BailingMoeV3KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention: gated linear attention over short-convolved q/k/v."""

    def __init__(self, config: BailingMoeV3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.lower_bound = config.kda_lower_bound if config.kda_safe_gate else None
        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        conv_kwargs = dict(kernel_size=config.short_conv_kernel_size, activation="silu")
        self.q_conv1d = ShortConvolution(hidden_size=projection_size, **conv_kwargs)
        self.k_conv1d = ShortConvolution(hidden_size=projection_size, **conv_kwargs)
        self.v_conv1d = ShortConvolution(hidden_size=projection_size, **conv_kwargs)

        self.A_log = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))
        self.b_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        if config.no_kda_lora:
            self.f_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
            self.g_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        else:
            self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
            self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
            self.g_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=config.rms_norm_eps, activation="sigmoid"
        )
        self.o_proj = nn.Linear(projection_size, config.hidden_size, bias=False)

    def _forget_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.no_kda_lora:
            gate = self.f_proj(hidden_states)
        else:
            gate = self.f_b_proj(self.f_a_proj(hidden_states))
        gate = rearrange(gate, "... (h d) -> ... h d", d=self.head_dim)
        if self.lower_bound is None:
            return fused_kda_gate(gate, self.A_log, self.dt_bias)
        # `safe_gate` swaps the activation out for one bounded by construction rather
        # than clamping the softplus one; fla 0.4.1 has no kernel for it, so mirror
        # `naive_kda_lowerbound_gate` elementwise here
        gate = gate.float() + self.dt_bias.view(self.num_heads, -1)
        return self.lower_bound * torch.sigmoid(
            self.A_log.view(-1, 1).float().exp() * gate
        )

    def _output_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.no_kda_lora:
            gate = self.g_proj(hidden_states)
        else:
            gate = self.g_b_proj(self.g_a_proj(hidden_states))
        return rearrange(gate, "... (h d) -> ... h d", d=self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        batch_size, q_len, _ = hidden_states.shape
        cache_layer = (
            None if past_key_values is None else past_key_values.layers[self.layer_idx]
        )
        recurrent_state, conv_states = None, (None, None, None)
        if cache_layer is not None and cache_layer.is_conv_states_initialized[0]:
            recurrent_state = cache_layer.recurrent_states[0]
            conv_states = tuple(cache_layer.conv_states[idx] for idx in range(3))

        cu_seqlens, indices = kwargs.get("cu_seqlens"), None
        if attention_mask is not None:
            # multipack encodes one id per packed document, so unpadding also splits documents
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)
        elif cu_seqlens is None:
            cu_seqlens = cu_seqlens_from_position_ids(position_ids)

        projections = (
            (self.q_proj, self.q_conv1d),
            (self.k_proj, self.k_conv1d),
            (self.v_proj, self.v_conv1d),
        )
        projected = [
            conv(
                x=proj(hidden_states),
                cache=conv_states[idx],
                output_final_state=cache_layer is not None,
                cu_seqlens=cu_seqlens,
            )
            for idx, (proj, conv) in enumerate(projections)
        ]
        (q, conv_q), (k, conv_k), (v, conv_v) = projected
        q, k, v = (
            rearrange(x, "... (h d) -> ... h d", d=self.head_dim) for x in (q, k, v)
        )
        beta = self.b_proj(hidden_states).float().sigmoid()

        kernel = chunk_kda if self.training or q_len > 64 else fused_recurrent_kda
        out, recurrent_state = kernel(
            q=q,
            k=k,
            v=v,
            g=self._forget_gate(hidden_states),
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_layer is not None,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        if cache_layer is not None:
            for idx, conv_state in enumerate((conv_q, conv_k, conv_v)):
                cache_layer.update_conv_state(conv_state, state_idx=idx)
            cache_layer.update_recurrent_state(recurrent_state)

        out = self.o_norm(out, self._output_gate(hidden_states))
        out = self.o_proj(rearrange(out, "b t h d -> b t (h d)"))
        if indices is not None:
            out = pad_input(out.squeeze(0), indices, batch_size, q_len)
        return out


class BailingMoeV3SparseMoeBlock(DeepseekV3MoE):
    def __init__(self, config: BailingMoeV3Config):
        super().__init__(config)
        self.shared_experts = DeepseekV3MLP(
            config,
            intermediate_size=config.moe_shared_expert_intermediate_size
            * config.num_shared_experts,
        )


class BailingMoeV3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: BailingMoeV3Config, layer_idx: int):
        super().__init__()
        self.is_linear_attention = config.is_linear_attention_layer(layer_idx)
        attention_cls = (
            BailingMoeV3KimiDeltaAttention
            if self.is_linear_attention
            else BailingMoeV3MultiLatentAttention
        )
        self.attention = attention_cls(config, layer_idx)
        self.mlp = (
            BailingMoeV3SparseMoeBlock(config)
            if layer_idx >= config.first_k_dense_replace
            else DeepseekV3MLP(config)
        )
        self.input_layernorm = DeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        rope = (
            {}
            if self.is_linear_attention
            else {"position_embeddings": position_embeddings}
        )
        residual = hidden_states
        hidden_states = residual + self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **rope,
            **kwargs,
        )

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class BailingMoeV3PreTrainedModel(PreTrainedModel):
    config: BailingMoeV3Config
    config_class = BailingMoeV3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BailingMoeV3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _is_stateful = True
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _can_record_outputs = {"hidden_states": BailingMoeV3DecoderLayer}

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, DeepseekV3TopkRouter):
            module.weight.normal_(mean=0.0, std=std)
            module.e_score_correction_bias.zero_()
        elif isinstance(module, DeepseekV3Experts):
            module.gate_up_proj.normal_(mean=0.0, std=std)
            module.down_proj.normal_(mean=0.0, std=std)
        elif isinstance(module, BailingMoeV3KimiDeltaAttention):
            module.A_log.copy_(torch.empty_like(module.A_log).uniform_(1, 16).log())
            module.dt_bias.zero_()


class BailingMoeV3Model(BailingMoeV3PreTrainedModel):
    def __init__(self, config: BailingMoeV3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            BailingMoeV3DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BailingMoeV3RotaryEmbedding(config)
        self.gradient_checkpointing = False
        if config.num_nextn_predict_layers:
            LOG.warning(
                "Multi-token-prediction layers are not built for training; their checkpoint "
                "weights are ignored and will be missing from a full-parameter save."
            )
            mtp_layers = "|".join(
                str(config.num_hidden_layers + offset)
                for offset in range(config.num_nextn_predict_layers)
            )
            self._keys_to_ignore_on_load_unexpected = [rf"layers\.({mtp_layers})\."]
        _warn_on_unsupported_swiglu_limits(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + past_seen_tokens
            ).unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        # the linear-attention kernels take the raw 2D mask, not a causal one
        linear_mask = (
            attention_mask
            if attention_mask is not None and attention_mask.dim() == 2
            else None
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=linear_mask
                if decoder_layer.is_linear_attention
                else causal_mask,
                position_embeddings=position_embeddings,
                # both branches isolate packed documents off position_ids: the MLA
                # layers through transformers' varlen path, the KDA layers through
                # the cu_seqlens handed to the fla kernels
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        return BaseModelOutputWithPast(
            last_hidden_state=self.norm(hidden_states), past_key_values=past_key_values
        )


class BailingMoeV3ForCausalLM(BailingMoeV3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.word_embeddings.weight"}

    def __init__(self, config: BailingMoeV3Config):
        super().__init__(config)
        self.model = BailingMoeV3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.word_embeddings

    def set_input_embeddings(self, value):
        self.model.word_embeddings = value

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def _warn_on_unsupported_swiglu_limits(config: BailingMoeV3Config) -> None:
    """Ling-3.0-flash caps the SwiGLU activation on its last few layers.

    The published modeling code never reads these keys, so neither does this port,
    but the official serving stack may - warn rather than diverge silently.
    """
    limits = {
        key: getattr(config, key, None)
        for key in ("expert_swiglu_limit_list", "share_expert_swiglu_limit_list")
    }
    capped = {key: value for key, value in limits.items() if value and any(value)}
    if capped:
        LOG.warning(
            "%s set on this checkpoint but not applied: the published modeling code "
            "ignores these keys too, so training matches the reference implementation. "
            "Verify against the serving stack before relying on the result.",
            ", ".join(sorted(capped)),
        )


__all__ = [
    "BailingMoeV3ForCausalLM",
    "BailingMoeV3Model",
    "BailingMoeV3PreTrainedModel",
    "BailingMoeV3SparseMoeBlock",
]
