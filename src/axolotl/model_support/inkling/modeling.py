"""Sample packing for Inkling.

Every layer runs four short causal convs (k/v inside attention, and after the
attention and the MLP) that the stock forward never hands a document boundary,
and the full-attention log scaling reads the row offset instead of each token's
position in its document. Packed, uncached calls carry ``seq_idx`` and
``position_ids`` down to both; every other call keeps the stock forward.
"""

import functools
import importlib

import torch
import torch.nn.functional as F
from transformers.integrations.accelerate import force_accelerate_hooks

from axolotl.monkeypatch.models.mamba_utils import get_seq_idx
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

MODULE_NAME = "transformers.models.inkling.modeling_inkling"
SEQ_IDX_KEY = "seq_idx"
POSITION_IDS_KEY = "packed_position_ids"


def segmented_causal_conv(x, weight, bias, seq_idx):
    """Depthwise causal conv over ``[B, C, T]`` that never reads across ``seq_idx`` boundaries."""
    kernel_size = weight.shape[-1]
    out = x * weight[:, -1:]
    for shift in range(1, min(kernel_size, x.shape[-1])):
        same_doc = (seq_idx[:, shift:] == seq_idx[:, :-shift])[:, None]
        shifted = F.pad(x[..., :-shift] * same_doc, (shift, 0))
        out = out + shifted * weight[:, kernel_size - 1 - shift, None]
    return out if bias is None else out + bias[:, None]


def _make_model_forward(original):
    @functools.wraps(original)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        if position_ids is not None and past_key_values is None:
            seq_idx = get_seq_idx(position_ids)
            if not use_cache:
                kwargs[SEQ_IDX_KEY] = seq_idx
                kwargs[POSITION_IDS_KEY] = position_ids
                use_cache = False
            elif seq_idx[:, -1].any():
                raise ValueError(
                    "Inkling cannot build a cache from packed sequences; "
                    "pass use_cache=False"
                )
        return original(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

    return forward


def _make_layer_forward(original):
    @functools.wraps(original)
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        conv_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        seq_idx = kwargs.get(SEQ_IDX_KEY)
        if seq_idx is None:
            return original(
                self,
                hidden_states,
                attention_mask,
                conv_mask,
                past_key_values,
                **kwargs,
            )
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            conv_mask=conv_mask,
            **kwargs,
        )
        hidden_states = residual + self.attn_sconv(
            hidden_states, conv_mask=conv_mask, seq_idx=seq_idx
        )
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + self.mlp_sconv(
            hidden_states, conv_mask=conv_mask, seq_idx=seq_idx
        )

    return forward


def _make_attention_forward(modeling, original):
    @functools.wraps(original)
    def forward(
        self,
        hidden_states,
        attention_mask,
        conv_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        position_ids = kwargs.pop(POSITION_IDS_KEY, None)
        if position_ids is None:
            return original(
                self,
                hidden_states,
                attention_mask,
                conv_mask,
                past_key_values,
                **kwargs,
            )
        seq_idx = kwargs.pop(SEQ_IDX_KEY)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        key = self.k_sconv(
            self.k_proj(hidden_states), conv_mask=conv_mask, seq_idx=seq_idx
        )
        value = self.v_sconv(
            self.v_proj(hidden_states), conv_mask=conv_mask, seq_idx=seq_idx
        )
        query = query.transpose(1, 2)
        key = self.k_norm(key.view(hidden_shape)).transpose(1, 2)
        value = value.view(hidden_shape).transpose(1, 2)

        positions = torch.arange(input_shape[1], device=hidden_states.device)
        relative_states = self.r_proj(hidden_states).view(
            *input_shape, self.num_heads, -1
        )
        position_bias = self.rel_logits_proj(relative_states, positions, positions)

        if not self.is_sliding and self.config.log_scaling_n_floor is not None:
            effective_n = (position_ids + 1).float()
            tau = 1.0 + self.config.log_scaling_alpha * torch.log(
                (effective_n / self.config.log_scaling_n_floor).clamp(min=1.0)
            )
            tau = tau[:, None, :, None]
            query = (query.float() * tau).to(query.dtype)
            position_bias = (position_bias.float() * tau).to(position_bias.dtype)

        attention_interface = modeling.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, modeling.eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            position_bias=position_bias,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights

    return forward


@force_accelerate_hooks("conv1d")
def _packed_short_conv(self, modeling, hidden_states, conv_mask, seq_idx):
    input_dtype = hidden_states.dtype
    residual = hidden_states.float()
    if conv_mask is not None:
        # multipack masks carry one id per document
        conv_mask = conv_mask != 0
    hidden_states = modeling.apply_mask_to_padding_states(residual, conv_mask)
    hidden_states = hidden_states.transpose(1, 2)
    weight = self.conv1d.weight.squeeze(1)
    hidden_states = segmented_causal_conv(
        hidden_states.to(weight.dtype), weight, self.conv1d.bias, seq_idx
    )
    hidden_states = hidden_states.to(residual.dtype).transpose(1, 2)
    return (hidden_states + residual).to(input_dtype)


def _make_short_conv_forward(modeling, original):
    @functools.wraps(original)
    def forward(self, hidden_states, past_key_values=None, conv_mask=None, **kwargs):
        seq_idx = kwargs.get(SEQ_IDX_KEY)
        if seq_idx is None or past_key_values is not None:
            return original(self, hidden_states, past_key_values, conv_mask, **kwargs)
        return _packed_short_conv(self, modeling, hidden_states, conv_mask, seq_idx)

    return forward


def patch_inkling_modeling_packing() -> None:
    modeling = importlib.import_module(MODULE_NAME)
    if getattr(modeling.InklingTextModel.forward, "_axolotl_packing_patch", False):
        return
    modeling.InklingShortConvolution.forward = _make_short_conv_forward(
        modeling, modeling.InklingShortConvolution.forward
    )
    modeling.InklingAttention.forward = _make_attention_forward(
        modeling, modeling.InklingAttention.forward
    )
    modeling.InklingDecoderLayer.forward = _make_layer_forward(
        modeling.InklingDecoderLayer.forward
    )
    modeling.InklingTextModel.forward = _make_model_forward(
        modeling.InklingTextModel.forward
    )
    modeling.InklingTextModel.forward._axolotl_packing_patch = True
    LOG.info("Applied inkling sample packing patch")
