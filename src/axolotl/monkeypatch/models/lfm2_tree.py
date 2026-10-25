"""The LFM2 short-conv tree component (LFM2, LFM2-MoE).

LFM2's recurrent layers are a gated short causal conv with no scan, so a tree
row only needs each segment's conv to start from the last inputs of its path:
every shared token runs once. The decoder layer hands its conv nothing but
``seq_idx``, so its patched forward passes the tree kwargs on for conv layers.
"""

from __future__ import annotations

import functools
import importlib

from axolotl.monkeypatch.models.recurrent_tree import (
    RecurrentComponent,
    TreeFamily,
    mark_tree_forward,
    register_tree_family,
    tree_conv,
    tree_kwargs,
    tree_plan,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# no recurrence to scan: only the conv's path history
SHORT_CONV = RecurrentComponent("short_conv", initial_state=False)

# model type -> (conv class, decoder layer class)
LFM2_LAYERS = {
    "lfm2": ("Lfm2ShortConv", "Lfm2DecoderLayer"),
    "lfm2_moe": ("Lfm2MoeShortConv", "Lfm2MoeDecoderLayer"),
}

for _model_type, (_conv, _) in LFM2_LAYERS.items():
    register_tree_family(TreeFamily((_model_type,), (_conv,), SHORT_CONV))


def short_conv_tree_forward(conv, module, hidden_states, plan):
    """LFM2's gated short conv on a tree row (``hidden_states`` [1, T, H])."""
    b, c, x = conv.in_proj(hidden_states).chunk(3, dim=-1)
    weight = conv.conv.weight.squeeze(1)

    def run(v, cu_seqlens):
        out = module.causal_conv1d_fn(v.transpose(1, 2), weight, conv.conv.bias)
        return out.transpose(1, 2)

    return conv.out_proj(c * tree_conv(run, b * x, plan, conv.conv_kernel_size))


def patch_lfm2_tree(model_type: str) -> bool:
    """Route ``model_type``'s short convs through the tree driver on tree rows."""
    if model_type not in LFM2_LAYERS:
        return False
    conv_name, layer_name = LFM2_LAYERS[model_type]
    module = importlib.import_module(
        f"transformers.models.{model_type}.modeling_{model_type}"
    )
    conv_cls, layer_cls = getattr(module, conv_name), getattr(module, layer_name)
    if getattr(conv_cls.forward, "_axolotl_recurrent_tree", False):
        return True
    original_conv, original_layer = conv_cls.forward, layer_cls.forward

    @functools.wraps(original_conv)
    def conv_forward(
        self,
        hidden_states,
        past_key_values=None,
        attention_mask=None,
        seq_idx=None,
        **kwargs,
    ):
        plan = tree_plan(kwargs) if past_key_values is None else None
        if plan is None:
            return original_conv(
                self, hidden_states, past_key_values, attention_mask, seq_idx
            )
        return short_conv_tree_forward(self, module, hidden_states, plan)

    @functools.wraps(original_layer)
    def layer_forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ):
        if self.is_attention_layer or tree_plan(kwargs) is None:
            return original_layer(
                self,
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
        residual = hidden_states
        hidden_states = self.conv(
            hidden_states=self.operator_norm(hidden_states),
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **tree_kwargs(kwargs),
        )
        hidden_states = hidden_states + residual
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))

    conv_cls.forward = mark_tree_forward(conv_forward)
    layer_cls.forward = layer_forward
    LOG.info(f"Routed {conv_name} through the recurrent tree driver")
    return True
