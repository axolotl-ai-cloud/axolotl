"""LFM2 short-conv models on tree rows score every sample as it scores alone."""

import importlib

import pytest
import torch
from trl_async_test_utils import (
    assert_tree_rows_match_alone,
    branching_samples,
    reference_tree_attention,
)

from axolotl.core.trainers.grpo.trl_async_packing import shares_prefixes

LAYERS = {"layer_types": ["conv", "full_attention", "conv"], "conv_L_cache": 3}
FAMILIES = {
    "lfm2": ("Lfm2", {"block_auto_adjust_ff_dim": False, **LAYERS}),
    "lfm2_moe": (
        "Lfm2Moe",
        {
            "moe_intermediate_size": 32,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "num_dense_layers": 1,
            **LAYERS,
        },
    ),
}


@pytest.fixture
def lfm2_family(request, monkeypatch):
    from trl.experimental.async_grpo import packing

    from axolotl.monkeypatch.models.lfm2_tree import patch_lfm2_tree
    from axolotl.monkeypatch.models.mamba_utils import patch_model_forward_seq_idx

    model_type = request.param
    prefix, config_kwargs = FAMILIES[model_type]
    modeling = importlib.import_module(
        f"transformers.models.{model_type}.modeling_{model_type}"
    )
    monkeypatch.setattr(packing, "tree_attention", reference_tree_attention)
    for name in ("Model", "ShortConv", "DecoderLayer"):
        cls = getattr(modeling, f"{prefix}{name}")
        monkeypatch.setattr(cls, "forward", cls.forward)
    convs = []
    causal_conv1d_fn = modeling.causal_conv1d_fn

    def conv(x, *args, **kwargs):
        convs.append(x.shape[-1])
        return causal_conv1d_fn(x, *args, **kwargs)

    monkeypatch.setattr(modeling, "causal_conv1d_fn", conv)
    patch_model_forward_seq_idx(getattr(modeling, f"{prefix}Model"))
    assert patch_lfm2_tree(model_type)

    model_cls = getattr(modeling, f"{prefix}ForCausalLM")
    config = model_cls.config_class(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        **config_kwargs,
    )
    torch.manual_seed(0)
    return model_cls(config).float(), convs


@pytest.mark.parametrize("lfm2_family", list(FAMILIES), indirect=True)
def test_tree_rows_match_each_sample_alone(lfm2_family):
    model, convs = lfm2_family
    assert shares_prefixes(model)
    rows = []

    def on_row(row_tokens):
        rows.append((row_tokens, list(convs)))
        convs.clear()

    assert_tree_rows_match_alone(model, branching_samples(), on_row)
    # the trie's convs ran with each segment's path history, not over the row
    shared_len, shared_convs = rows[0]
    assert shared_convs and all(n > shared_len for n in shared_convs)
