# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""LFM2 short-conv models on tree rows score every sample as it scores alone."""

import importlib

import pytest
import torch
from tree_recurrence import install, uninstall
from tree_testing import assert_tree_matches_alone, branching_sequences, pack

pytest.importorskip("transformers")

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


def build(model_type, gradient_checkpointing=False):
    prefix, config_kwargs = FAMILIES[model_type]
    modeling = importlib.import_module(
        f"transformers.models.{model_type}.modeling_{model_type}"
    )
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
    model = model_cls(config).float()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable({"use_reentrant": False})
    return modeling, model.train()


@pytest.mark.parametrize("model_type", list(FAMILIES))
@pytest.mark.parametrize(
    "gradient_checkpointing", [False, True], ids=["", "checkpointed"]
)
def test_tree_rows_match_each_sample_alone(
    monkeypatch, model_type, gradient_checkpointing
):
    modeling, model = build(model_type, gradient_checkpointing)
    convs = []
    causal_conv1d_fn = modeling.causal_conv1d_fn

    def conv(x, *args, **kwargs):
        convs.append(x.shape[-1])
        return causal_conv1d_fn(x, *args, **kwargs)

    monkeypatch.setattr(modeling, "causal_conv1d_fn", conv)
    support = install(model)
    assert support.mode == "tree", support.reason
    sequences = branching_sequences()
    assert_tree_matches_alone(model, sequences)
    # the tree's convs ran over each segment with its path history, not the row
    row_tokens = len(pack(sequences).input_ids)
    assert any(n > row_tokens for n in convs)


def test_chains_and_collation_padding():
    _, model = build("lfm2")
    install(model)
    assert_tree_matches_alone(model, branching_sequences(), share=False)
    assert_tree_matches_alone(model, branching_sequences(), collation_padding=3)


def test_layer_forward_is_restored():
    modeling, model = build("lfm2")
    forward = modeling.Lfm2DecoderLayer.forward
    install(model)
    assert modeling.Lfm2DecoderLayer.forward is not forward
    assert_tree_matches_alone(model, branching_sequences())
    assert not any("forward" in vars(m) for m in model.modules())
    uninstall(model)
    assert modeling.Lfm2DecoderLayer.forward is forward


def test_without_hooks_packed_samples_leak():
    _, model = build("lfm2")
    with pytest.raises(AssertionError):
        assert_tree_matches_alone(model, branching_sequences())
