# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Mamba2 models on tree rows score every sample as it scores alone: through
transformers' PyTorch reference (unfused, every path as one padded batch) and through
kernels that reset at ``seq_idx`` (the fused mixer over every path in one call)."""

import importlib

import pytest
import torch
from tree_recurrence import install, reference, uninstall
from tree_recurrence.integrations import transformers as integration
from tree_testing import assert_tree_matches_alone, branching_sequences, pack

pytest.importorskip("transformers")

MAMBA2 = {"mamba_n_heads": 4, "mamba_d_head": 16, "mamba_n_groups": 2}
MAMBA2.update(mamba_d_state=8, mamba_d_conv=4, mamba_expand=2, mamba_chunk_size=8)
ATTENTION = {"num_attention_heads": 2, "num_key_value_heads": 1}
# model type -> (causal LM, config kwargs)
FAMILIES = {
    "nemotron_h": (
        "NemotronHForCausalLM",
        {
            "layers_block_type": [
                "linear_attention",
                "full_attention",
                "linear_attention",
                "mlp",
            ],
            "head_dim": 16,
            "intermediate_size": 64,
            "ssm_state_size": 8,
            "mamba_num_heads": 4,
            "mamba_head_dim": 16,
            "n_groups": 2,
            "conv_kernel": 4,
            "chunk_size": 8,
            **ATTENTION,
        },
    ),
    "bamba": (
        "BambaForCausalLM",
        {
            "num_hidden_layers": 2,
            "attn_layer_indices": [1],
            "intermediate_size": 64,
            **MAMBA2,
            **ATTENTION,
        },
    ),
    "granitemoehybrid": (
        "GraniteMoeHybridForCausalLM",
        {
            "num_hidden_layers": 2,
            "layer_types": ["mamba", "attention"],
            "intermediate_size": 64,
            "shared_intermediate_size": 64,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            **MAMBA2,
            **ATTENTION,
        },
    ),
    "mamba2": (
        "Mamba2ForCausalLM",
        {
            "num_hidden_layers": 2,
            "num_heads": 4,
            "head_dim": 16,
            "expand": 2,
            "n_groups": 2,
            "state_size": 8,
            "conv_kernel": 4,
            "chunk_size": 8,
        },
    ),
}


def build(model_type, gradient_checkpointing=False):
    causal_lm, config_kwargs = FAMILIES[model_type]
    modeling = importlib.import_module(
        f"transformers.models.{model_type}.modeling_{model_type}"
    )
    model_cls = getattr(modeling, causal_lm)
    config = model_cls.config_class(vocab_size=64, hidden_size=32, **config_kwargs)
    torch.manual_seed(0)
    model = model_cls(config).float()
    for name, param in model.named_parameters():
        if name.endswith(("A_log", "D", "dt_bias")):
            torch.nn.init.normal_(param, std=0.5)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable({"use_reentrant": False})
    return modeling, model.train()


@pytest.fixture
def scans(monkeypatch):
    """The token count of every scan the tree driver runs."""
    calls = []
    tree_scan = integration.tree_scan

    def spy(scan, inputs, plan, **kwargs):
        calls.append(plan.num_tokens)
        return tree_scan(scan, inputs, plan, **kwargs)

    monkeypatch.setattr(integration, "tree_scan", spy)
    return calls


@pytest.mark.parametrize("model_type", list(FAMILIES))
def test_torch_reference_kernels(model_type, scans, caplog):
    modeling, model = build(model_type)
    support = install(model)
    assert support.mode == "tree", support.reason
    assert_tree_matches_alone(model, branching_sequences(), rtol=1e-4)
    assert "mamba2_chunk_scan has no kernel" in caplog.text
    mixers = sum(type(m).__name__ in support.owners for m in model.modules())
    assert len(scans) == mixers


@pytest.mark.parametrize("model_type", list(FAMILIES))
@pytest.mark.parametrize(
    "gradient_checkpointing", [False, True], ids=["", "checkpointed"]
)
def test_seq_idx_kernels(monkeypatch, caplog, model_type, gradient_checkpointing):
    """Kernels that reset at seq_idx: the fused mixer runs every path in one call."""
    modeling, model = build(model_type, gradient_checkpointing)
    fused_calls = []

    def fused(zxbcdt, *args, seq_idx=None, **kwargs):
        fused_calls.append((zxbcdt.shape[1], seq_idx is not None))
        return reference.mamba_split_conv1d_scan_combined(
            zxbcdt, *args, seq_idx=seq_idx, **kwargs
        )

    monkeypatch.setattr(modeling, "mamba2_split_conv1d_scan_combined", fused)
    monkeypatch.setattr(
        modeling, "mamba2_chunk_scan", reference.mamba_chunk_scan_combined
    )
    monkeypatch.setattr(modeling, "causal_conv1d_fn", reference.causal_conv1d)
    assert install(model).mode == "tree"
    caplog.set_level("INFO", logger="tree_recurrence")
    sequences = branching_sequences()
    assert_tree_matches_alone(model, sequences, rtol=1e-4)
    row_tokens = len(pack(sequences).input_ids)
    tree_calls = [tokens for tokens, with_seq_idx in fused_calls if with_seq_idx]
    # every root-to-leaf path, laid end to end
    assert tree_calls and all(tokens > row_tokens for tokens in tree_calls)
    assert "split_conv1d_scan_combined runs trees on a kernel" in caplog.text


@pytest.mark.parametrize("model_type", list(FAMILIES))
def test_declining_fused_kernel_falls_back_to_the_unfused_path(monkeypatch, model_type):
    """A fused kernel that returns None (a guard for quantized out_proj, say) leaves
    the tree to the unfused path's conv and scan hooks."""
    modeling, model = build(model_type)
    declined = []

    def fused(zxbcdt, *args, **kwargs):
        declined.append(zxbcdt.shape[1])

    monkeypatch.setattr(modeling, "mamba2_split_conv1d_scan_combined", fused)
    monkeypatch.setattr(
        modeling, "mamba2_chunk_scan", reference.mamba_chunk_scan_combined
    )
    monkeypatch.setattr(modeling, "causal_conv1d_fn", reference.causal_conv1d)
    assert install(model).mode == "tree"
    assert_tree_matches_alone(model, branching_sequences(), rtol=1e-4)
    assert declined


def test_chains_and_collation_padding():
    _, model = build("nemotron_h")
    install(model)
    assert_tree_matches_alone(model, branching_sequences(), share=False, rtol=1e-4)
    assert_tree_matches_alone(
        model, branching_sequences(), collation_padding=3, rtol=1e-4
    )


def test_mamba2_causal_lm_hands_the_tree_on():
    modeling, model = build("mamba2")
    forward = modeling.Mamba2ForCausalLM.forward
    install(model)
    assert modeling.Mamba2ForCausalLM.forward is not forward
    uninstall(model)
    assert modeling.Mamba2ForCausalLM.forward is forward
    assert "forward" not in vars(model.backbone)


def test_without_hooks_packed_samples_leak():
    _, model = build("nemotron_h")
    with pytest.raises(AssertionError):
        assert_tree_matches_alone(model, branching_sequences(), rtol=1e-4)
