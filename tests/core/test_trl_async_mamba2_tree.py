"""Mamba2 models on tree rows score every sample as it scores alone."""

import importlib

import pytest
import torch
from trl_async_test_utils import (
    assert_tree_rows_match_alone,
    branching_samples,
    reference_tree_attention,
)

from axolotl.core.trainers.grpo.trl_async_packing import shares_prefixes


def _segments(seq_idx, tokens):
    if seq_idx is None:
        return [(0, tokens)]
    ids = seq_idx[0].tolist()
    starts = [t for t in range(tokens) if t == 0 or ids[t] != ids[t - 1]]
    return list(zip(starts, starts[1:] + [tokens], strict=True))


def reference_conv(x, weight, bias=None, activation=None, seq_idx=None, **kwargs):
    """causal_conv1d_fn on ``[B, D, T]`` that resets at ``seq_idx`` boundaries."""
    outs = []
    for start, end in _segments(seq_idx, x.shape[-1]):
        seq = torch.nn.functional.pad(x[..., start:end], (weight.shape[-1] - 1, 0))
        outs.append(
            torch.nn.functional.conv1d(seq, weight[:, None], bias, groups=x.shape[1])
        )
    out = torch.cat(outs, dim=-1)
    return torch.nn.functional.silu(out) if activation else out


def reference_scan(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    seq_idx=None,
    **kwargs,
):
    """mamba_chunk_scan_combined token by token, resetting at ``seq_idx`` boundaries."""
    heads, groups = x.shape[2], B.shape[2]
    dt = dt.float() + (0 if dt_bias is None else dt_bias.float())
    dt = torch.nn.functional.softplus(dt) if dt_softplus else dt
    dt = dt.clamp(*dt_limit)
    B = B.float().repeat_interleave(heads // groups, dim=2)
    C = C.float().repeat_interleave(heads // groups, dim=2)
    outs = []
    for start, end in _segments(seq_idx, x.shape[1]):
        state = x.new_zeros(heads, x.shape[-1], B.shape[-1], dtype=torch.float32)
        for t in range(start, end):
            decay = torch.exp(dt[0, t] * A)[:, None, None]
            update = (dt[0, t, :, None] * x[0, t].float())[:, :, None] * B[
                0, t, :, None
            ]
            state = state * decay + update
            y = (state * C[0, t, :, None]).sum(-1)
            if D is not None:
                y = y + D[:, None] * x[0, t].float()
            outs.append(y)
    return torch.stack(outs)[None].to(x.dtype)


def _patch_nemotron_h(modeling):
    from axolotl.monkeypatch.models.nemotron_h.modeling import (
        patch_nemotron_h_modeling_packing,
    )

    patch_nemotron_h_modeling_packing(kernels_enabled=True)


def _patch_bamba(modeling):
    from axolotl.monkeypatch.models.mamba_utils import patch_model_forward_seq_idx

    patch_model_forward_seq_idx(modeling.BambaModel)


def _patch_granitemoehybrid(modeling):
    from axolotl.monkeypatch.models.granitemoehybrid.modeling import (
        patch_granitemoehybrid_modeling_packing,
    )

    patch_granitemoehybrid_modeling_packing()


def _patch_mamba2(modeling):
    from axolotl.monkeypatch.models.mamba.modeling import (
        patch_mamba2_modeling_packing,
    )

    patch_mamba2_modeling_packing(kernels_enabled=True)


MAMBA2 = {"mamba_n_heads": 4, "mamba_d_head": 16, "mamba_n_groups": 2}
MAMBA2.update(mamba_d_state=8, mamba_d_conv=4, mamba_expand=2, mamba_chunk_size=8)
ATTENTION = {"num_attention_heads": 2, "num_key_value_heads": 1}
# model type -> (packing patch, classes it patches, causal LM, config kwargs)
FAMILIES = {
    "nemotron_h": (
        _patch_nemotron_h,
        ("NemotronHBlock", "NemotronHMamba2Mixer"),
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
        _patch_bamba,
        ("BambaModel", "BambaMixer"),
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
        _patch_granitemoehybrid,
        ("GraniteMoeHybridModel", "GraniteMoeHybridMambaLayer"),
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
        _patch_mamba2,
        ("Mamba2ForCausalLM", "Mamba2Block", "Mamba2Mixer"),
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


@pytest.fixture
def mamba2_family(request, monkeypatch):
    """A tiny model of one family, with axolotl's packing and tree patches applied
    over reference kernels; returns it and the token count of every scan call."""
    from trl.experimental.async_grpo import packing

    from axolotl.monkeypatch.models.mamba2_tree import patch_mamba2_tree

    model_type = request.param
    apply_packing, classes, causal_lm, config_kwargs = FAMILIES[model_type]
    modeling = importlib.import_module(
        f"transformers.models.{model_type}.modeling_{model_type}"
    )
    monkeypatch.setattr(packing, "tree_attention", reference_tree_attention)
    for name in classes:
        cls = getattr(modeling, name)
        monkeypatch.setattr(cls, "forward", cls.forward)
    scans = []

    def scan(*args, seq_idx=None, **kwargs):
        scans.append(args[0].shape[1])
        return reference_scan(*args, seq_idx=seq_idx, **kwargs)

    monkeypatch.setattr(modeling, "mamba2_chunk_scan", scan)
    monkeypatch.setattr(modeling, "causal_conv1d_fn", reference_conv)
    monkeypatch.setattr(
        modeling, "mamba2_split_conv1d_scan_combined", lambda *a, **k: None
    )
    monkeypatch.setattr(modeling, "_axolotl_fused_scan_guard", True, raising=False)
    if model_type == "mamba2":
        from axolotl.monkeypatch.models.mamba import modeling as mamba_packing

        # the reference kernels honor seq_idx on CPU
        monkeypatch.setattr(mamba_packing, "_assert_packed_ready", lambda *a: None)
    apply_packing(modeling)
    assert patch_mamba2_tree(model_type)

    model_cls = getattr(modeling, causal_lm)
    config = model_cls.config_class(vocab_size=64, hidden_size=32, **config_kwargs)
    torch.manual_seed(0)
    model = model_cls(config)
    for name, param in model.named_parameters():
        if name.endswith(("A_log", "D", "dt_bias")):
            torch.nn.init.normal_(param, std=0.5)
    return model.float(), scans


@pytest.mark.parametrize("mamba2_family", list(FAMILIES), indirect=True)
def test_tree_rows_match_each_sample_alone(mamba2_family):
    model, scans = mamba2_family
    assert shares_prefixes(model)
    rows = []

    def on_row(row_tokens):
        rows.append((row_tokens, list(scans)))
        scans.clear()

    assert_tree_rows_match_alone(model, branching_samples(), on_row)
    # the trie's scan ran over every root-to-segment path
    shared_len, shared_scans = rows[0]
    assert shared_scans and all(n > shared_len for n in shared_scans)
