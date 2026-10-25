"""Mamba2 models on tree rows score every sample as it scores alone."""

import importlib

import numpy as np
import pytest
import torch
from trl.experimental.async_grpo.packing import TreeRow
from trl_async_test_utils import reference_tree_attention

from axolotl.core.trainers.grpo.trl_async_packing import (
    ChainRow,
    sample_boundaries,
    shares_prefixes,
)


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


def _sample(ids, prompt_len, group_id):
    ids = np.asarray(ids, dtype=np.int64)
    return {
        "input_ids": ids,
        "completion_mask": np.asarray([0] * prompt_len + [1] * (len(ids) - prompt_len)),
        "old_log_probs": np.zeros(len(ids), np.float32),
        "advantage": 1.0,
        "group_id": group_id,
    }


def _set_attention(model, implementation):
    if hasattr(model.config, "_attn_implementation") and any(
        "Attention" in type(m).__name__ for m in model.modules()
    ):
        model.set_attn_implementation(implementation)


@pytest.mark.parametrize("mamba2_family", list(FAMILIES), indirect=True)
def test_tree_rows_match_each_sample_alone(mamba2_family):
    from trl.experimental.async_grpo import packing

    model, scans = mamba2_family
    packing.register_tree_attention()
    assert shares_prefixes(model)
    rng = np.random.default_rng(0)
    samples = []
    for group_id in range(2):
        prompt = rng.integers(1, 64, size=6 + group_id).tolist()
        for g in range(3):
            completion = rng.integers(1, 64, size=3 + g).tolist()
            samples.append(_sample(prompt + completion, len(prompt), group_id))
        # a rollout branching off the last one after its first completion token
        samples.append(
            _sample(
                samples[-1]["input_ids"][: len(prompt) + 1].tolist() + [5, 6],
                len(prompt),
                group_id,
            )
        )

    def grads():
        found = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }
        model.zero_grad()
        return found

    def score_row(row):
        _set_attention(model, packing.TREE_ATTENTION)
        for sample in samples:
            row.add(sample)
        packed = row.pack()
        tree = {
            "tree_kv_index": packed["tree_kv_index"],
            "tree_cu_seqlens_q": packed["tree_cu_seqlens_q"],
            "tree_cu_seqlens_k": packed["tree_cu_seqlens_k"],
            "tree_max_seqlen_q": int(packed["tree_max_seqlens"][0]),
            "tree_max_seqlen_k": int(packed["tree_max_seqlens"][1]),
        }
        if isinstance(row, ChainRow):
            tree.update(sample_boundaries(tree))
        scans.clear()
        logits = model(
            input_ids=packed["input_ids"][None],
            position_ids=packed["position_ids"][None],
            use_cache=False,
            **tree,
        ).logits[0]
        index = packed["token_index"]
        log_probs = torch.log_softmax(logits[index], -1).gather(
            1, packed["shift_labels"][index][:, None]
        )[:, 0]
        log_probs.sum().backward()
        return len(packed["input_ids"]), list(scans), log_probs.detach(), grads()

    def score_alone():
        _set_attention(model, "sdpa")
        log_probs = []
        for sample in samples:
            ids = torch.as_tensor(sample["input_ids"])[None]
            logits = model(input_ids=ids, use_cache=False).logits[0, :-1]
            trained = torch.as_tensor(sample["completion_mask"][1:]).bool()
            log_probs.append(
                torch.log_softmax(logits[trained], -1).gather(
                    1, ids[0, 1:][trained][:, None]
                )[:, 0]
            )
        torch.cat(log_probs).sum().backward()
        return torch.cat(log_probs).detach(), grads()

    model.train()
    shared_len, shared_scans, shared, shared_grads = score_row(TreeRow())
    chain_len, _, chains, chain_grads = score_row(ChainRow())
    alone, alone_grads = score_alone()
    assert shared_len < chain_len
    # the trie's scan ran over every root-to-segment path
    assert shared_scans and all(n > shared_len for n in shared_scans)
    for packed_log_probs, packed_grads in (
        (shared, shared_grads),
        (chains, chain_grads),
    ):
        torch.testing.assert_close(packed_log_probs, alone, rtol=1e-5, atol=1e-5)
        assert packed_grads.keys() == alone_grads.keys()
        for name, grad in packed_grads.items():
            torch.testing.assert_close(grad, alone_grads[name], rtol=1e-4, atol=1e-5)
