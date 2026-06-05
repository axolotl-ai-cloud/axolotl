"""Tests for the MiniMax ``kernelize()`` / ``use_kernels`` crash fix.

The patch strips the non-Module ``apply_rotary_pos_emb`` ``_hidden_kernels``
entry that crashes ``model.kernelize()`` under ``use_kernels=True``. See
:mod:`axolotl.monkeypatch.minimax_kernelize` for the full rationale.
"""

import pytest

pytest.importorskip(
    "transformers.models.minimax_m2",
    reason="minimax_kernelize patch only matters when MiniMax M2 is available",
)


@pytest.fixture
def restore_minimax_attention():
    """Snapshot the patched ``__init__``\\ s and reset patch state after each test."""
    from transformers.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2Attention

    saved = {MiniMaxM2Attention: MiniMaxM2Attention.__init__}
    try:
        from transformers.models.minimax.modeling_minimax import MiniMaxAttention
    except ImportError:
        pass
    else:
        saved[MiniMaxAttention] = MiniMaxAttention.__init__
    yield
    from axolotl.monkeypatch import minimax_kernelize

    for cls, init in saved.items():
        cls.__init__ = init
    minimax_kernelize._PATCHED_CLASSES.clear()


def _m2_attention():
    from transformers.models.minimax_m2.configuration_minimax_m2 import MiniMaxM2Config
    from transformers.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2Attention

    cfg = MiniMaxM2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=256,
        max_position_embeddings=128,
    )
    return MiniMaxM2Attention, cfg


def test_patch_installs_and_is_idempotent(restore_minimax_attention):
    from transformers.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2Attention

    from axolotl.monkeypatch.minimax_kernelize import patch_minimax_kernelize

    assert patch_minimax_kernelize() is True
    init_first = MiniMaxM2Attention.__init__
    assert patch_minimax_kernelize() is True
    assert MiniMaxM2Attention.__init__ is init_first
    assert hasattr(init_first, "_axolotl_original")


def test_patch_strips_non_module_hidden_kernels(restore_minimax_attention):
    import torch.nn as nn

    from axolotl.monkeypatch.minimax_kernelize import patch_minimax_kernelize

    attn_cls, cfg = _m2_attention()

    # Before the patch, the bare function is registered (the crash source).
    attn_before = attn_cls(cfg, layer_idx=0)
    assert "apply_rotary_pos_emb" in getattr(attn_before, "_hidden_kernels", {})

    # The patch only drops non-Module entries; module-backed kernels stay.
    patch_minimax_kernelize()
    attn_after = attn_cls(cfg, layer_idx=0)
    hidden_kernels = dict(getattr(attn_after, "_hidden_kernels", {}))
    assert "apply_rotary_pos_emb" not in hidden_kernels
    assert all(isinstance(fn, nn.Module) for fn in hidden_kernels.values())


def test_patch_does_not_alter_weights(restore_minimax_attention):
    import torch

    from axolotl.monkeypatch.minimax_kernelize import patch_minimax_kernelize

    attn_cls, cfg = _m2_attention()
    torch.manual_seed(0)
    before = attn_cls(cfg, layer_idx=0).state_dict()

    patch_minimax_kernelize()
    torch.manual_seed(0)
    after = attn_cls(cfg, layer_idx=0).state_dict()

    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_unpatch_restores_original(restore_minimax_attention):
    from transformers.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2Attention

    from axolotl.monkeypatch.minimax_kernelize import (
        patch_minimax_kernelize,
        unpatch_minimax_kernelize,
    )

    original = MiniMaxM2Attention.__init__
    patch_minimax_kernelize()
    assert MiniMaxM2Attention.__init__ is not original
    unpatch_minimax_kernelize()
    assert MiniMaxM2Attention.__init__ is original


def test_unpatch_is_safe_without_prior_patch(restore_minimax_attention):
    from axolotl.monkeypatch.minimax_kernelize import unpatch_minimax_kernelize

    unpatch_minimax_kernelize()


def test_full_model_kernelize_succeeds_with_patch(restore_minimax_attention):
    """End-to-end: a tiny full MiniMax M2 model crashes in ``kernelize()`` without
    the patch and succeeds with it. No real weights or CUDA required."""
    from transformers import AutoModelForCausalLM
    from transformers.models.minimax_m2.configuration_minimax_m2 import MiniMaxM2Config

    from axolotl.monkeypatch.minimax_kernelize import patch_minimax_kernelize

    def build():
        cfg = MiniMaxM2Config(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_local_experts=4,
            num_experts_per_tok=2,
            vocab_size=256,
            max_position_embeddings=128,
        )
        return AutoModelForCausalLM.from_config(cfg)

    model = build()
    model.train()
    with pytest.raises((TypeError, AttributeError, ValueError)):
        model.kernelize()

    patch_minimax_kernelize()
    model = build()
    model.train()
    model.kernelize()
