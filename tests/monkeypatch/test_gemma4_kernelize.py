"""Tests for the Gemma 4 ``kernelize()`` / ``use_kernels`` crash fix.

transformers decorates ``Gemma4VisionAttention`` with
``@use_kernelized_func(apply_rotary_pos_emb)`` where the target is a plain
function. Under ``use_kernels=True``, ``model.kernelize()`` then tries to
``register_module()`` that function and crashes with::

    TypeError: ...apply_rotary_pos_emb is not a Module subclass

(and a follow-on ``AttributeError`` from the cleanup path). The patch strips the
dead non-Module ``_hidden_kernels`` entry so ``kernelize()`` succeeds. The entry
is never read by ``Gemma4VisionAttention.forward`` (which uses
``apply_multidimensional_rope``), so removing it is behavior-neutral.
"""

import pytest

pytest.importorskip(
    "transformers.models.gemma4",
    reason="gemma4_kernelize patch only matters when Gemma 4 is available",
)


@pytest.fixture
def restore_gemma4_vision_attention():
    """Snapshot ``Gemma4VisionAttention.__init__`` and reset patch state after
    each test so patch state doesn't leak across the suite."""
    from transformers.models.gemma4 import modeling_gemma4

    saved_init = modeling_gemma4.Gemma4VisionAttention.__init__
    yield modeling_gemma4
    modeling_gemma4.Gemma4VisionAttention.__init__ = saved_init
    from axolotl.monkeypatch import gemma4_kernelize

    gemma4_kernelize._PATCH_APPLIED = False


def _vision_config():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    return Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )


def test_patch_installs_and_is_idempotent(restore_gemma4_vision_attention):
    from axolotl.monkeypatch.gemma4_kernelize import patch_gemma4_kernelize

    assert patch_gemma4_kernelize() is True
    init_first = restore_gemma4_vision_attention.Gemma4VisionAttention.__init__
    # Second call must not re-wrap.
    assert patch_gemma4_kernelize() is True
    init_second = restore_gemma4_vision_attention.Gemma4VisionAttention.__init__
    assert init_first is init_second
    assert hasattr(init_first, "_axolotl_original")


def test_patch_strips_non_module_hidden_kernels(restore_gemma4_vision_attention):
    modeling_gemma4 = restore_gemma4_vision_attention
    from axolotl.monkeypatch.gemma4_kernelize import patch_gemma4_kernelize

    cfg = _vision_config()

    # Before the patch, the bare function is registered (the crash source).
    attn_before = modeling_gemma4.Gemma4VisionAttention(cfg, layer_idx=0)
    assert "apply_rotary_pos_emb" in getattr(attn_before, "_hidden_kernels", {})

    patch_gemma4_kernelize()
    attn_after = modeling_gemma4.Gemma4VisionAttention(cfg, layer_idx=0)
    assert dict(getattr(attn_after, "_hidden_kernels", {})) == {}


def test_register_module_path_no_longer_crashes(restore_gemma4_vision_attention):
    """The exact step that crashed: kernelize()'s ``attach_hidden_kernels``
    does ``register_module(name, fn)`` for each ``_hidden_kernels`` entry."""
    modeling_gemma4 = restore_gemma4_vision_attention
    from axolotl.monkeypatch.gemma4_kernelize import patch_gemma4_kernelize

    cfg = _vision_config()
    patch_gemma4_kernelize()
    attn = modeling_gemma4.Gemma4VisionAttention(cfg, layer_idx=0)

    # Replicate attach_hidden_kernels; with the entry stripped there is nothing
    # to (mis)register, so this must not raise.
    for name, fn in getattr(attn, "_hidden_kernels", {}).items():
        if name not in dict(attn.named_children()):
            attn.register_module(name, fn)


def test_patch_does_not_alter_weights(restore_gemma4_vision_attention):
    """The shim only mutates ``_hidden_kernels``; parameters are untouched."""
    import torch

    modeling_gemma4 = restore_gemma4_vision_attention
    from axolotl.monkeypatch.gemma4_kernelize import patch_gemma4_kernelize

    cfg = _vision_config()
    torch.manual_seed(0)
    before = modeling_gemma4.Gemma4VisionAttention(cfg, layer_idx=0).state_dict()

    patch_gemma4_kernelize()
    torch.manual_seed(0)
    after = modeling_gemma4.Gemma4VisionAttention(cfg, layer_idx=0).state_dict()

    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_forward_does_not_reference_stripped_entry(restore_gemma4_vision_attention):
    """Behavior-invariance guarantee: forward never reads the stripped names,
    so dropping them cannot change the forward result."""
    modeling_gemma4 = restore_gemma4_vision_attention
    names = modeling_gemma4.Gemma4VisionAttention.forward.__code__.co_names
    assert "apply_rotary_pos_emb" not in names
    assert "_hidden_kernels" not in names


def test_unpatch_restores_original(restore_gemma4_vision_attention):
    modeling_gemma4 = restore_gemma4_vision_attention
    from axolotl.monkeypatch.gemma4_kernelize import (
        patch_gemma4_kernelize,
        unpatch_gemma4_kernelize,
    )

    original = modeling_gemma4.Gemma4VisionAttention.__init__
    patch_gemma4_kernelize()
    assert modeling_gemma4.Gemma4VisionAttention.__init__ is not original
    unpatch_gemma4_kernelize()
    assert modeling_gemma4.Gemma4VisionAttention.__init__ is original


def test_unpatch_is_safe_without_prior_patch(restore_gemma4_vision_attention):
    from axolotl.monkeypatch.gemma4_kernelize import unpatch_gemma4_kernelize

    # No-op, no exception.
    unpatch_gemma4_kernelize()


def test_full_model_kernelize_succeeds_with_patch(restore_gemma4_vision_attention):
    """End-to-end: a tiny full Gemma4 model crashes in ``kernelize()`` without
    the patch and succeeds with it. No real 26B weights or CUDA required."""
    modeling_gemma4 = restore_gemma4_vision_attention
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4AudioConfig,
        Gemma4Config,
        Gemma4TextConfig,
        Gemma4VisionConfig,
    )

    from axolotl.monkeypatch.gemma4_kernelize import patch_gemma4_kernelize

    def build():
        text = Gemma4TextConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=128,
            num_experts=4,
            num_experts_per_tok=2,
        )
        vis = Gemma4VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
        aud = Gemma4AudioConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
        )
        cfg = Gemma4Config(text_config=text, vision_config=vis, audio_config=aud)
        return modeling_gemma4.Gemma4ForConditionalGeneration(cfg)

    # Without the patch, kernelize() crashes.
    model = build()
    model.train()
    with pytest.raises((TypeError, AttributeError, ValueError)):
        model.kernelize()

    # With the patch, it succeeds.
    patch_gemma4_kernelize()
    model = build()
    model.train()
    model.kernelize()
