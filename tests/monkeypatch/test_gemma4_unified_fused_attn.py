"""Tests for the Gemma4 *Unified* fused-attention monkeypatch.

These cover only behavior that diverges from standard gemma4 (whose fused-attn
patch is already covered by ``test_gemma4_fused_attn.py`` /
``test_gemma4_fused_attn_patch.py``):

* the idempotency guard on ``patch_gemma4_unified_fused_attn`` (standard gemma4
  has none),
* that the side channel routes through the *unified* store/decoder-layer class
  (a separate object that standard-gemma4 tests never touch), and
* shared-KV forward/backward under the unified backbone's hardcoded
  ``layer_type`` keying (vs the version-tolerant helpers in standard gemma4).
"""

import pytest
import torch

gemma4_unified_modeling = pytest.importorskip(
    "transformers.models.gemma4_unified.modeling_gemma4_unified",
    reason="unified fused_attn patch only matters when gemma4_unified is available",
)


@pytest.fixture
def clean_unified_slate():
    """Snapshot/restore ``Gemma4UnifiedTextAttention.forward``,
    ``Gemma4UnifiedTextDecoderLayer.__call__`` and both axolotl sentinels so the
    patch (and its idempotency flag) can't leak across tests."""
    from axolotl.monkeypatch.models.gemma4_unified import fused_attn

    attn_cls = gemma4_unified_modeling.Gemma4UnifiedTextAttention
    decoder_cls = gemma4_unified_modeling.Gemma4UnifiedTextDecoderLayer
    original_forward = attn_cls.forward
    original_call = decoder_cls.__call__
    had_attn_flag = getattr(attn_cls, "_axolotl_fused_attn_patched", False)
    had_dec_flag = getattr(decoder_cls, "_axolotl_shared_kv_patched", False)

    if had_attn_flag:
        del attn_cls._axolotl_fused_attn_patched
    if had_dec_flag:
        del decoder_cls._axolotl_shared_kv_patched

    try:
        yield attn_cls, decoder_cls, original_forward, original_call, fused_attn
    finally:
        attn_cls.forward = original_forward
        decoder_cls.__call__ = original_call
        for cls, attr, had in (
            (attn_cls, "_axolotl_fused_attn_patched", had_attn_flag),
            (decoder_cls, "_axolotl_shared_kv_patched", had_dec_flag),
        ):
            if had:
                setattr(cls, attr, True)
            elif hasattr(cls, attr):
                delattr(cls, attr)
        fused_attn._set_shared_kv_states(None)


class TestUnifiedFusedAttnEntryPoint:
    def test_patch_is_idempotent(self, clean_unified_slate):
        """Unified guards re-patching via ``_axolotl_fused_attn_patched`` — a
        second call must not re-wrap forward (standard gemma4 has no such guard)."""
        pytest.importorskip("triton")
        attn_cls, _decoder_cls, _orig_fwd, _orig_call, fused_attn = clean_unified_slate

        fused_attn.patch_gemma4_unified_fused_attn()
        wrapped = attn_cls.forward
        assert getattr(attn_cls, "_axolotl_fused_attn_patched", False) is True

        fused_attn.patch_gemma4_unified_fused_attn()
        assert attn_cls.forward is wrapped, "second patch call must be a no-op"

    def test_default_flag_swaps_only_attention_forward(self, clean_unified_slate):
        pytest.importorskip("triton")
        attn_cls, decoder_cls, original_forward, original_call, fused_attn = (
            clean_unified_slate
        )

        fused_attn.patch_gemma4_unified_fused_attn()

        assert attn_cls.forward is not original_forward
        assert decoder_cls.__call__ is original_call
        assert not getattr(decoder_cls, "_axolotl_shared_kv_patched", False)

    def test_workaround_flag_installs_decoder_layer_patch(self, clean_unified_slate):
        pytest.importorskip("triton")
        attn_cls, decoder_cls, original_forward, original_call, fused_attn = (
            clean_unified_slate
        )

        fused_attn.patch_gemma4_unified_fused_attn(install_shared_kv_workaround=True)

        assert attn_cls.forward is not original_forward
        assert decoder_cls.__call__ is not original_call
        assert getattr(decoder_cls, "_axolotl_shared_kv_patched", False) is True


class TestUnifiedDecoderLayerCall:
    """The PR#3611 side channel uses a *separate* store + decoder-layer class
    from standard gemma4; a copy-paste regression (writing to gemma4's store, or
    failing to clear) would be caught by nothing in the standard-gemma4 tests."""

    def test_store_is_distinct_from_standard_gemma4(self):
        from axolotl.monkeypatch.models.gemma4 import fused_attn as regular
        from axolotl.monkeypatch.models.gemma4_unified import fused_attn as unified

        assert (
            unified._GEMMA4_UNIFIED_SHARED_KV_STORE
            is not regular._GEMMA4_SHARED_KV_STORE
        )

    def test_pops_and_clears_shared_kv_states(self, clean_unified_slate):
        _attn_cls, decoder_cls, _orig_fwd, _orig_call, fused_attn = clean_unified_slate

        captured = {}

        def spy(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = dict(kwargs)
            return "spy_return"

        decoder_cls.__call__ = spy
        fused_attn._patch_decoder_layer_call()

        assert getattr(decoder_cls, "_axolotl_shared_kv_patched", False) is True

        shared_kv = {"full_attention": ("k", "v")}
        result = decoder_cls.__call__(
            object(),
            "positional_arg",
            shared_kv_states=shared_kv,
            other_kwarg="keep_me",
        )

        assert result == "spy_return"
        assert captured["args"] == ("positional_arg",)
        assert "shared_kv_states" not in captured["kwargs"]
        assert captured["kwargs"] == {"other_kwarg": "keep_me"}
        assert fused_attn._get_shared_kv_states() is shared_kv

        # A later call without the kwarg must clear the store (no cross-step leak).
        decoder_cls.__call__(object())
        assert fused_attn._get_shared_kv_states() is None


def _build_unified_kv_shared_config():
    """Tiny unified text config with ``num_kv_shared_layers > 0`` so the fused
    shared-KV branch (hardcoded ``self.layer_type`` keying) runs."""
    from transformers.models.gemma4_unified.configuration_gemma4_unified import (
        Gemma4UnifiedTextConfig,
    )

    cfg = Gemma4UnifiedTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_kv_shared_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        global_head_dim=64,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        sliding_window=64,
        max_position_embeddings=2048,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
        },
    )
    cfg._attn_implementation = "sdpa"
    return cfg


class TestUnifiedFusedAttnSharedKV:
    """Guards the unified backbone's hardcoded ``layer_type`` shared-KV keying
    (standard gemma4 uses version-tolerant ``_shared_kv_*_key`` helpers)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_shared_kv_forward_backward(self, clean_unified_slate):
        _attn_cls, _decoder_cls, _orig_fwd, _orig_call, fused_attn = clean_unified_slate
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            Gemma4UnifiedTextModel,
        )

        torch.manual_seed(4)
        m = (
            Gemma4UnifiedTextModel(_build_unified_kv_shared_config())
            .cuda()
            .to(torch.bfloat16)
            .train()
        )
        assert any(layer.self_attn.is_kv_shared_layer for layer in m.layers), (
            "test config must exercise at least one kv-shared layer"
        )

        ids = torch.randint(0, 128, (2, 16), device="cuda")
        mask = torch.ones(2, 16, dtype=torch.long, device="cuda")

        with torch.no_grad():
            ref = m(input_ids=ids, attention_mask=mask).last_hidden_state.clone()

        fused_attn.patch_gemma4_unified_fused_attn()
        out = m(input_ids=ids, attention_mask=mask).last_hidden_state
        out.sum().backward()

        assert out.shape == ref.shape
        assert torch.isfinite(out).all()
        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().float(), out.detach().flatten().float(), dim=0
        )
        assert cos_sim > 0.999, f"shared-kv fused vs stock cosine_sim={cos_sim:.6f}"
