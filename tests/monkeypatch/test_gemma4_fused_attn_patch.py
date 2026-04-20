"""Unit tests for the Gemma4 fused-attention shared_kv_states routing patch."""

import pytest

gemma4_modeling = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")


@pytest.fixture
def clean_decoder_layer_patch_slate():
    """Save and restore Gemma4TextDecoderLayer.__call__ and the sentinel."""
    from axolotl.monkeypatch.models.gemma4 import fused_attn

    cls = gemma4_modeling.Gemma4TextDecoderLayer
    original_call = cls.__call__
    had_sentinel = getattr(cls, "_axolotl_shared_kv_patched", False)

    if had_sentinel:
        del cls._axolotl_shared_kv_patched

    try:
        yield cls, fused_attn
    finally:
        cls.__call__ = original_call
        if had_sentinel:
            cls._axolotl_shared_kv_patched = True
        elif hasattr(cls, "_axolotl_shared_kv_patched"):
            del cls._axolotl_shared_kv_patched
        fused_attn._set_shared_kv_states(None)


class TestPatchedDecoderLayerCall:
    def test_pops_shared_kv_states_and_populates_store(
        self, clean_decoder_layer_patch_slate
    ):
        cls, fused_attn = clean_decoder_layer_patch_slate

        captured = {}

        def spy(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = dict(kwargs)
            return "spy_return"

        cls.__call__ = spy
        fused_attn._patch_decoder_layer_call()

        assert getattr(cls, "_axolotl_shared_kv_patched", False) is True
        assert cls.__call__ is not spy

        shared_kv = {"layer_0": ("k", "v")}
        result = cls.__call__(
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

    def test_clears_store_when_kwarg_absent(self, clean_decoder_layer_patch_slate):
        """Regression for commit 251021e1: a prior step's dict must not leak
        into a later call that omits `shared_kv_states`."""
        cls, fused_attn = clean_decoder_layer_patch_slate

        def spy(self, *args, **kwargs):
            return None

        cls.__call__ = spy
        fused_attn._patch_decoder_layer_call()

        stale = {"stale_step": True}
        fused_attn._set_shared_kv_states(stale)
        assert fused_attn._get_shared_kv_states() is stale

        cls.__call__(object())

        assert fused_attn._get_shared_kv_states() is None

    def test_store_visible_across_threads(self):
        """Regression for commit e3669b2c: the store must be readable from
        threads other than the one that set it. `threading.local()` failed
        this invariant, crashing with 'NoneType' object is not subscriptable'
        on MoE Gemma4 variants when autograd worker threads ran backward
        recompute under HF-Trainer gradient_checkpointing."""
        import threading

        from axolotl.monkeypatch.models.gemma4 import fused_attn

        sentinel = {"layer_0": ("k", "v")}
        try:
            fused_attn._set_shared_kv_states(sentinel)

            seen = {}

            def worker():
                seen["value"] = fused_attn._get_shared_kv_states()

            t = threading.Thread(target=worker)
            t.start()
            t.join()

            assert seen["value"] is sentinel
        finally:
            fused_attn._set_shared_kv_states(None)


@pytest.fixture
def clean_entry_point_patch_slate():
    """Save and restore Gemma4TextAttention.forward and Gemma4TextDecoderLayer.__call__."""
    from axolotl.monkeypatch.models.gemma4 import fused_attn

    decoder_cls = gemma4_modeling.Gemma4TextDecoderLayer
    attn_cls = gemma4_modeling.Gemma4TextAttention

    original_call = decoder_cls.__call__
    original_forward = attn_cls.forward
    had_sentinel = getattr(decoder_cls, "_axolotl_shared_kv_patched", False)

    if had_sentinel:
        del decoder_cls._axolotl_shared_kv_patched

    try:
        yield decoder_cls, attn_cls, original_call, original_forward, fused_attn
    finally:
        decoder_cls.__call__ = original_call
        attn_cls.forward = original_forward
        if had_sentinel:
            decoder_cls._axolotl_shared_kv_patched = True
        elif hasattr(decoder_cls, "_axolotl_shared_kv_patched"):
            del decoder_cls._axolotl_shared_kv_patched
        fused_attn._set_shared_kv_states(None)


class TestPatchGemma4FusedAttnEntryPoint:
    def test_default_flag_swaps_only_attention_forward(
        self, clean_entry_point_patch_slate
    ):
        (
            decoder_cls,
            attn_cls,
            original_call,
            original_forward,
            fused_attn,
        ) = clean_entry_point_patch_slate

        fused_attn.patch_gemma4_fused_attn()

        assert attn_cls.forward is not original_forward
        assert decoder_cls.__call__ is original_call
        assert not getattr(decoder_cls, "_axolotl_shared_kv_patched", False)

    def test_workaround_flag_installs_decoder_layer_patch(
        self, clean_entry_point_patch_slate
    ):
        (
            decoder_cls,
            attn_cls,
            original_call,
            original_forward,
            fused_attn,
        ) = clean_entry_point_patch_slate

        fused_attn.patch_gemma4_fused_attn(install_shared_kv_workaround=True)

        assert attn_cls.forward is not original_forward
        assert decoder_cls.__call__ is not original_call
        assert getattr(decoder_cls, "_axolotl_shared_kv_patched", False) is True
