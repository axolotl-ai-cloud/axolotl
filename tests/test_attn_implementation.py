"""
Tests for attn_implementation normalization, registry registration,
capability properties, and backwards compatibility with legacy boolean
attention flags.
"""

import pytest

from axolotl.utils.schemas.config import AxolotlInputConfig
from axolotl.utils.schemas.enums import (
    _NO_DTYPE_CAST_ATTN_IMPLS,
    _NON_PACKING_ATTN_IMPLS,
    FLASH_ATTN_LIB_IMPLS,
)


class TestAttnImplementationNormalizer:
    """Test the normalize_attn_implementation validator."""

    @staticmethod
    def _normalize(data):
        return AxolotlInputConfig.normalize_attn_implementation(data)

    # --- Forward mapping: attn_implementation -> legacy flags ---

    @pytest.mark.parametrize(
        "impl,expected_flag",
        [
            ("eager", "eager_attention"),
            ("flash", "flash_attention"),
            ("sdpa", "sdp_attention"),
            ("flex", "flex_attention"),
            ("xformers", "xformers_attention"),
            ("sage", "sage_attention"),
            ("s2", "s2_attention"),
        ],
    )
    def test_attn_impl_sets_primary_legacy_flag(self, impl, expected_flag):
        data = {"attn_implementation": impl}
        result = AxolotlInputConfig.normalize_attn_implementation(data)
        assert result.get(expected_flag) is True, (
            f"{impl}: expected {expected_flag}=True"
        )

    @pytest.mark.parametrize("impl", ["xformers", "sage", "s2"])
    def test_attn_impl_does_not_set_flash_for_non_flash(self, impl):
        """xformers, sage, s2 should NOT set flash_attention=True anymore."""
        result = self._normalize({"attn_implementation": impl})
        assert not result.get("flash_attention"), (
            f"{impl} should not set flash_attention"
        )

    def test_fp8_sets_no_legacy_flags(self):
        result = self._normalize({"attn_implementation": "fp8"})
        for flag in [
            "flash_attention",
            "sdp_attention",
            "eager_attention",
            "xformers_attention",
            "sage_attention",
            "flex_attention",
            "s2_attention",
        ]:
            assert not result.get(flag), f"fp8 should not set {flag}"

    # --- Reverse mapping: legacy flags -> attn_implementation ---

    @pytest.mark.parametrize(
        "flag,expected_impl",
        [
            ("flash_attention", "flash"),
            ("sdp_attention", "sdpa"),
            ("xformers_attention", "xformers"),
            ("flex_attention", "flex"),
            ("sage_attention", "sage"),
            ("eager_attention", "eager"),
            ("s2_attention", "s2"),
        ],
    )
    def test_legacy_flag_sets_attn_impl(self, flag, expected_impl):
        result = self._normalize({flag: True})
        assert result["attn_implementation"] == expected_impl

    # --- Priority: s2/sage should win over flash when both set ---

    def test_s2_plus_flash_maps_to_s2(self):
        """Legacy configs often have both s2_attention and flash_attention."""
        result = self._normalize({"s2_attention": True, "flash_attention": True})
        assert result["attn_implementation"] == "s2"

    def test_sage_plus_flash_maps_to_sage(self):
        """sage_attention should take priority over flash_attention."""
        result = self._normalize({"sage_attention": True, "flash_attention": True})
        assert result["attn_implementation"] == "sage"

    # --- Consistency: both set, matching ---

    def test_consistent_both_set_no_error(self):
        result = self._normalize(
            {"attn_implementation": "flash", "flash_attention": True}
        )
        assert result["attn_implementation"] == "flash"
        assert result["flash_attention"] is True

    def test_consistent_xformers_with_own_flag(self):
        """xformers + xformers_attention should be OK."""
        result = self._normalize(
            {"attn_implementation": "xformers", "xformers_attention": True}
        )
        assert result["attn_implementation"] == "xformers"

    # --- Conflict detection ---

    def test_conflicting_impl_and_flag_raises(self):
        with pytest.raises(ValueError, match="conflicts with"):
            self._normalize({"attn_implementation": "flash", "sdp_attention": True})

    def test_conflicting_xformers_impl_with_sdp_flag(self):
        with pytest.raises(ValueError, match="conflicts with"):
            self._normalize({"attn_implementation": "xformers", "sdp_attention": True})

    def test_xformers_with_flash_flag_conflicts(self):
        """After normalizer change, xformers no longer expects flash_attention."""
        with pytest.raises(ValueError, match="conflicts with"):
            self._normalize(
                {
                    "attn_implementation": "xformers",
                    "xformers_attention": True,
                    "flash_attention": True,
                }
            )

    def test_s2_with_flash_flag_conflicts(self):
        """After normalizer change, s2 no longer expects flash_attention."""
        with pytest.raises(ValueError, match="conflicts with"):
            self._normalize(
                {
                    "attn_implementation": "s2",
                    "s2_attention": True,
                    "flash_attention": True,
                }
            )

    # --- Hub kernel strings pass through ---

    def test_hub_kernel_passthrough(self):
        result = self._normalize(
            {"attn_implementation": "kernels-community/flash-attn3"}
        )
        assert result["attn_implementation"] == "kernels-community/flash-attn3"
        # Should not set any legacy flags
        for flag in [
            "flash_attention",
            "sdp_attention",
            "eager_attention",
            "xformers_attention",
        ]:
            assert not result.get(flag)

    def test_custom_string_passthrough(self):
        result = self._normalize({"attn_implementation": "my_custom_kernel"})
        assert result["attn_implementation"] == "my_custom_kernel"

    # --- No attention set ---

    def test_no_attention_set_is_noop(self):
        result = self._normalize({"some_other_config": True})
        assert result.get("attn_implementation") is None

    # --- Gemma4 hybrid ---

    def test_gemma4_hybrid_sets_flash(self):
        """gemma4_hybrid_attn_impl should default attn_implementation to flash."""
        result = self._normalize({"gemma4_hybrid_attn_impl": True})
        assert result["attn_implementation"] == "flash"
        assert result["flash_attention"] is True

    def test_gemma4_hybrid_does_not_override_explicit(self):
        """If attn_implementation is already set, gemma4 should not override it."""
        result = self._normalize(
            {"gemma4_hybrid_attn_impl": True, "attn_implementation": "sdpa"}
        )
        assert result["attn_implementation"] == "sdpa"


class TestAttnCapabilityProperties:
    """Test the capability properties on the normalizer data.

    Since these are @property on AxolotlInputConfig (a Pydantic model),
    we test the underlying logic directly using the constant sets.
    """

    # --- attn_supports_packing ---

    @pytest.mark.parametrize("impl", ["flash", "flex", "xformers", "sage"])
    def test_supports_packing_true(self, impl):
        assert impl not in _NON_PACKING_ATTN_IMPLS

    @pytest.mark.parametrize("impl", ["eager", "sdpa", "s2", "fp8"])
    def test_supports_packing_false(self, impl):
        assert impl in _NON_PACKING_ATTN_IMPLS

    def test_hub_kernel_supports_packing(self):
        """Unknown hub kernels should default to packing-capable."""
        assert "kernels-community/flash-attn3" not in _NON_PACKING_ATTN_IMPLS

    # --- attn_uses_flash_lib ---

    @pytest.mark.parametrize("impl", ["flash", "s2"])
    def test_uses_flash_lib_true(self, impl):
        assert impl in FLASH_ATTN_LIB_IMPLS

    @pytest.mark.parametrize(
        "impl", ["eager", "sdpa", "xformers", "flex", "sage", "fp8"]
    )
    def test_uses_flash_lib_false(self, impl):
        assert impl not in FLASH_ATTN_LIB_IMPLS

    def test_hub_kernel_not_flash_lib(self):
        """Hub kernels are HF-managed, not axolotl monkeypatch targets."""
        assert "kernels-community/flash-attn3" not in FLASH_ATTN_LIB_IMPLS

    # --- attn_needs_dtype_cast ---

    @pytest.mark.parametrize("impl", ["eager", "sdpa"])
    def test_no_dtype_cast(self, impl):
        assert impl in _NO_DTYPE_CAST_ATTN_IMPLS

    @pytest.mark.parametrize("impl", ["flash", "flex", "sage", "xformers", "s2", "fp8"])
    def test_needs_dtype_cast(self, impl):
        assert impl not in _NO_DTYPE_CAST_ATTN_IMPLS


class TestAttnImplToHFMapping:
    """Test that attn_implementation enum values map correctly to HF strings."""

    # This dict mirrors _ATTN_IMPL_TO_HF in model.py
    _ATTN_IMPL_TO_HF = {
        "eager": "eager",
        "flash": "flash_attention_2",
        "sdpa": "sdpa",
        "xformers": "xformers",
        "flex": "flex_attention",
        "sage": "sage",
        "s2": "flash_attention_2",
        "fp8": "sdpa",
    }

    @pytest.mark.parametrize(
        "impl,expected_hf",
        [
            ("eager", "eager"),
            ("flash", "flash_attention_2"),
            ("sdpa", "sdpa"),
            ("xformers", "xformers"),
            ("flex", "flex_attention"),
            ("sage", "sage"),
            ("s2", "flash_attention_2"),
            ("fp8", "sdpa"),
        ],
    )
    def test_known_impl_maps_correctly(self, impl, expected_hf):
        assert self._ATTN_IMPL_TO_HF[impl] == expected_hf

    def test_hub_kernel_falls_through(self):
        """Hub kernel strings should pass through .get() unchanged."""
        hub_str = "kernels-community/flash-attn3"
        result = self._ATTN_IMPL_TO_HF.get(hub_str, hub_str)
        assert result == hub_str


def _xformers_available():
    try:
        import xformers.ops  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


class TestAttentionRegistration:
    """Test that attention backends register correctly in HF's registries."""

    @pytest.mark.skipif(not _xformers_available(), reason="xformers not available")
    def test_register_xformers(self):
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        from axolotl.monkeypatch.attention import register_xformers_attn

        register_xformers_attn()

        assert "xformers" in ALL_ATTENTION_FUNCTIONS
        assert "xformers" in ALL_MASK_ATTENTION_FUNCTIONS
        # xformers mask should be the same function as flash_attention_2's mask
        assert (
            ALL_MASK_ATTENTION_FUNCTIONS["xformers"]
            == ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"]
        )

    def test_register_sage(self):
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        from axolotl.monkeypatch.attention import register_sage_attn

        register_sage_attn()

        assert "sage" in ALL_ATTENTION_FUNCTIONS
        assert "sage" in ALL_MASK_ATTENTION_FUNCTIONS
        assert (
            ALL_MASK_ATTENTION_FUNCTIONS["sage"]
            == ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"]
        )

    @pytest.mark.skipif(not _xformers_available(), reason="xformers not available")
    def test_xformers_does_not_overwrite_fa2(self):
        """Registering xformers should not modify the flash_attention_2 slot."""
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        original_fa2 = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        from axolotl.monkeypatch.attention import register_xformers_attn

        register_xformers_attn()

        assert ALL_ATTENTION_FUNCTIONS["flash_attention_2"] is original_fa2

    def test_sage_does_not_overwrite_fa2(self):
        """Registering sage should not modify the flash_attention_2 slot."""
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        original_fa2 = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        from axolotl.monkeypatch.attention import register_sage_attn

        register_sage_attn()

        assert ALL_ATTENTION_FUNCTIONS["flash_attention_2"] is original_fa2
