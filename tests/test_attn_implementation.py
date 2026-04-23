"""Tests for attn_implementation normalization and capability computation.

Covers the Phase 1 contract:
- `attn_implementation` accepts canonical names only; short-form aliases are rejected.
- Legacy boolean flags are mapped to the canonical value, warned on, and stripped.
- Canonical `attn_implementation` + legacy flag raises.
- Capability flags are computed from `attn_implementation`.
"""

import pytest

from axolotl.utils.schemas.config import AxolotlInputConfig
from axolotl.utils.schemas.enums import (
    ATTN_IMPLS_SUPPORTING_PACKING,
    ATTN_IMPLS_USING_FLASH_LIB,
    ATTN_IMPLS_WITHOUT_DTYPE_CAST,
    CANONICAL_ATTN_IMPLS,
)


class TestNormalizerLegacyMapping:
    """Legacy boolean flags map to canonical attn_implementation."""

    @staticmethod
    def _normalize(data):
        return AxolotlInputConfig.normalize_attn_implementation(data)

    @pytest.mark.parametrize(
        "flag,expected",
        [
            ("flash_attention", "flash_attention_2"),
            ("sdp_attention", "sdpa"),
            ("xformers_attention", "xformers"),
            ("flex_attention", "flex_attention"),
            ("sage_attention", "sage"),
            ("eager_attention", "eager"),
            ("s2_attention", "s2"),
        ],
    )
    def test_legacy_flag_maps_to_canonical(self, flag, expected):
        result = self._normalize({flag: True})
        assert result["attn_implementation"] == expected

    def test_legacy_flags_are_stripped_after_mapping(self):
        result = self._normalize({"flash_attention": True})
        for flag in [
            "flash_attention",
            "sdp_attention",
            "xformers_attention",
            "flex_attention",
            "sage_attention",
            "eager_attention",
            "s2_attention",
        ]:
            assert flag not in result

    def test_s2_plus_flash_priority_is_s2(self):
        result = self._normalize({"s2_attention": True, "flash_attention": True})
        assert result["attn_implementation"] == "s2"

    def test_sage_plus_flash_priority_is_sage(self):
        result = self._normalize({"sage_attention": True, "flash_attention": True})
        assert result["attn_implementation"] == "sage"


class TestNormalizerConflicts:
    """Canonical attn_implementation + legacy flag raises."""

    @staticmethod
    def _normalize(data):
        return AxolotlInputConfig.normalize_attn_implementation(data)

    def test_canonical_plus_legacy_flag_raises(self):
        with pytest.raises(ValueError, match="cannot be combined with legacy"):
            self._normalize(
                {"attn_implementation": "flash_attention_2", "flash_attention": True}
            )

    def test_canonical_plus_unrelated_legacy_flag_raises(self):
        with pytest.raises(ValueError, match="cannot be combined with legacy"):
            self._normalize(
                {"attn_implementation": "xformers", "flash_attention": True}
            )


class TestNormalizerPassthrough:
    """Canonical values and hub-kernel paths pass through."""

    def test_canonical_no_legacy_is_noop(self):
        data = {"attn_implementation": "flash_attention_2"}
        result = AxolotlInputConfig.normalize_attn_implementation(data)
        assert result["attn_implementation"] == "flash_attention_2"

    def test_hub_kernel_passes_through(self):
        data = {"attn_implementation": "kernels-community/flash-attn3"}
        result = AxolotlInputConfig.normalize_attn_implementation(data)
        assert result["attn_implementation"] == "kernels-community/flash-attn3"

    def test_no_attention_set_is_noop(self):
        result = AxolotlInputConfig.normalize_attn_implementation(
            {"some_other_config": True}
        )
        assert result.get("attn_implementation") is None


class TestGemma4Hybrid:
    """gemma4_hybrid_attn_impl defaults to flash_attention_2."""

    def test_gemma4_hybrid_defaults_to_fa2(self):
        result = AxolotlInputConfig.normalize_attn_implementation(
            {"gemma4_hybrid_attn_impl": True}
        )
        assert result["attn_implementation"] == "flash_attention_2"

    def test_gemma4_hybrid_respects_explicit(self):
        result = AxolotlInputConfig.normalize_attn_implementation(
            {"gemma4_hybrid_attn_impl": True, "attn_implementation": "sdpa"}
        )
        assert result["attn_implementation"] == "sdpa"


class TestFieldValidator:
    """attn_implementation field_validator rejects short-form aliases."""

    def test_canonical_accepted(self):
        for impl in CANONICAL_ATTN_IMPLS:
            assert AxolotlInputConfig.validate_attn_implementation(impl) == impl

    def test_hub_kernel_accepted(self):
        for impl in (
            "kernels-community/flash-attn3",
            "kernels-community/sage-attention",
            "someorg/custom-kernel",
        ):
            assert AxolotlInputConfig.validate_attn_implementation(impl) == impl

    def test_none_accepted(self):
        assert AxolotlInputConfig.validate_attn_implementation(None) is None

    @pytest.mark.parametrize("alias", ["flash", "flex", "sdp"])
    def test_short_form_alias_rejected(self, alias):
        with pytest.raises(ValueError, match="is not accepted"):
            AxolotlInputConfig.validate_attn_implementation(alias)

    def test_unknown_without_slash_rejected(self):
        with pytest.raises(ValueError, match="not a recognized backend"):
            AxolotlInputConfig.validate_attn_implementation("not_a_real_backend")


class TestCapabilityTables:
    """Capability tables are keyed by canonical names and cover the expected backends."""

    @pytest.mark.parametrize(
        "impl",
        [
            "flash_attention_2",
            "flash_attention_3",
            "flex_attention",
            "xformers",
            "sage",
        ],
    )
    def test_supports_packing(self, impl):
        assert impl in ATTN_IMPLS_SUPPORTING_PACKING

    @pytest.mark.parametrize("impl", ["eager", "sdpa", "s2", "fp8"])
    def test_does_not_support_packing(self, impl):
        assert impl not in ATTN_IMPLS_SUPPORTING_PACKING

    @pytest.mark.parametrize("impl", ["flash_attention_2", "flash_attention_3", "s2"])
    def test_uses_flash_lib(self, impl):
        assert impl in ATTN_IMPLS_USING_FLASH_LIB

    @pytest.mark.parametrize(
        "impl", ["eager", "sdpa", "xformers", "flex_attention", "sage", "fp8"]
    )
    def test_does_not_use_flash_lib(self, impl):
        assert impl not in ATTN_IMPLS_USING_FLASH_LIB

    @pytest.mark.parametrize("impl", ["eager", "sdpa"])
    def test_no_dtype_cast(self, impl):
        assert impl in ATTN_IMPLS_WITHOUT_DTYPE_CAST

    @pytest.mark.parametrize(
        "impl",
        [
            "flash_attention_2",
            "flash_attention_3",
            "flex_attention",
            "xformers",
            "sage",
            "s2",
            "fp8",
        ],
    )
    def test_needs_dtype_cast(self, impl):
        assert impl not in ATTN_IMPLS_WITHOUT_DTYPE_CAST

    def test_known_hub_kernels_classified(self):
        assert "kernels-community/flash-attn3" in ATTN_IMPLS_SUPPORTING_PACKING
        assert "kernels-community/flash-attn3" in ATTN_IMPLS_USING_FLASH_LIB
        assert "kernels-community/sage-attention" in ATTN_IMPLS_SUPPORTING_PACKING


def _xformers_available():
    try:
        import xformers.ops  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


class TestAttentionRegistration:
    """Axolotl-owned backends register under their canonical names in HF's registries."""

    @pytest.mark.skipif(not _xformers_available(), reason="xformers not available")
    def test_register_xformers(self):
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        from axolotl.monkeypatch.attention import register_xformers_attn

        register_xformers_attn()

        assert "xformers" in ALL_ATTENTION_FUNCTIONS
        assert "xformers" in ALL_MASK_ATTENTION_FUNCTIONS
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
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        original_fa2 = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        from axolotl.monkeypatch.attention import register_xformers_attn

        register_xformers_attn()

        assert ALL_ATTENTION_FUNCTIONS["flash_attention_2"] is original_fa2

    def test_sage_does_not_overwrite_fa2(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        original_fa2 = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        from axolotl.monkeypatch.attention import register_sage_attn

        register_sage_attn()

        assert ALL_ATTENTION_FUNCTIONS["flash_attention_2"] is original_fa2
