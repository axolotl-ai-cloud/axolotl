"""Tests for attn_implementation: normalization, canonical-value acceptance,
capability flags, backend registration, and downstream validators.
"""

import logging
import subprocess
import sys
from contextlib import contextmanager
from functools import lru_cache

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.config import AxolotlInputConfig
from axolotl.utils.schemas.enums import (
    ATTN_IMPLS_SUPPORTING_PACKING,
    ATTN_IMPLS_USING_FLASH_LIB,
    ATTN_IMPLS_WITHOUT_DTYPE_CAST,
    CANONICAL_ATTN_IMPLS,
)


@contextmanager
def _capture_axolotl_warnings(caplog):
    """Capture WARNINGs from `axolotl.*` loggers via caplog.

    `axolotl.cli` calls `configure_logging()` at import time, which sets
    `propagate=False` on the `axolotl` logger so records do not reach the root
    logger that pytest's `caplog` hooks. This helper temporarily re-enables
    propagation for the duration of the block.
    """
    ax_logger = logging.getLogger("axolotl")
    old_propagate = ax_logger.propagate
    ax_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="axolotl"):
            yield
    finally:
        ax_logger.propagate = old_propagate


@lru_cache(maxsize=1)
def _xformers_available():
    try:
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-c",
                (
                    "import warnings; "
                    "warnings.filterwarnings('ignore', category=DeprecationWarning); "
                    "import xformers.ops"
                ),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:  # pylint: disable=broad-except
        return False


class TestCapabilityTables:
    """Backend capability classification via frozensets and computed_field properties."""

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

    @pytest.mark.parametrize("impl", ["eager", "sdpa", "fp8"])
    def test_does_not_support_packing(self, impl):
        assert impl not in ATTN_IMPLS_SUPPORTING_PACKING

    @pytest.mark.parametrize("impl", ["flash_attention_2", "flash_attention_3"])
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
            "fp8",
        ],
    )
    def test_needs_dtype_cast(self, impl):
        assert impl not in ATTN_IMPLS_WITHOUT_DTYPE_CAST

    def test_known_hub_kernels_classified(self):
        assert "kernels-community/flash-attn3" in ATTN_IMPLS_SUPPORTING_PACKING
        assert "kernels-community/flash-attn3" in ATTN_IMPLS_USING_FLASH_LIB
        assert "kernels-community/sage-attention" in ATTN_IMPLS_SUPPORTING_PACKING

    def test_computed_flags_readable_on_validated_cfg(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(attn_implementation="sdpa")
        validated = validate_config(cfg)
        assert validated.attn_implementation == "sdpa"
        assert validated.attn_supports_packing is False
        assert validated.attn_uses_flash_lib is False
        assert validated.attn_needs_dtype_cast is False

    def test_computed_flags_not_overridable_from_yaml(self, min_base_cfg):
        """YAML attempts to override a computed field must not win."""
        cfg = min_base_cfg | DictDefault(
            attn_implementation="eager", attn_uses_flash_lib=True
        )
        validated = validate_config(cfg)
        # The computed field reflects the backend, not the YAML input.
        assert validated.attn_uses_flash_lib is False


class TestBackendRegistration:
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


class TestSparseAttentionBackends:
    """nsa/fsa: canonical names, packing support, registration stub, MLA-only
    model guard, and compute-capability guard."""

    @pytest.mark.parametrize("impl", ["nsa", "fsa"])
    def test_capability_classification(self, impl):
        assert impl in CANONICAL_ATTN_IMPLS
        assert impl in ATTN_IMPLS_SUPPORTING_PACKING
        assert impl not in ATTN_IMPLS_USING_FLASH_LIB
        assert impl not in ATTN_IMPLS_WITHOUT_DTYPE_CAST

    @pytest.mark.parametrize("impl", ["nsa", "fsa"])
    def test_field_validator_accepts(self, impl):
        assert AxolotlInputConfig.validate_attn_implementation(impl) == impl

    def test_register_sparse_attn(self):
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        from axolotl.monkeypatch.attention import register_sparse_attn

        register_sparse_attn()

        for name in ("nsa", "fsa"):
            assert name in ALL_ATTENTION_FUNCTIONS
            assert (
                ALL_MASK_ATTENTION_FUNCTIONS[name]
                == ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"]
            )

    def test_register_does_not_overwrite_fa2(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        original_fa2 = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        from axolotl.monkeypatch.attention import register_sparse_attn

        register_sparse_attn()

        assert ALL_ATTENTION_FUNCTIONS["flash_attention_2"] is original_fa2

    def test_stub_must_not_be_called(self):
        from axolotl.monkeypatch.attention.sparse_attn import (
            sparse_attention_stub,
        )

        with pytest.raises(NotImplementedError, match="module swap"):
            sparse_attention_stub(None, None, None, None)

    def test_import_guard(self):
        from axolotl.monkeypatch.attention.sparse_attn import (
            _check_fsa_imported,
            _is_fsa_available,
        )

        if _is_fsa_available():
            _check_fsa_imported()
        else:
            with pytest.raises(ImportError, match="Flash-Sparse-Attention"):
                _check_fsa_imported()

    @staticmethod
    def _guard(data):
        return AxolotlInputConfig.check_sparse_attn_requires_mla_model(data)

    @pytest.mark.parametrize("impl", ["nsa", "fsa"])
    @pytest.mark.parametrize(
        "model_type", ["kimi_linear", "deepseek_v2", "deepseek_v3"]
    )
    def test_guard_allows_mla_models(self, impl, model_type):
        data = {"attn_implementation": impl, "model_config_type": model_type}
        assert self._guard(data) is data

    @pytest.mark.parametrize("impl", ["nsa", "fsa"])
    def test_guard_rejects_non_mla(self, impl):
        with pytest.raises(ValueError, match="only.*supported for MLA"):
            self._guard({"attn_implementation": impl, "model_config_type": "llama"})

    def test_guard_defers_when_model_type_unknown(self):
        data = {"attn_implementation": "fsa"}
        assert self._guard(data) is data

    def test_guard_ignores_non_sparse_backend(self):
        data = {"attn_implementation": "sdpa", "model_config_type": "llama"}
        assert self._guard(data) is data

    def test_compute_capability_guard_rejects_old_gpu(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            attn_implementation="fsa", model_config_type="kimi_linear"
        )
        with pytest.raises(ValueError, match="requires compute capability sm_80"):
            validate_config(
                cfg,
                capabilities={"compute_capability": "sm_75"},
                env_capabilities={"torch_version": "2.9.0"},
            )

    def test_compute_capability_guard_allows_ampere(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            attn_implementation="fsa", model_config_type="kimi_linear"
        )
        validated = validate_config(
            cfg,
            capabilities={"compute_capability": "sm_90"},
            env_capabilities={"torch_version": "2.9.0"},
        )
        assert validated.attn_implementation == "fsa"
        assert validated.attn_supports_packing is True

    def test_build_cu_seqlens_varlen_tensor(self):
        """A multi-element cu_seqlens tensor in kwargs must pass through; the
        old ``a or b`` form raised on bool() of a >1-element tensor."""
        import torch

        from axolotl.monkeypatch.attention.sparse_attn import _build_cu_seqlens

        cu = torch.tensor([0, 3, 7], dtype=torch.int64)
        out = _build_cu_seqlens(1, 7, None, {"cu_seq_lens_q": cu}, torch.device("cpu"))
        assert out.dtype == torch.int32
        assert out.tolist() == [0, 3, 7]

    def test_build_cu_seqlens_tuple_pair(self):
        import torch

        from axolotl.monkeypatch.attention.sparse_attn import _build_cu_seqlens

        cu_q = torch.tensor([0, 4, 8], dtype=torch.int32)
        out = _build_cu_seqlens(
            1, 8, None, {"cu_seq_lens_q": (cu_q, cu_q)}, torch.device("cpu")
        )
        assert out.tolist() == [0, 4, 8]

    def test_build_cu_seqlens_from_position_ids(self):
        import torch

        from axolotl.monkeypatch.attention.sparse_attn import _build_cu_seqlens

        # two packed sequences of length 3 and 2 in one row
        position_ids = torch.tensor([[0, 1, 2, 0, 1]])
        out = _build_cu_seqlens(1, 5, position_ids, {}, torch.device("cpu"))
        assert out.tolist() == [0, 3, 5]

    def test_build_cu_seqlens_fallback_per_row(self):
        import torch

        from axolotl.monkeypatch.attention.sparse_attn import _build_cu_seqlens

        out = _build_cu_seqlens(2, 4, None, {}, torch.device("cpu"))
        assert out.tolist() == [0, 4, 8]

    def test_build_cu_seqlens_batched_broadcast_position_ids(self):
        """Non-packing: transformers shares position_ids as (1, q_len) across the
        batch, but the FSA stream is bsz*q_len. Each row must be its own sequence
        (regression for cu=[0, q_len] undercount that crashed the kernel)."""
        import torch

        from axolotl.monkeypatch.attention.sparse_attn import _build_cu_seqlens

        position_ids = torch.arange(3).view(1, 3)  # shared across 4 rows
        out = _build_cu_seqlens(4, 3, position_ids, {}, torch.device("cpu"))
        assert out.tolist() == [0, 3, 6, 9, 12]

    def test_build_cu_seqlens_batched_with_inner_packing(self):
        import torch

        from axolotl.monkeypatch.attention.sparse_attn import _build_cu_seqlens

        # bsz=2, q_len=5, each row packs a len-3 and a len-2 sub-sequence
        position_ids = torch.tensor([[0, 1, 2, 0, 1], [0, 1, 2, 0, 1]])
        out = _build_cu_seqlens(2, 5, position_ids, {}, torch.device("cpu"))
        assert out.tolist() == [0, 3, 5, 8, 10]

    def test_enforce_min_segment_merges_short_tail(self):
        """A packed remainder shorter than kernel_size must merge into the
        previous segment (regression for the FSA compression NaN)."""
        import torch

        from axolotl.monkeypatch.attention.sparse_attn import _enforce_min_segment

        # segments 748, 252, 10 with kernel_size 32 -> the 10 tail merges back
        cu = torch.tensor([0, 748, 1000, 1010], dtype=torch.int32)
        assert _enforce_min_segment(cu, 32).tolist() == [0, 748, 1010]

    def test_enforce_min_segment_merges_interior_short(self):
        import torch

        from axolotl.monkeypatch.attention.sparse_attn import _enforce_min_segment

        # an interior 8-token segment is absorbed into its predecessor
        cu = torch.tensor([0, 100, 108, 400], dtype=torch.int32)
        assert _enforce_min_segment(cu, 32).tolist() == [0, 100, 400]

    def test_enforce_min_segment_noop_when_all_long(self):
        import torch

        from axolotl.monkeypatch.attention.sparse_attn import _enforce_min_segment

        cu = torch.tensor([0, 100, 200, 300], dtype=torch.int32)
        assert _enforce_min_segment(cu, 32).tolist() == [0, 100, 200, 300]

    def test_build_cu_seqlens_min_seg_merges_packed_tail(self):
        import torch

        from axolotl.monkeypatch.attention.sparse_attn import _build_cu_seqlens

        # one row of 50 packed as [0..39] + [0..9]; kernel_size 32 drops the
        # 10-token tail so no segment is shorter than the compression kernel.
        position_ids = torch.tensor([list(range(40)) + list(range(10))])
        out = _build_cu_seqlens(
            1, 50, position_ids, {}, torch.device("cpu"), min_seg=32
        )
        assert out.tolist() == [0, 50]


class TestLegacyFlagDeprecation:
    """Legacy boolean flags (flash_attention, sdp_attention, ...) map to a
    canonical attn_implementation value, are stripped from the validated
    config, and cannot be combined with an explicit canonical value.
    """

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
        ]:
            assert flag not in result

    def test_sage_plus_flash_priority_is_sage(self):
        result = self._normalize({"sage_attention": True, "flash_attention": True})
        assert result["attn_implementation"] == "sage"

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

    def test_legacy_flag_stripped_on_validated_cfg(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(flash_attention=True)
        validated = validate_config(cfg)
        assert validated.attn_implementation == "flash_attention_2"
        # Legacy flag must not survive to the validated DictDefault
        # (normalizer pops it, model_dump excludes Nones).
        assert "flash_attention" not in dict(validated)

    def test_canonical_plus_legacy_rejected_on_full_validation(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            attn_implementation="flash_attention_2", flash_attention=True
        )
        with pytest.raises(ValueError, match="cannot be combined with legacy"):
            validate_config(cfg)


class TestCanonicalValueAcceptance:
    """`attn_implementation` accepts canonical names and `org/name` hub-kernel
    paths. Short-form aliases (`flash`, `flex`, `sdp`) and unknown bare names
    are rejected. Absent input is a noop.
    """

    @staticmethod
    def _normalize(data):
        return AxolotlInputConfig.normalize_attn_implementation(data)

    def test_canonical_value_is_passthrough(self):
        data = {"attn_implementation": "flash_attention_2"}
        result = self._normalize(data)
        assert result["attn_implementation"] == "flash_attention_2"

    def test_hub_kernel_is_passthrough(self):
        data = {"attn_implementation": "kernels-community/flash-attn3"}
        result = self._normalize(data)
        assert result["attn_implementation"] == "kernels-community/flash-attn3"

    def test_no_attention_set_is_noop(self):
        result = self._normalize({"some_other_config": True})
        assert result.get("attn_implementation") is None

    def test_field_validator_accepts_all_canonical(self):
        for impl in CANONICAL_ATTN_IMPLS:
            assert AxolotlInputConfig.validate_attn_implementation(impl) == impl

    def test_field_validator_accepts_hub_kernels(self):
        for impl in (
            "kernels-community/flash-attn3",
            "kernels-community/sage-attention",
            "someorg/custom-kernel",
        ):
            assert AxolotlInputConfig.validate_attn_implementation(impl) == impl

    def test_field_validator_accepts_none(self):
        assert AxolotlInputConfig.validate_attn_implementation(None) is None

    @pytest.mark.parametrize("alias", ["flash", "flex", "sdp"])
    def test_short_form_alias_rejected(self, alias):
        with pytest.raises(ValueError, match="is not accepted"):
            AxolotlInputConfig.validate_attn_implementation(alias)

    def test_unknown_bare_name_rejected(self):
        with pytest.raises(ValueError, match="not a recognized backend"):
            AxolotlInputConfig.validate_attn_implementation("not_a_real_backend")

    def test_canonical_value_passes_through_full_validation(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(attn_implementation="flash_attention_3")
        validated = validate_config(cfg)
        assert validated.attn_implementation == "flash_attention_3"
        assert validated.attn_uses_flash_lib is True
        assert validated.attn_supports_packing is True

    def test_hub_kernel_passes_through_full_validation(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            attn_implementation="kernels-community/flash-attn3"
        )
        validated = validate_config(cfg)
        assert validated.attn_implementation == "kernels-community/flash-attn3"
        assert validated.attn_uses_flash_lib is True
        assert validated.attn_supports_packing is True

    def test_short_form_alias_rejected_on_full_validation(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(attn_implementation="flash")
        with pytest.raises(ValueError, match="is not accepted"):
            validate_config(cfg)


class TestGemma4HybridMode:
    """`gemma4_hybrid_attn_impl` pins `attn_implementation` to `flash_attention_2`."""

    @staticmethod
    def _normalize(data):
        return AxolotlInputConfig.normalize_attn_implementation(data)

    def test_defaults_to_flash_attention_2(self):
        result = self._normalize({"gemma4_hybrid_attn_impl": True})
        assert result["attn_implementation"] == "flash_attention_2"

    def test_explicit_fa2_passes(self):
        result = self._normalize(
            {
                "gemma4_hybrid_attn_impl": True,
                "attn_implementation": "flash_attention_2",
            }
        )
        assert result["attn_implementation"] == "flash_attention_2"

    def test_non_fa2_raises(self):
        with pytest.raises(
            ValueError, match="requires attn_implementation=flash_attention_2"
        ):
            self._normalize(
                {"gemma4_hybrid_attn_impl": True, "attn_implementation": "sdpa"}
            )


class TestSamplePackingValidation:
    """`sample_packing` warns for non-varlen backends; s2 raises outright."""

    def test_eager_warns(self, min_base_cfg, caplog):
        cfg = min_base_cfg | DictDefault(
            attn_implementation="eager", sample_packing=True
        )
        with _capture_axolotl_warnings(caplog):
            validate_config(cfg)
        assert any(
            "does not handle cross-sample decontamination" in r.getMessage()
            for r in caplog.records
        )

    def test_sdpa_warns(self, min_base_cfg, caplog):
        cfg = min_base_cfg | DictDefault(
            attn_implementation="sdpa", sample_packing=True
        )
        with _capture_axolotl_warnings(caplog):
            validate_config(cfg)
        assert any(
            "does not handle cross-sample decontamination" in r.getMessage()
            for r in caplog.records
        )

    def test_flash_attention_2_does_not_warn(self, min_base_cfg, caplog):
        cfg = min_base_cfg | DictDefault(
            attn_implementation="flash_attention_2", sample_packing=True
        )
        with _capture_axolotl_warnings(caplog):
            validate_config(cfg)
        assert not any(
            "does not handle cross-sample decontamination" in r.getMessage()
            for r in caplog.records
        )


class TestScalingSoftmaxValidation:
    """`scaling_softmax` is only implemented under flex_attention."""

    def test_non_flex_raises(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            attn_implementation="flash_attention_2", scaling_softmax=True
        )
        with pytest.raises(ValueError, match="scaling_softmax requires flex"):
            validate_config(cfg)

    def test_flex_passes(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            attn_implementation="flex_attention", scaling_softmax=True
        )
        validated = validate_config(cfg)
        assert validated.attn_implementation == "flex_attention"
