"""Tests for the per-architecture model support registry."""

import pytest
from transformers import AutoModelForImageTextToText

from axolotl.model_support import (
    ModelSupport,
    get_model_support,
    get_model_support_for_processor,
    register_model_support,
    registry as model_support_registry,
)
from axolotl.utils.dict import DictDefault


class TestRegistry:
    """Registration and lookup semantics."""

    def test_unknown_model_type_returns_none(self):
        assert get_model_support("llama") is None
        assert get_model_support(None) is None

    def test_unmatched_processor_returns_none(self):
        assert get_model_support_for_processor(object()) is None

    def test_register_requires_model_types(self):
        class MissingTypes(ModelSupport):
            pass

        with pytest.raises(ValueError, match="model_types"):
            register_model_support(MissingTypes)

    def test_register_custom_descriptor(self):
        class CustomSupport(ModelSupport):
            model_types = ("my_custom_arch",)
            supports_liger = False

        try:
            register_model_support(CustomSupport)
            support = get_model_support("my_custom_arch")
            assert isinstance(support, CustomSupport)
            assert support.supports_liger is False
            # capability default is tri-state unknown
            assert support.supports_cut_cross_entropy is None
        finally:
            model_support_registry._REGISTRY.pop("my_custom_arch", None)


class TestKimiLinearSupport:
    """Built-in Kimi-Linear descriptor: cfg-based matching for remote-code patching."""

    def test_matches_cfg_by_model_name(self):
        from axolotl.model_support.registry import get_model_support_for_cfg

        cfg = DictDefault(base_model_config="moonshotai/Kimi-Linear-48B-A3B-Instruct")
        support = get_model_support_for_cfg(cfg)
        assert support is not None
        assert support is get_model_support("kimi_linear")

    def test_no_match_for_other_models(self):
        from axolotl.model_support.registry import get_model_support_for_cfg

        cfg = DictDefault(base_model_config="meta-llama/Llama-3.1-8B-Instruct")
        assert get_model_support_for_cfg(cfg) is None

    def test_pre_config_load_patches_dynamic_module_loading(self):
        from transformers.dynamic_module_utils import get_class_in_module

        cfg = DictDefault(base_model_config="moonshotai/Kimi-Linear-48B-A3B-Instruct")
        get_model_support("kimi_linear").pre_config_load(cfg)

        import transformers.dynamic_module_utils

        assert getattr(
            transformers.dynamic_module_utils.get_class_in_module,
            "_axolotl_patched",
            False,
        )
        del get_class_in_module  # silence unused; imported pre-patch for clarity


class TestPaddleOCRVLSupport:
    """Built-in PaddleOCR-VL descriptor and the generic capability guards."""

    def test_registered_and_multimodal(self):
        support = get_model_support("paddleocr_vl")
        assert support is not None
        assert support.is_multimodal is True

    def test_auto_model_cls(self):
        support = get_model_support("paddleocr_vl")
        assert support.get_auto_model_cls() is AutoModelForImageTextToText

    def test_processing_strategy_cls(self):
        from axolotl.model_support.paddleocr_vl.processing import (
            PaddleOCRVLProcessingStrategy,
        )

        support = get_model_support("paddleocr_vl")
        assert support.get_processing_strategy_cls() is PaddleOCRVLProcessingStrategy

    def test_cut_cross_entropy_rejected(self):
        from axolotl.integrations.cut_cross_entropy import CutCrossEntropyPlugin

        cfg = DictDefault(
            model_config_type="paddleocr_vl",
            cut_cross_entropy=True,
        )
        with pytest.raises(ValueError, match="paddleocr_vl"):
            CutCrossEntropyPlugin().pre_model_load(cfg)

    def test_cut_cross_entropy_disabled_is_noop(self):
        from axolotl.integrations.cut_cross_entropy import CutCrossEntropyPlugin

        cfg = DictDefault(
            model_config_type="paddleocr_vl",
            cut_cross_entropy=False,
        )
        CutCrossEntropyPlugin().pre_model_load(cfg)

    @pytest.mark.parametrize(
        "flags",
        [
            {"liger_cross_entropy": True},
            {"liger_fused_linear_cross_entropy": True},
            {"liger_glu_activation": True},
        ],
    )
    def test_liger_rejected(self, flags):
        from axolotl.integrations.liger.plugin import LigerPlugin

        cfg = DictDefault(model_config_type="paddleocr_vl")
        cfg.update(flags)
        with pytest.raises(ValueError, match="Liger is not supported"):
            LigerPlugin().pre_model_load(cfg)
