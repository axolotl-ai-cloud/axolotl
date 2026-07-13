"""Tests for the per-architecture model support registry."""

import pytest
from transformers import AutoModelForImageTextToText

from axolotl.model_support import (
    Experimental,
    ModelSupport,
    Unsupported,
    check_capability,
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
            capabilities = {"liger": Unsupported()}

        try:
            register_model_support(CustomSupport)
            support = get_model_support("my_custom_arch")
            assert isinstance(support, CustomSupport)
            assert isinstance(support.capabilities["liger"], Unsupported)
            # a missing key means unknown: features use their generic fallback
            assert support.capabilities.get("cut_cross_entropy") is None
        finally:
            model_support_registry._REGISTRY.pop("my_custom_arch", None)


class TestCheckCapability:
    """Capability enforcement: raise on Unsupported, warn on Experimental."""

    class _Support(ModelSupport):
        model_types = ("cap_test_arch",)
        capabilities = {
            "cut_cross_entropy": Unsupported("No CCE forward implementation."),
            "sample_packing": Experimental("Verify loss parity vs unpacked."),
        }

    def test_unsupported_raises_with_reason_and_hint(self):
        with pytest.raises(ValueError, match="No CCE forward implementation"):
            check_capability(
                self._Support(),
                "cut_cross_entropy",
                "cap_test_arch",
                hint="Disable cut_cross_entropy for this model.",
            )

    def test_experimental_warns_and_does_not_raise(self, caplog):
        with caplog.at_level("WARNING", logger="axolotl"):
            check_capability(
                self._Support(), "sample_packing", "cap_test_arch", feature="packing"
            )
        assert any("Verify loss parity" in r.getMessage() for r in caplog.records)

    def test_unknown_capability_and_missing_descriptor_are_noops(self):
        check_capability(self._Support(), "liger", "cap_test_arch")
        check_capability(None, "liger", "cap_test_arch")


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

    def test_lora_kernels_not_auto_enabled(self):
        from axolotl.utils.config import validate_config

        cfg = DictDefault(
            {
                "base_model": "PaddlePaddle/PaddleOCR-VL-1.6",
                "model_config_type": "paddleocr_vl",
                "learning_rate": 0.000001,
                "datasets": [{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "adapter": "qlora",
                "load_in_4bit": True,
            }
        )
        cfg = validate_config(cfg)
        assert not any(
            cfg.get(k)
            for k in (
                "lora_mlp_kernel",
                "lora_qkv_kernel",
                "lora_o_kernel",
                "lora_embedding_kernel",
            )
        )

    def test_normalize_config_disables_lora_kernels(self):
        """model_type is usually unknown when the auto-enable validator runs;
        normalize_config must turn the kernels back off once it is resolved."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from axolotl.utils.config import normalize_config

        cfg = DictDefault(
            {
                "base_model": "PaddlePaddle/PaddleOCR-VL-1.6",
                "num_epochs": 1,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "adapter": "qlora",
                "load_in_4bit": True,
                "lora_mlp_kernel": True,
                "lora_qkv_kernel": True,
                "lora_o_kernel": True,
                "lora_embedding_kernel": True,
            }
        )
        with patch(
            "axolotl.utils.config.load_model_config",
            return_value=SimpleNamespace(model_type="paddleocr_vl"),
        ):
            normalize_config(cfg)
        assert not any(
            cfg[k]
            for k in (
                "lora_mlp_kernel",
                "lora_qkv_kernel",
                "lora_o_kernel",
                "lora_embedding_kernel",
            )
        )

    def test_explicit_lora_qkv_kernel_rejected(self):
        from axolotl.loaders.patch_manager import PatchManager

        cfg = DictDefault(
            model_config_type="paddleocr_vl",
            lora_qkv_kernel=True,
        )
        patch_manager = PatchManager(cfg, DictDefault())
        with pytest.raises(
            ValueError, match="not supported for model_type=paddleocr_vl"
        ):
            patch_manager._apply_self_attention_lora_patch()
