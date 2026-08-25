"""Tests for the per-architecture model support registry."""

import pytest
from transformers import AutoModelForImageTextToText

from axolotl.model_support import (
    Experimental,
    ModelHookContext,
    ModelHookPhase,
    ModelSupport,
    Unsupported,
    check_capability,
    get_model_support,
    get_model_support_for_processor,
    register_model_support,
    registry as model_support_registry,
    resolve_model_support,
    run_model_support_hooks,
)
from axolotl.utils.dict import DictDefault

from tests.conftest import capture_axolotl_warnings


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
        # an earlier configure_logging() sets propagate=False, blinding caplog
        with capture_axolotl_warnings(caplog):
            check_capability(
                self._Support(), "sample_packing", "cap_test_arch", feature="packing"
            )
        assert any("Verify loss parity" in r.getMessage() for r in caplog.records)

    def test_unknown_capability_and_missing_descriptor_are_noops(self):
        check_capability(self._Support(), "liger", "cap_test_arch")
        check_capability(None, "liger", "cap_test_arch")


class TestKimiLinearSupport:
    """Built-in Kimi-Linear descriptor: cfg-based matching for remote-code patching."""

    @pytest.fixture
    def restore_dynamic_module_loader(self):
        import transformers.dynamic_module_utils as dynamic_module_utils

        original = dynamic_module_utils.get_class_in_module
        try:
            yield dynamic_module_utils
        finally:
            dynamic_module_utils.get_class_in_module = original

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

    def test_resolves_vanilla_family_and_declared_hooks(self):
        resolved = resolve_model_support(get_model_support("kimi_linear"))

        assert resolved.family == "vanilla_causal_lm"
        assert resolved.hooks.for_phase(ModelHookPhase.BEFORE_CONFIG_LOAD)
        assert resolved.hooks.for_phase(ModelHookPhase.BEFORE_TOKENIZER_LOAD)
        assert resolved.hooks.for_phase(ModelHookPhase.BEFORE_MODEL_BUILD)

    def test_dynamic_module_redirect_requires_exact_module_stem(self):
        from axolotl.model_support.kimi_linear.patch_kimi_linear import KIMI_MODULES
        from axolotl.model_support.remote_code import owned_stem

        def entry(module_path):
            return owned_stem(KIMI_MODULES, module_path)

        assert entry("transformers_modules/x/repo/modeling_kimi.py")
        assert entry("transformers_modules.x.repo.modeling_kimi")
        assert entry("tokenization_kimi.py")
        assert entry("transformers_modules/x/modeling_kimi_vl.py") is None
        assert entry("modeling_kimi_vl") is None

    def test_pre_config_load_patches_dynamic_module_loading(
        self, restore_dynamic_module_loader
    ):
        cfg = DictDefault(base_model_config="moonshotai/Kimi-Linear-48B-A3B-Instruct")
        support = get_model_support("kimi_linear")
        context = ModelHookContext(cfg=cfg)
        run_model_support_hooks(
            support,
            ModelHookPhase.BEFORE_CONFIG_LOAD,
            context,
        )
        patched = restore_dynamic_module_loader.get_class_in_module
        run_model_support_hooks(
            support,
            ModelHookPhase.BEFORE_CONFIG_LOAD,
            context,
        )

        assert getattr(
            restore_dynamic_module_loader.get_class_in_module,
            "_axolotl_patched",
            False,
        )
        assert restore_dynamic_module_loader.get_class_in_module is patched


class TestPaddleOCRVLSupport:
    """Built-in PaddleOCR-VL descriptor and the generic capability guards."""

    def test_registered_and_multimodal(self):
        support = get_model_support("paddleocr_vl")
        assert support is not None
        resolved = resolve_model_support(support)
        assert resolved.is_multimodal is True
        assert resolved.family == "image_text_to_text"
        assert support.is_multimodal is True
        assert set(support.capabilities) == {
            "cut_cross_entropy",
            "liger",
            "lora_kernels",
        }

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
        # the auto-enable validator lives on AxolotlConfigWCapabilities; a bare
        # validate_config(cfg) never runs it
        cfg = validate_config(
            cfg,
            capabilities={"n_gpu": 1, "bf16": True, "compute_capability": None},
            env_capabilities={"torch_version": "2.9.0"},
        )
        assert not any(
            cfg.get(k)
            for k in (
                "lora_mlp_kernel",
                "lora_qkv_kernel",
                "lora_o_kernel",
                "lora_embedding_kernel",
            )
        )

    def test_lora_kernels_validator_sees_profile_capabilities_of_hybrid_descriptor(
        self,
    ):
        """A legacy class-level ``capabilities`` shadows the profile projection."""
        from axolotl.model_support import VANILLA_CAUSAL_LM, ModelProfile
        from axolotl.utils.config import validate_config

        class HybridSupport(ModelSupport):
            model_types = ("hybrid_caps_arch",)
            profile = ModelProfile(
                family=VANILLA_CAUSAL_LM,
                capabilities={"lora_kernels": Unsupported("No fused-QKV rewrite.")},
            )
            capabilities = {"cut_cross_entropy": Unsupported()}

        cfg = DictDefault(
            {
                "base_model": "fake/hybrid-caps-model",
                "model_config_type": "hybrid_caps_arch",
                "learning_rate": 0.000001,
                "datasets": [{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "adapter": "qlora",
                "load_in_4bit": True,
            }
        )
        try:
            register_model_support(HybridSupport)
            cfg = validate_config(
                cfg,
                capabilities={"n_gpu": 1, "bf16": True, "compute_capability": None},
                env_capabilities={"torch_version": "2.9.0"},
            )
            assert not any(
                cfg.get(k)
                for k in (
                    "lora_mlp_kernel",
                    "lora_qkv_kernel",
                    "lora_o_kernel",
                    "lora_embedding_kernel",
                )
            )
        finally:
            model_support_registry._REGISTRY.pop("hybrid_caps_arch", None)

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


class TestCohereCompassSupport:
    """Built-in CohereCompass descriptor (North Micro Vision Instruct and siblings)."""

    def test_registered_and_multimodal(self):
        support = get_model_support("cohere_compass")
        assert support is not None
        resolved = resolve_model_support(support)
        assert resolved.is_multimodal is True
        assert resolved.family == "image_text_to_text"
        assert set(support.capabilities) == {
            "cut_cross_entropy",
            "liger",
            "lora_kernels",
        }

    def test_auto_model_cls(self):
        support = get_model_support("cohere_compass")
        assert support.get_auto_model_cls() is AutoModelForImageTextToText

    def test_processing_strategy_cls(self):
        from axolotl.model_support.cohere_compass.processing import (
            CohereCompassProcessingStrategy,
        )

        support = get_model_support("cohere_compass")
        assert support.get_processing_strategy_cls() is CohereCompassProcessingStrategy

    def test_verified_capabilities_not_marked_unsupported(self):
        """CCE, LoRA kernels and Liger were all verified on hardware for this arch."""
        from axolotl.model_support.base import Unsupported

        caps = resolve_model_support(get_model_support("cohere_compass")).capabilities
        for name in ("cut_cross_entropy", "lora_kernels", "liger"):
            assert not isinstance(caps[name], Unsupported), name


class TestMuseGlimmerSupport:
    """Built-in Muse Glimmer descriptor and its declared capability guards."""

    def test_registered_and_multimodal(self):
        support = get_model_support("muse_glimmer")
        assert support is not None
        resolved = resolve_model_support(support)
        assert resolved.is_multimodal is True
        assert resolved.family == "image_text_to_text"
        assert support.is_multimodal is True
        assert set(support.capabilities) == {
            "cut_cross_entropy",
            "liger",
            "lora_kernels",
        }

    def test_auto_model_cls(self):
        support = get_model_support("muse_glimmer")
        assert support.get_auto_model_cls() is AutoModelForImageTextToText

    def test_processing_strategy_cls(self):
        from axolotl.model_support.muse_glimmer.processing import (
            MuseGlimmerProcessingStrategy,
        )

        support = get_model_support("muse_glimmer")
        assert support.get_processing_strategy_cls() is MuseGlimmerProcessingStrategy

    def test_cut_cross_entropy_supported(self):
        """The fork patches MuseGlimmerForConditionalGeneration directly and reproduces
        the output_multiplier scaling plus the tanh softcap inside apply_lce."""
        support = get_model_support("muse_glimmer")
        check_capability(support, "cut_cross_entropy", "muse_glimmer")

    def test_lora_kernels_rejected(self):
        """The fused QKV/O rewrite cannot express the sigmoid-gated attention output."""
        support = get_model_support("muse_glimmer")
        with pytest.raises(ValueError, match="muse_glimmer"):
            check_capability(support, "lora_kernels", "muse_glimmer")


class TestBailingHybridSupport:
    """Built-in Ling 3.0 descriptor: cfg matching, remote-code redirect, conversions."""

    def test_registered_and_vanilla_family(self):
        support = get_model_support("bailing_hybrid")
        assert support is not None
        assert type(support).__name__ == "BailingHybridSupport"
        assert resolve_model_support(support).family == "vanilla_causal_lm"

    def test_matches_cfg_by_model_name(self):
        from axolotl.model_support.registry import get_model_support_for_cfg

        cfg = DictDefault(base_model_config="inclusionAI/Ling-3.0-flash")
        assert get_model_support_for_cfg(cfg) is get_model_support("bailing_hybrid")

    def test_no_match_for_other_models(self):
        from axolotl.model_support.registry import get_model_support_for_cfg

        cfg = DictDefault(base_model_config="Qwen/Qwen3-30B-A3B")
        assert get_model_support_for_cfg(cfg) is None

    def test_redirects_remote_code_at_every_pre_load_phase(self):
        support = get_model_support("bailing_hybrid")
        hooks = resolve_model_support(support).hooks
        for phase in (
            ModelHookPhase.BEFORE_CONFIG_LOAD,
            ModelHookPhase.BEFORE_TOKENIZER_LOAD,
            ModelHookPhase.BEFORE_MODEL_BUILD,
        ):
            assert hooks.for_phase(phase)

    def test_remote_code_resolves_to_the_in_tree_config(self):
        import transformers.dynamic_module_utils as dynamic_module_utils

        from axolotl.model_support.bailing_hybrid.configuration_bailing_moe_v3 import (
            BailingMoeV3Config,
        )

        original = dynamic_module_utils.get_class_in_module
        try:
            run_model_support_hooks(
                get_model_support("bailing_hybrid"),
                ModelHookPhase.BEFORE_CONFIG_LOAD,
                ModelHookContext(
                    cfg=DictDefault(base_model_config="inclusionAI/Ling-3.0-flash")
                ),
            )
            resolved = dynamic_module_utils.get_class_in_module(
                "BailingMoeV3Config",
                "transformers_modules/inclusionAI/Ling-3.0-flash/abc/configuration_bailing_moe_v3.py",
            )
        finally:
            dynamic_module_utils.get_class_in_module = original

        assert resolved is BailingMoeV3Config

    def test_weight_conversions_are_reversible(self):
        """``save_pretrained`` reverses these to re-emit the published layout."""
        from axolotl.model_support.bailing_hybrid import _weight_conversions

        transforms = _weight_conversions()["bailing_hybrid"]
        assert transforms
        for transform in transforms:
            for operation in getattr(transform, "operations", None) or []:
                assert operation.reverse_op is not None
