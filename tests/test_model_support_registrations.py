"""Offline contracts for declarative transformers-registry payloads."""

import pytest
from torch import nn
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import (
    ConversionOps,
    WeightConverter,
    WeightRenaming,
)
from transformers.monkey_patching import get_patch_mapping, unregister_patch_mapping

import axolotl.loaders.patch_manager as patch_manager_module
from axolotl.model_support import (
    VANILLA_CAUSAL_LM,
    AutoClassRegistration,
    ModelFamilyTemplate,
    ModelHooks,
    ModelProfile,
    ModelRegistrationOverrides,
    ModelRegistrations,
    ModelStrategyOverrides,
    ModelSupport,
    QuantizerRegistration,
    resolve_model_support,
)
from axolotl.utils.dict import DictDefault

from tests.conftest import capture_axolotl_warnings


class _Replacement(nn.Module):
    pass


class _IrreversibleOp(ConversionOps):
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        return input_dict


def _patch_manager_for(monkeypatch, support, model_type):
    monkeypatch.setattr(
        patch_manager_module,
        "get_model_support",
        lambda candidate: support if candidate == model_type else None,
    )
    return patch_manager_module.PatchManager(
        DictDefault(model_config_type=model_type),
        DictDefault(model_type=model_type),
    )


def test_registrations_resolution_inherits_overrides_and_clears():
    def family_conversions():
        return {}

    def profile_patches():
        return {}

    family = ModelFamilyTemplate(
        name="registrations_family",
        registrations=ModelRegistrations(weight_conversions=family_conversions),
    )

    class InheritingSupport(ModelSupport):
        model_types = ("registrations_inherit_test",)
        profile = ModelProfile(
            family=family,
            registrations=ModelRegistrationOverrides(patch_mappings=profile_patches),
        )

    class ClearingSupport(ModelSupport):
        model_types = ("registrations_clear_test",)
        profile = ModelProfile(
            family=family,
            registrations=ModelRegistrationOverrides(weight_conversions=None),
        )

    inherited = resolve_model_support(InheritingSupport())
    assert inherited.registrations.weight_conversions is family_conversions
    assert inherited.registrations.patch_mappings is profile_patches

    cleared = resolve_model_support(ClearingSupport())
    assert cleared.registrations.weight_conversions is None
    assert cleared.registrations.patch_mappings is None

    assert resolve_model_support(ModelSupport()).registrations == ModelRegistrations()


def test_pre_model_load_registers_weight_conversions_and_patch_mappings(monkeypatch):
    from transformers.conversion_mapping import register_checkpoint_conversion_mapping

    class RegisteringSupport(ModelSupport):
        model_types = ("registrations_dispatch_test",)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            registrations=ModelRegistrationOverrides(
                weight_conversions=lambda: {
                    "registrations_dispatch_test": [
                        WeightRenaming(["legacy_embed.weight"], ["embed.weight"])
                    ]
                },
                patch_mappings=lambda: {
                    "RegistrationsDispatchTestModule": _Replacement
                },
            ),
        )

    manager = _patch_manager_for(
        monkeypatch, RegisteringSupport(), "registrations_dispatch_test"
    )
    try:
        manager._apply_model_support_registrations()

        registered = get_checkpoint_conversion_mapping("registrations_dispatch_test")
        assert registered is not None
        assert [entry.source_patterns for entry in registered] == [
            ["legacy_embed.weight"]
        ]
        assert get_patch_mapping()["RegistrationsDispatchTestModule"] is _Replacement

        # Idempotent: re-registration must not raise on the existing entries.
        manager._apply_model_support_registrations()
    finally:
        unregister_patch_mapping(["RegistrationsDispatchTestModule"])
        register_checkpoint_conversion_mapping(
            "registrations_dispatch_test", [], overwrite=True
        )


def test_irreversible_weight_conversions_warn_at_registration(monkeypatch, caplog):
    from transformers.conversion_mapping import register_checkpoint_conversion_mapping

    class WarningSupport(ModelSupport):
        model_types = ("registrations_warn_test",)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            registrations=ModelRegistrationOverrides(
                weight_conversions=lambda: {
                    "registrations_warn_test": [
                        WeightConverter(
                            ["experts.gate_proj.weight"],
                            ["experts.gate_up_proj.weight"],
                            operations=[_IrreversibleOp()],
                        )
                    ]
                },
            ),
        )

    manager = _patch_manager_for(
        monkeypatch, WarningSupport(), "registrations_warn_test"
    )
    try:
        with capture_axolotl_warnings(caplog):
            manager._apply_model_support_registrations()
    finally:
        register_checkpoint_conversion_mapping(
            "registrations_warn_test", [], overwrite=True
        )

    assert "cannot be reversed at save time" in caplog.text
    assert "_IrreversibleOp" in caplog.text


def test_registration_free_profiles_do_not_touch_transformers(monkeypatch):
    class PlainSupport(ModelSupport):
        model_types = ("registrations_plain_test",)
        profile = ModelProfile(family=VANILLA_CAUSAL_LM)

    manager = _patch_manager_for(
        monkeypatch, PlainSupport(), "registrations_plain_test"
    )
    before = dict(get_patch_mapping())

    manager._apply_model_support_registrations()

    assert dict(get_patch_mapping()) == before
    assert get_checkpoint_conversion_mapping("registrations_plain_test") is None

    unregistered = _patch_manager_for(monkeypatch, None, "registrations_absent_test")
    unregistered._apply_model_support_registrations()
    assert get_checkpoint_conversion_mapping("registrations_absent_test") is None


def _fake_attention_fn(*args, **kwargs):
    return None


def _fake_mask_fn(*args, **kwargs):
    return None


def _fake_experts_fn(*args, **kwargs):
    return None


def test_interface_functions_register_into_transformers_interfaces(monkeypatch):
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    class InterfaceSupport(ModelSupport):
        model_types = ("registrations_interface_test",)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            registrations=ModelRegistrationOverrides(
                attention_functions=lambda: {
                    "registrations_test_attn": _fake_attention_fn
                },
                attention_mask_functions=lambda: {
                    "registrations_test_attn": _fake_mask_fn
                },
                experts_functions=lambda: {
                    "registrations_test_experts": _fake_experts_fn
                },
            ),
        )

    manager = _patch_manager_for(
        monkeypatch, InterfaceSupport(), "registrations_interface_test"
    )
    try:
        manager._apply_model_support_registrations()
        manager._apply_model_support_registrations()

        assert ALL_ATTENTION_FUNCTIONS["registrations_test_attn"] is _fake_attention_fn
        assert ALL_MASK_ATTENTION_FUNCTIONS["registrations_test_attn"] is _fake_mask_fn
        assert ALL_EXPERTS_FUNCTIONS["registrations_test_experts"] is _fake_experts_fn
    finally:
        ALL_ATTENTION_FUNCTIONS._global_mapping.pop("registrations_test_attn", None)
        ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(
            "registrations_test_attn", None
        )
        ALL_EXPERTS_FUNCTIONS._global_mapping.pop("registrations_test_experts", None)


def test_quantizer_registration_is_idempotent_and_overrides_with_warning(
    monkeypatch, caplog
):
    from transformers.quantizers.auto import (
        AUTO_QUANTIZATION_CONFIG_MAPPING,
        AUTO_QUANTIZER_MAPPING,
    )
    from transformers.quantizers.base import HfQuantizer
    from transformers.utils.quantization_config import QuantizationConfigMixin

    class _FakeQuantizer(HfQuantizer):
        def _process_model_before_weight_loading(self, model, **kwargs):
            return model

        def _process_model_after_weight_loading(self, model, **kwargs):
            return model

        def is_serializable(self, safe_serialization=None):
            return True

        @property
        def is_trainable(self):
            return True

    class _OtherQuantizer(_FakeQuantizer):
        pass

    class _FakeQuantConfig(QuantizationConfigMixin):
        pass

    class QuantizerSupport(ModelSupport):
        model_types = ("registrations_quantizer_test",)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            registrations=ModelRegistrationOverrides(
                quantizers=lambda: {
                    "registrations_test_method": QuantizerRegistration(
                        quantizer_cls=_FakeQuantizer,
                        config_cls=_FakeQuantConfig,
                    )
                },
            ),
        )

    manager = _patch_manager_for(
        monkeypatch, QuantizerSupport(), "registrations_quantizer_test"
    )
    try:
        manager._apply_model_support_registrations()
        assert AUTO_QUANTIZER_MAPPING["registrations_test_method"] is _FakeQuantizer
        assert (
            AUTO_QUANTIZATION_CONFIG_MAPPING["registrations_test_method"]
            is _FakeQuantConfig
        )

        with capture_axolotl_warnings(caplog):
            manager._apply_model_support_registrations()
        assert "Overriding quantizer" not in caplog.text

        AUTO_QUANTIZER_MAPPING["registrations_test_method"] = _OtherQuantizer
        with capture_axolotl_warnings(caplog):
            manager._apply_model_support_registrations()
        assert "Overriding quantizer" in caplog.text
        assert AUTO_QUANTIZER_MAPPING["registrations_test_method"] is _FakeQuantizer
    finally:
        AUTO_QUANTIZER_MAPPING.pop("registrations_test_method", None)
        AUTO_QUANTIZATION_CONFIG_MAPPING.pop("registrations_test_method", None)


def test_auto_class_registration_wires_config_model_and_processor(monkeypatch):
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

    class _FakeAutoConfig(PreTrainedConfig):
        model_type = "registrations_auto_test"

    class _FakeAutoModel:
        config_class = _FakeAutoConfig

    class AutoClassSupport(ModelSupport):
        model_types = ("registrations_auto_test",)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            registrations=ModelRegistrationOverrides(
                auto_classes=lambda: [
                    AutoClassRegistration(
                        config_cls=_FakeAutoConfig,
                        model_classes={"AutoModelForCausalLM": _FakeAutoModel},
                    )
                ],
            ),
        )

    manager = _patch_manager_for(
        monkeypatch, AutoClassSupport(), "registrations_auto_test"
    )
    try:
        manager._apply_model_support_registrations()
        manager._apply_model_support_registrations()

        assert AutoConfig.for_model("registrations_auto_test").__class__ is (
            _FakeAutoConfig
        )
        assert AutoModelForCausalLM._model_mapping[_FakeAutoConfig] is _FakeAutoModel
    finally:
        CONFIG_MAPPING._extra_content.pop("registrations_auto_test", None)
        MODEL_FOR_CAUSAL_LM_MAPPING._extra_content.pop(_FakeAutoConfig, None)


def test_unknown_auto_class_name_raises(monkeypatch):
    from transformers import PreTrainedConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    class _BadAutoConfig(PreTrainedConfig):
        model_type = "registrations_bad_auto_test"

    class BadAutoClassSupport(ModelSupport):
        model_types = ("registrations_bad_auto_test",)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            registrations=ModelRegistrationOverrides(
                auto_classes=lambda: [
                    AutoClassRegistration(
                        config_cls=_BadAutoConfig,
                        model_classes={"AutoModelForNoSuchTask": object},
                    )
                ],
            ),
        )

    manager = _patch_manager_for(
        monkeypatch, BadAutoClassSupport(), "registrations_bad_auto_test"
    )
    try:
        with pytest.raises(ValueError, match="AutoModelForNoSuchTask"):
            manager._apply_model_support_registrations()
    finally:
        CONFIG_MAPPING._extra_content.pop("registrations_bad_auto_test", None)


def test_model_class_attrs_are_applied_with_setattr(monkeypatch):
    class _PatchTarget:
        _no_split_modules = None

    class ClassAttrSupport(ModelSupport):
        model_types = ("registrations_class_attrs_test",)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            registrations=ModelRegistrationOverrides(
                model_class_attrs=lambda: {
                    _PatchTarget: {
                        "_no_split_modules": ["FakeDecoderLayer"],
                        "supports_gradient_checkpointing": True,
                    }
                },
            ),
        )

    manager = _patch_manager_for(
        monkeypatch, ClassAttrSupport(), "registrations_class_attrs_test"
    )
    manager._apply_model_support_registrations()

    assert _PatchTarget._no_split_modules == ["FakeDecoderLayer"]
    assert _PatchTarget.supports_gradient_checkpointing is True


def test_loss_function_strategy_sets_per_instance_loss(monkeypatch):
    def _custom_loss(*args, **kwargs):
        return None

    class LossSupport(ModelSupport):
        model_types = ("registrations_loss_test",)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            strategies=ModelStrategyOverrides(loss_function=lambda: _custom_loss),
        )

    class _FakeModel:
        pass

    manager = _patch_manager_for(monkeypatch, LossSupport(), "registrations_loss_test")
    model = _FakeModel()
    manager._apply_model_support_loss_function(model)
    assert model.loss_function is _custom_loss

    plain_manager = _patch_manager_for(
        monkeypatch, None, "registrations_loss_absent_test"
    )
    plain_model = _FakeModel()
    plain_manager._apply_model_support_loss_function(plain_model)
    assert not hasattr(plain_model, "loss_function")


def test_before_save_hooks_dispatch_from_save_trained_model():
    from axolotl.model_support import ModelHookContext, ModelHookPhase
    from axolotl.train import save_trained_model

    events = []

    def _on_save(context: ModelHookContext) -> None:
        events.append((context.model, context.processor))

    class SaveHookSupport(ModelSupport):
        model_types = ("registrations_save_test",)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            hooks=ModelHooks({ModelHookPhase.BEFORE_SAVE: (_on_save,)}),
        )

    class _FakeTrainedModel:
        def named_modules(self):
            return []

    import axolotl.train as train_module

    support = SaveHookSupport()
    original = train_module.get_model_support
    train_module.get_model_support = lambda model_type: (
        support if model_type == "registrations_save_test" else None
    )
    model = _FakeTrainedModel()
    processor = object()
    try:
        # relora with no merge_and_unload returns right after the save hooks
        save_trained_model(
            DictDefault(model_config_type="registrations_save_test", relora=True),
            trainer=None,
            model=model,
            processor=processor,
        )
    finally:
        train_module.get_model_support = original

    assert events == [(model, processor)]
