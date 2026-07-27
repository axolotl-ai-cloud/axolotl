"""Offline contracts for declarative transformers-registry payloads."""

import logging

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
    ModelFamilyTemplate,
    ModelProfile,
    ModelRegistrationOverrides,
    ModelRegistrations,
    ModelSupport,
    resolve_model_support,
)
from axolotl.utils.dict import DictDefault


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
        with caplog.at_level(logging.WARNING):
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
