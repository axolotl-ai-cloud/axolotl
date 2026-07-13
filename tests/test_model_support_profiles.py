"""Offline contracts for model-family templates and resolved model profiles."""

from collections.abc import Mapping
from typing import cast

import pytest

from axolotl.model_support import (
    IMAGE_TEXT_TO_TEXT,
    VANILLA_CAUSAL_LM,
    Experimental,
    ModelFamilyTemplate,
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelMatchers,
    ModelProfile,
    ModelStrategies,
    ModelStrategyOverrides,
    ModelSupport,
    Supported,
    Unsupported,
    check_capability,
    resolve_model_support,
    run_model_support_hooks,
)
from axolotl.utils.dict import DictDefault


def _provider(component_cls):
    return lambda: component_cls


def test_builtin_families_describe_the_vanilla_loading_paths():
    class CausalSupport(ModelSupport):
        model_types = ("causal_profile_test",)
        profile = ModelProfile(family=VANILLA_CAUSAL_LM)

    class MultimodalSupport(ModelSupport):
        model_types = ("multimodal_profile_test",)
        profile = ModelProfile(family=IMAGE_TEXT_TO_TEXT)

    causal = resolve_model_support(CausalSupport())
    multimodal = resolve_model_support(MultimodalSupport())

    assert causal.family == "vanilla_causal_lm"
    assert causal.is_multimodal is False
    assert causal.strategies.auto_model_cls is not None
    assert causal.strategies.auto_model_cls().__name__ == "AutoModelForCausalLM"
    assert multimodal.family == "image_text_to_text"
    assert multimodal.is_multimodal is True
    assert multimodal.strategies.auto_model_cls is not None
    assert (
        multimodal.strategies.auto_model_cls().__name__ == "AutoModelForImageTextToText"
    )


def test_profile_resolution_precedence_and_fieldwise_strategy_inheritance():
    class FamilyAutoModel:
        pass

    class FamilyProcessingStrategy:
        pass

    class ProfileAutoModel:
        pass

    class LegacyProcessingStrategy:
        pass

    family_cfg_matcher = lambda cfg: cfg.source == "family"  # noqa: E731
    family_processor_matcher = lambda processor: processor == "family"  # noqa: E731
    profile_cfg_matcher = lambda cfg: cfg.source == "profile"  # noqa: E731

    family = ModelFamilyTemplate(
        name="shared_family",
        is_multimodal=True,
        strategies=ModelStrategies(
            auto_model_cls=_provider(FamilyAutoModel),
            processing_strategy_cls=_provider(FamilyProcessingStrategy),
        ),
        matchers=ModelMatchers(
            cfg=family_cfg_matcher,
            processor=family_processor_matcher,
        ),
    )

    class LayeredSupport(ModelSupport):
        model_types = ("layered_profile_test",)
        profile = ModelProfile(
            family=family,
            is_multimodal=True,
            strategies=ModelStrategyOverrides(
                auto_model_cls=_provider(ProfileAutoModel)
            ),
            matchers=ModelMatchers(cfg=profile_cfg_matcher),
        )
        is_multimodal = False

        def get_processing_strategy_cls(self):
            return LegacyProcessingStrategy

    resolved = resolve_model_support(LayeredSupport())

    assert resolved.is_multimodal is False
    assert resolved.strategies.auto_model_cls is not None
    assert resolved.strategies.auto_model_cls() is ProfileAutoModel
    assert resolved.strategies.processing_strategy_cls is not None
    assert resolved.strategies.processing_strategy_cls() is LegacyProcessingStrategy
    assert resolved.matchers.cfg is profile_cfg_matcher
    assert resolved.matchers.processor is family_processor_matcher


def test_inherited_base_defaults_do_not_shadow_a_profile():
    class ProfileOnlySupport(ModelSupport):
        model_types = ("profile_only_test",)
        profile = ModelProfile(family=IMAGE_TEXT_TO_TEXT)

    support = ProfileOnlySupport()

    assert resolve_model_support(support).is_multimodal is True
    assert support.is_multimodal is True


def test_profile_projections_agree_for_descriptor_class_and_instance():
    family = ModelFamilyTemplate(
        name="projection_family",
        is_multimodal=True,
        capabilities={"projection": Supported("family")},
    )

    class ProjectedSupport(ModelSupport):
        model_types = ("projected_profile_test",)
        profile = ModelProfile(family=family)

    support = ProjectedSupport()

    assert ProjectedSupport.is_multimodal is support.is_multimodal is True
    assert ProjectedSupport.capabilities == support.capabilities
    assert isinstance(ProjectedSupport.capabilities["projection"], Supported)


def test_strategy_override_omission_inherits_and_explicit_none_clears():
    class FamilyAutoModel:
        pass

    class FamilyProcessingStrategy:
        pass

    family = ModelFamilyTemplate(
        name="strategy_clear_family",
        strategies=ModelStrategies(
            auto_model_cls=_provider(FamilyAutoModel),
            processing_strategy_cls=_provider(FamilyProcessingStrategy),
        ),
    )

    class InheritingSupport(ModelSupport):
        model_types = ("inheriting_strategy_test",)
        profile = ModelProfile(family=family)

    class ClearingSupport(ModelSupport):
        model_types = ("clearing_strategy_test",)
        profile = ModelProfile(
            family=family,
            strategies=ModelStrategyOverrides(auto_model_cls=None),
        )

    inherited = resolve_model_support(InheritingSupport()).strategies
    cleared = resolve_model_support(ClearingSupport()).strategies

    assert inherited.auto_model_cls is not None
    assert inherited.auto_model_cls() is FamilyAutoModel
    assert inherited.processing_strategy_cls is not None
    assert inherited.processing_strategy_cls() is FamilyProcessingStrategy
    assert cleared.auto_model_cls is None
    assert cleared.processing_strategy_cls is not None
    assert cleared.processing_strategy_cls() is FamilyProcessingStrategy


def test_unprofiled_legacy_multimodal_support_preserves_loader_fallback():
    class LegacyMultimodalSupport(ModelSupport):
        model_types = ("legacy_multimodal_fallback_test",)
        is_multimodal = True

    resolved = resolve_model_support(LegacyMultimodalSupport())

    assert resolved.family is None
    assert resolved.is_multimodal is True
    assert resolved.strategies.auto_model_cls is None
    assert resolved.strategies.processing_strategy_cls is None


def test_legacy_nullable_component_providers_remain_nullable():
    class NullableLegacySupport(ModelSupport):
        model_types = ("nullable_legacy_components_test",)

        def get_auto_model_cls(self):
            return None

        def get_processing_strategy_cls(self):
            return None

    resolved = resolve_model_support(NullableLegacySupport())

    assert resolved.strategies.auto_model_cls is not None
    assert resolved.strategies.auto_model_cls() is None
    assert resolved.strategies.processing_strategy_cls is not None
    assert resolved.strategies.processing_strategy_cls() is None


def test_legacy_component_and_matcher_super_calls_use_declarative_values_once():
    class ProfileAutoModel:
        pass

    class ProfileProcessingStrategy:
        pass

    family = ModelFamilyTemplate(
        name="super_call_family",
        strategies=ModelStrategies(
            auto_model_cls=_provider(ProfileAutoModel),
            processing_strategy_cls=_provider(ProfileProcessingStrategy),
        ),
        matchers=ModelMatchers(
            cfg=lambda cfg: cfg.marker == "profile",
            processor=lambda processor: processor == "profile",
        ),
    )

    class SuperCallingSupport(ModelSupport):
        model_types = ("super_call_profile_test",)
        profile = ModelProfile(family=family)

        def get_auto_model_cls(self):
            return super().get_auto_model_cls()

        def get_processing_strategy_cls(self):
            return super().get_processing_strategy_cls()

        def matches_cfg(self, cfg):
            return super().matches_cfg(cfg)

        def matches_processor(self, processor):
            return super().matches_processor(processor)

    resolved = resolve_model_support(SuperCallingSupport())

    assert resolved.strategies.auto_model_cls is not None
    assert resolved.strategies.auto_model_cls() is ProfileAutoModel
    assert resolved.strategies.processing_strategy_cls is not None
    assert resolved.strategies.processing_strategy_cls() is ProfileProcessingStrategy
    assert resolved.matchers.cfg is not None
    assert resolved.matchers.cfg(DictDefault(marker="profile")) is True
    assert resolved.matchers.processor is not None
    assert resolved.matchers.processor("profile") is True


def test_model_support_class_defaults_remain_neutral_values():
    assert type(ModelSupport.is_multimodal) is bool
    assert ModelSupport.is_multimodal is False
    assert isinstance(ModelSupport.capabilities, Mapping)
    assert not ModelSupport.capabilities


def test_capabilities_overlay_by_key_and_none_removes_a_family_default():
    family = ModelFamilyTemplate(
        name="capability_family",
        capabilities={
            "removed": Supported("family"),
            "replaced": Supported("family"),
            "preserved": Supported("family"),
        },
    )

    class CapabilitySupport(ModelSupport):
        model_types = ("capability_profile_test",)
        profile = ModelProfile(
            family=family,
            capabilities={
                "removed": None,
                "replaced": Experimental("profile"),
                "profile_only": Supported("profile"),
            },
        )
        capabilities = {
            "replaced": Unsupported("legacy"),
            "legacy_only": Supported("legacy"),
        }

    support = CapabilitySupport()
    capabilities = resolve_model_support(support).capabilities

    assert "removed" not in capabilities
    assert isinstance(capabilities["replaced"], Unsupported)
    assert isinstance(capabilities["preserved"], Supported)
    assert isinstance(capabilities["profile_only"], Supported)
    assert isinstance(capabilities["legacy_only"], Supported)
    with pytest.raises(ValueError, match="legacy"):
        check_capability(support, "replaced", "capability_profile_test")


def test_profile_value_objects_defensively_copy_and_freeze_mappings():
    family_capabilities = {"family": Supported()}
    profile_capabilities = {"profile": Supported()}
    hook_map = {ModelHookPhase.BEFORE_MODEL_BUILD: (lambda context: None,)}

    family = ModelFamilyTemplate(
        name="immutable_family",
        capabilities=family_capabilities,
        hooks=ModelHooks(hook_map),
    )
    model_profile = ModelProfile(
        family=family,
        capabilities=profile_capabilities,
    )

    class ImmutableSupport(ModelSupport):
        model_types = ("immutable_profile_test",)
        profile = model_profile

    family_capabilities["late"] = Supported()
    profile_capabilities["late"] = Supported()
    hook_map.clear()

    resolved = resolve_model_support(ImmutableSupport())
    assert "late" not in family.capabilities
    assert "late" not in model_profile.capabilities
    assert family.hooks.for_phase(ModelHookPhase.BEFORE_MODEL_BUILD)
    with pytest.raises(TypeError):
        cast(dict, resolved.capabilities)["late"] = Supported()
    with pytest.raises(TypeError):
        cast(dict, family.hooks.by_phase)[ModelHookPhase.BEFORE_MODEL_BUILD] = ()


def test_hooks_run_in_family_profile_legacy_order_on_every_dispatch():
    events = []

    def family_hook(context):
        events.append(("family", context))

    def profile_hook(context):
        events.append(("profile", context))

    family = ModelFamilyTemplate(
        name="hook_family",
        hooks=ModelHooks({ModelHookPhase.BEFORE_MODEL_BUILD: (family_hook,)}),
    )

    class HookedSupport(ModelSupport):
        model_types = ("hook_profile_test",)
        profile = ModelProfile(
            family=family,
            hooks=ModelHooks({ModelHookPhase.BEFORE_MODEL_BUILD: (profile_hook,)}),
        )

        def pre_model_load(self, cfg):
            events.append(("legacy", cfg))

    cfg = DictDefault(marker="cfg")
    context = ModelHookContext(cfg=cfg)
    support = HookedSupport()

    run_model_support_hooks(
        support,
        ModelHookPhase.BEFORE_MODEL_BUILD,
        context,
    )
    run_model_support_hooks(
        support,
        ModelHookPhase.BEFORE_MODEL_BUILD,
        context,
    )

    assert [name for name, _value in events] == [
        "family",
        "profile",
        "legacy",
        "family",
        "profile",
        "legacy",
    ]
    assert events[0][1] is context
    assert events[1][1] is context
    assert events[2][1] is cfg


def test_profile_hooks_can_replace_or_suppress_family_phases():
    events = []

    def family_before_hook(_context):
        events.append("family_before")

    def family_after_hook(_context):
        events.append("family_after")

    def profile_before_hook(_context):
        events.append("profile_before")

    family = ModelFamilyTemplate(
        name="replace_hook_family",
        hooks=ModelHooks(
            {
                ModelHookPhase.BEFORE_MODEL_BUILD: (family_before_hook,),
                ModelHookPhase.AFTER_BASE_MODEL_BUILD: (family_after_hook,),
            }
        ),
    )

    class ReplacingHookSupport(ModelSupport):
        model_types = ("replacing_hook_profile_test",)
        profile = ModelProfile(
            family=family,
            hooks=ModelHooks(
                {
                    ModelHookPhase.BEFORE_MODEL_BUILD: (profile_before_hook,),
                },
                replace_phases=frozenset(
                    {
                        ModelHookPhase.BEFORE_MODEL_BUILD,
                        ModelHookPhase.AFTER_BASE_MODEL_BUILD,
                    }
                ),
            ),
        )

    support = ReplacingHookSupport()
    context = ModelHookContext(cfg=DictDefault())

    run_model_support_hooks(
        support,
        ModelHookPhase.BEFORE_MODEL_BUILD,
        context,
    )
    run_model_support_hooks(
        support,
        ModelHookPhase.AFTER_BASE_MODEL_BUILD,
        context,
    )

    assert events == ["profile_before"]
    resolved = resolve_model_support(support)
    assert resolved.hooks.for_phase(ModelHookPhase.BEFORE_MODEL_BUILD) == (
        profile_before_hook,
    )
    assert resolved.hooks.for_phase(ModelHookPhase.AFTER_BASE_MODEL_BUILD) == ()


def test_profile_hook_is_callable_through_the_legacy_base_method():
    events = []

    def profile_hook(context):
        events.append(context.cfg)

    class ProfileHookSupport(ModelSupport):
        model_types = ("profile_hook_legacy_surface_test",)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            hooks=ModelHooks({ModelHookPhase.BEFORE_MODEL_BUILD: (profile_hook,)}),
        )

    cfg = DictDefault(marker="legacy_surface")
    ProfileHookSupport().pre_model_load(cfg)

    assert events == [cfg]


def test_legacy_hook_calling_super_does_not_duplicate_profile_hook():
    events = []

    def profile_hook(context):
        events.append(("profile", context.cfg))

    class ProfileAndLegacyHookSupport(ModelSupport):
        model_types = ("profile_legacy_super_hook_test",)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            hooks=ModelHooks({ModelHookPhase.BEFORE_MODEL_BUILD: (profile_hook,)}),
        )

        def pre_model_load(self, cfg):
            events.append(("legacy", cfg))
            super().pre_model_load(cfg)

    cfg = DictDefault(marker="canonical")
    run_model_support_hooks(
        ProfileAndLegacyHookSupport(),
        ModelHookPhase.BEFORE_MODEL_BUILD,
        ModelHookContext(cfg=cfg),
    )

    assert events == [("profile", cfg), ("legacy", cfg)]


def test_legacy_hook_redispatch_through_public_runner_does_not_recurse():
    events = []

    def profile_hook(context):
        events.append(("profile", context.cfg))

    class RedispatchingSupport(ModelSupport):
        model_types = ("legacy_public_redispatch_test",)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            hooks=ModelHooks({ModelHookPhase.BEFORE_MODEL_BUILD: (profile_hook,)}),
        )

        def pre_model_load(self, cfg):
            events.append(("legacy", cfg))
            run_model_support_hooks(
                self,
                ModelHookPhase.BEFORE_MODEL_BUILD,
                ModelHookContext(cfg=cfg),
            )

    cfg = DictDefault(marker="public_redispatch")
    run_model_support_hooks(
        RedispatchingSupport(),
        ModelHookPhase.BEFORE_MODEL_BUILD,
        ModelHookContext(cfg=cfg),
    )

    assert events == [("profile", cfg), ("legacy", cfg)]


def test_legacy_descriptor_methods_are_adapted_without_a_profile():
    events = []

    class LegacyAutoModel:
        pass

    class LegacySupport(ModelSupport):
        model_types = ("legacy_profile_test",)
        is_multimodal = True
        capabilities = {"legacy": Supported()}

        def get_auto_model_cls(self):
            return LegacyAutoModel

        def matches_cfg(self, cfg):
            return cfg.legacy

        def post_model_load(self, cfg, model):
            events.append((cfg, model))

    cfg = DictDefault(legacy=True)
    model = object()
    support = LegacySupport()
    resolved = resolve_model_support(support)

    assert resolved.family is None
    assert resolved.is_multimodal is True
    assert isinstance(resolved.capabilities["legacy"], Supported)
    assert resolved.strategies.auto_model_cls is not None
    assert resolved.strategies.auto_model_cls() is LegacyAutoModel
    assert resolved.matchers.cfg is not None
    assert resolved.matchers.cfg(cfg) is True

    run_model_support_hooks(
        support,
        ModelHookPhase.AFTER_ADAPTER_LOAD,
        ModelHookContext(cfg=cfg, model=model),
    )
    assert events == [(cfg, model)]


def test_nullable_resolution_and_hook_dispatch_preserve_legacy_fallback():
    assert resolve_model_support(None) is None
    run_model_support_hooks(
        None,
        ModelHookPhase.BEFORE_MODEL_BUILD,
        ModelHookContext(cfg=DictDefault()),
    )
