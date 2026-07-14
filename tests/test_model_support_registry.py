"""Offline contract tests for model-support registration and matching."""

from dataclasses import dataclass
from typing import ClassVar

import pytest

from axolotl.model_support import ModelSupport, registry as support_registry
from axolotl.utils.dict import DictDefault


@pytest.fixture
def isolated_registry(monkeypatch):
    support_registry._ensure_builtins()
    monkeypatch.setattr(
        support_registry, "_REGISTRY", support_registry._REGISTRY.copy()
    )
    return support_registry


def test_plugin_registered_before_first_lookup_overrides_builtin(monkeypatch):
    monkeypatch.setattr(support_registry, "_REGISTRY", {})
    monkeypatch.setattr(support_registry, "_builtins_loaded", False)
    monkeypatch.setattr(support_registry, "_loading_builtins", False)
    monkeypatch.setattr(support_registry, "_BUILTIN_MODULES", ("fake_builtin",))

    def import_builtin(_module_name):
        class BuiltinSupport(ModelSupport):
            model_types = ("shared_arch",)

        support_registry.register_model_support(BuiltinSupport)

    monkeypatch.setattr(support_registry.importlib, "import_module", import_builtin)

    class PluginSupport(ModelSupport):
        model_types = ("shared_arch",)

    support_registry.register_model_support(PluginSupport)

    assert isinstance(support_registry.get_model_support("shared_arch"), PluginSupport)


def test_failed_builtin_import_retries_without_losing_partial_registrations(
    monkeypatch,
):
    monkeypatch.setattr(support_registry, "_REGISTRY", {})
    monkeypatch.setattr(support_registry, "_builtins_loaded", False)
    monkeypatch.setattr(support_registry, "_loading_builtins", False)
    monkeypatch.setattr(
        support_registry,
        "_BUILTIN_MODULES",
        ("first_builtin", "retrying_builtin"),
    )

    class FirstBuiltinSupport(ModelSupport):
        model_types = ("first_builtin_arch",)

    class RetryingBuiltinSupport(ModelSupport):
        model_types = ("retrying_builtin_arch",)

    attempts = 0

    def import_builtin(module_name):
        nonlocal attempts
        if module_name == "first_builtin":
            if "first_builtin_arch" not in support_registry._REGISTRY:
                support_registry.register_model_support(FirstBuiltinSupport)
            return
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient import failure")
        support_registry.register_model_support(RetryingBuiltinSupport)

    monkeypatch.setattr(support_registry.importlib, "import_module", import_builtin)

    with pytest.raises(RuntimeError, match="transient import failure"):
        support_registry.get_model_support("retrying_builtin_arch")

    assert support_registry._builtins_loaded is False
    assert isinstance(
        support_registry._REGISTRY["first_builtin_arch"],
        FirstBuiltinSupport,
    )

    assert isinstance(
        support_registry.get_model_support("retrying_builtin_arch"),
        RetryingBuiltinSupport,
    )
    assert support_registry._builtins_loaded is True
    assert isinstance(
        support_registry.get_model_support("first_builtin_arch"),
        FirstBuiltinSupport,
    )


@pytest.mark.parametrize(
    "model_types",
    [(), "not-a-tuple", ("",), ("valid", 1), ("duplicate", "duplicate")],
)
def test_registration_rejects_invalid_model_types(isolated_registry, model_types):
    invalid_support = type(
        "InvalidSupport",
        (ModelSupport,),
        {"model_types": model_types},
    )

    with pytest.raises(ValueError, match="model_types"):
        isolated_registry.register_model_support(invalid_support)


def test_unhashable_support_registered_under_aliases_matches_once(isolated_registry):
    @dataclass
    class UnhashableSupport(ModelSupport):
        model_types: ClassVar[tuple[str, ...]] = (
            "unhashable_arch",
            "unhashable_alias",
        )

        def matches_cfg(self, cfg):
            return cfg.base_model_config == "unhashable"

    isolated_registry.register_model_support(UnhashableSupport)

    support = isolated_registry.get_model_support_for_cfg(
        DictDefault(base_model_config="unhashable")
    )
    assert isinstance(support, UnhashableSupport)


def test_exact_model_type_cfg_lookup_precedes_broad_matchers(isolated_registry):
    class BroadSupport(ModelSupport):
        model_types = ("broad_arch",)

        def matches_cfg(self, cfg):
            return True

    class ExactSupport(ModelSupport):
        model_types = ("exact_arch",)

    isolated_registry.register_model_support(BroadSupport)
    isolated_registry.register_model_support(ExactSupport)

    support = isolated_registry.get_model_support_for_cfg(
        DictDefault(model_config_type="exact_arch")
    )
    assert isinstance(support, ExactSupport)


def test_ambiguous_cfg_match_raises_with_support_names(isolated_registry):
    class FirstSupport(ModelSupport):
        model_types = ("first_arch",)

        def matches_cfg(self, cfg):
            return True

    class SecondSupport(ModelSupport):
        model_types = ("second_arch",)

        def matches_cfg(self, cfg):
            return True

    isolated_registry.register_model_support(FirstSupport)
    isolated_registry.register_model_support(SecondSupport)

    with pytest.raises(ValueError, match=r"FirstSupport.*SecondSupport"):
        isolated_registry.get_model_support_for_cfg(DictDefault())


def test_ambiguous_processor_match_raises_with_support_names(isolated_registry):
    class FirstSupport(ModelSupport):
        model_types = ("first_processor_arch",)

        def matches_processor(self, processor):
            return True

    class SecondSupport(ModelSupport):
        model_types = ("second_processor_arch",)

        def matches_processor(self, processor):
            return True

    isolated_registry.register_model_support(FirstSupport)
    isolated_registry.register_model_support(SecondSupport)

    with pytest.raises(ValueError, match=r"FirstSupport.*SecondSupport"):
        isolated_registry.get_model_support_for_processor(object())
