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


def test_entry_point_model_support_overrides_builtin(monkeypatch):
    monkeypatch.setattr(support_registry, "_REGISTRY", {})
    monkeypatch.setattr(support_registry, "_builtins_loaded", False)
    monkeypatch.setattr(support_registry, "_loading_builtins", False)
    monkeypatch.setattr(support_registry, "_BUILTIN_MODULES", ("fake_builtin",))

    class BuiltinSupport(ModelSupport):
        model_types = ("shared_arch",)

    class ExternalSupport(ModelSupport):
        model_types = ("shared_arch",)

    def import_builtin(_module_name):
        support_registry.register_model_support(BuiltinSupport)

    class EntryPoint:
        def load(self):
            return ExternalSupport

    monkeypatch.setattr(support_registry.importlib, "import_module", import_builtin)
    monkeypatch.setattr(
        support_registry,
        "entry_points",
        lambda **kwargs: [EntryPoint()],
    )

    assert isinstance(
        support_registry.get_model_support("shared_arch"), ExternalSupport
    )


def test_entry_point_override_survives_reentrant_builtin_registration(monkeypatch):
    monkeypatch.setattr(support_registry, "_REGISTRY", {})
    monkeypatch.setattr(support_registry, "_builtins_loaded", False)
    monkeypatch.setattr(support_registry, "_loading_builtins", False)
    monkeypatch.setattr(support_registry, "_BUILTIN_MODULES", ("fake_builtin",))

    BuiltinSupport = type(
        "BuiltinSupport",
        (ModelSupport,),
        {"__module__": "fake_builtin", "model_types": ("shared_arch",)},
    )

    class ExternalSupport(ModelSupport):
        model_types = ("shared_arch",)

    class EntryPoint:
        def load(self):
            return ExternalSupport

    monkeypatch.setattr(support_registry.importlib, "import_module", lambda _: None)
    monkeypatch.setattr(
        support_registry,
        "entry_points",
        lambda **kwargs: [EntryPoint()],
    )

    support_registry.register_model_support(BuiltinSupport)

    assert isinstance(
        support_registry.get_model_support("shared_arch"), ExternalSupport
    )


def test_decorated_entry_point_model_support_is_not_registered_twice(monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr(support_registry, "_REGISTRY", {})
    monkeypatch.setattr(support_registry, "_builtins_loaded", False)
    monkeypatch.setattr(support_registry, "_loading_builtins", False)
    monkeypatch.setattr(support_registry, "_BUILTIN_MODULES", ())

    class ExternalSupport(ModelSupport):
        model_types = ("external_arch",)

    class EntryPoint:
        def load(self):
            support_registry.register_model_support(ExternalSupport)
            return ExternalSupport

    monkeypatch.setattr(
        support_registry,
        "entry_points",
        lambda **kwargs: [EntryPoint()],
    )
    warning = Mock()
    monkeypatch.setattr(support_registry.LOG, "warning", warning)

    assert isinstance(
        support_registry.get_model_support("external_arch"), ExternalSupport
    )
    warning.assert_not_called()


def test_entry_point_model_support_must_be_a_descriptor(monkeypatch):
    monkeypatch.setattr(support_registry, "_REGISTRY", {})
    monkeypatch.setattr(support_registry, "_builtins_loaded", False)
    monkeypatch.setattr(support_registry, "_loading_builtins", False)
    monkeypatch.setattr(support_registry, "_BUILTIN_MODULES", ())
    monkeypatch.setattr(
        support_registry,
        "entry_points",
        lambda **kwargs: [type("EntryPoint", (), {"load": lambda self: object})()],
    )

    with pytest.raises(TypeError, match="ModelSupport subclass"):
        support_registry.get_model_support("external_arch")


def test_broken_model_support_entry_point_is_logged_and_skipped(monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr(support_registry, "_REGISTRY", {})
    monkeypatch.setattr(support_registry, "_builtins_loaded", False)
    monkeypatch.setattr(support_registry, "_loading_builtins", False)
    monkeypatch.setattr(support_registry, "_BUILTIN_MODULES", ())

    class EntryPoint:
        value = "missing_model_support:Support"

        def load(self):
            raise ModuleNotFoundError("missing_model_support")

    monkeypatch.setattr(
        support_registry,
        "entry_points",
        lambda **kwargs: [EntryPoint()],
    )
    warning = Mock()
    monkeypatch.setattr(support_registry.LOG, "warning", warning)

    assert support_registry.get_model_support("external_arch") is None
    assert support_registry._builtins_loaded is True
    warning.assert_called_once()
    assert "Could not import model-support entry point" in warning.call_args.args[0]
    assert warning.call_args.kwargs["exc_info"] is True


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


def test_registration_rejects_non_model_support_class(isolated_registry):
    class NotADescriptor:
        model_types = ("duck_arch",)
        profile = None

    with pytest.raises(TypeError, match="ModelSupport subclass"):
        isolated_registry.register_model_support(NotADescriptor)

    assert isolated_registry.get_model_support("duck_arch") is None


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


def test_registration_normalizes_list_model_types_to_tuple(isolated_registry):
    class ListDeclaredSupport(ModelSupport):
        model_types = ["list_declared_arch"]  # type: ignore[assignment]

    isolated_registry.register_model_support(ListDeclaredSupport)

    assert ListDeclaredSupport.model_types == ("list_declared_arch",)
    assert isinstance(
        isolated_registry.get_model_support("list_declared_arch"),
        ListDeclaredSupport,
    )


def test_builtin_import_does_not_hold_registry_lock(monkeypatch):
    """A registry lookup during a concurrent builtin import must not deadlock."""
    import threading

    monkeypatch.setattr(support_registry, "_REGISTRY", {})
    monkeypatch.setattr(support_registry, "_builtins_loaded", False)
    monkeypatch.setattr(support_registry, "_loading_builtins", False)
    monkeypatch.setattr(support_registry, "_BUILTIN_MODULES", ("slow_builtin",))

    lookup_finished = threading.Event()

    def concurrent_lookup():
        support_registry.get_model_support("unrelated_arch")
        lookup_finished.set()

    def import_builtin(_module_name):
        lookup_thread = threading.Thread(target=concurrent_lookup)
        lookup_thread.start()
        finished = lookup_finished.wait(timeout=5)
        lookup_thread.join(timeout=5)
        assert finished, "registry lookup deadlocked during builtin import"

        class SlowBuiltinSupport(ModelSupport):
            model_types = ("slow_builtin_arch",)

        support_registry.register_model_support(SlowBuiltinSupport)

    monkeypatch.setattr(support_registry.importlib, "import_module", import_builtin)

    assert support_registry.get_model_support("slow_builtin_arch") is not None
