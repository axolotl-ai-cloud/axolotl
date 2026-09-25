"""Native tensor moves must coexist with DeepSpeed diagnostic weak references."""

import weakref

import pytest
import torch

from axolotl.monkeypatch.torchao_deepspeed_compat import (
    _suspend_native_debug_refs,
    install_native_nvfp4_debug_compat,
)


class NVFP4Tensor(torch.Tensor):
    pass


class DebugNames:
    def __init__(self):
        self._names = {}
        self._finalizers = {}

    def __setitem__(self, parameter, name):
        key = id(parameter)
        previous = self._finalizers.pop(key, None)
        if previous is not None:
            previous.detach()
        self._names[key] = name
        self._finalizers[key] = weakref.finalize(parameter, self._discard, key)

    def _discard(self, key):
        self._names.pop(key, None)
        self._finalizers.pop(key, None)

    def __getitem__(self, parameter):
        return self._names[id(parameter)]


@pytest.fixture
def native_model():
    model = torch.nn.Linear(2, 2, bias=False)
    model.weight = torch.nn.Parameter(
        torch.ones(2, 2).as_subclass(NVFP4Tensor), requires_grad=False
    )
    model.register_parameter("adapter", torch.nn.Parameter(torch.ones(2)))
    return model


@pytest.fixture
def swap_conversion():
    previous = torch.__future__.get_swap_module_params_on_conversion()
    torch.__future__.set_swap_module_params_on_conversion(True)
    yield
    torch.__future__.set_swap_module_params_on_conversion(previous)


def test_native_swap_suspends_only_deepspeed_debug_references(
    native_model, swap_conversion
):
    names = DebugNames()
    names[native_model.weight] = "weight"
    parameter = native_model.weight
    with pytest.raises(RuntimeError, match="Couldn.t swap"):
        native_model.to(dtype=torch.float64)

    with _suspend_native_debug_refs(native_model, names):
        assert not weakref.getweakrefs(parameter)
        assert names[parameter] == "weight"
        native_model.to(dtype=torch.float64)

    assert native_model.weight is parameter
    assert parameter.dtype == torch.float64
    assert names[parameter] == "weight"
    assert names._finalizers[id(parameter)].alive


def test_debug_references_restore_after_failure(native_model):
    names = DebugNames()
    names[native_model.weight] = "weight"
    names[native_model.adapter] = "adapter"
    adapter_finalizer = names._finalizers[id(native_model.adapter)]
    with pytest.raises(ValueError, match="conversion failed"):
        with _suspend_native_debug_refs(native_model, names):
            assert names._finalizers[id(native_model.adapter)] is adapter_finalizer
            raise ValueError("conversion failed")
    assert names._finalizers[id(native_model.weight)].alive
    assert names._finalizers[id(native_model.adapter)] is adapter_finalizer


def test_unrelated_weak_reference_is_not_removed(native_model, swap_conversion):
    names = DebugNames()
    names[native_model.weight] = "weight"
    external = weakref.ref(native_model.weight)
    with pytest.raises(RuntimeError, match="Couldn.t swap"):
        with _suspend_native_debug_refs(native_model, names):
            native_model.to(dtype=torch.float64)
    assert external() is native_model.weight
    assert names._finalizers[id(native_model.weight)].alive


def test_older_debug_dictionary_is_unchanged(native_model):
    names = {native_model.weight: "weight"}
    with _suspend_native_debug_refs(native_model, names):
        assert names[native_model.weight] == "weight"
    assert len(names) == 1


def test_installer_is_idempotent_and_delegates_ordinary_models():
    calls = []

    class Engine:
        def _configure_distributed_model(self, model, *, flag):
            calls.append((model, flag))
            return 42

    install_native_nvfp4_debug_compat(Engine)
    installed = Engine._configure_distributed_model
    install_native_nvfp4_debug_compat(Engine)
    assert Engine._configure_distributed_model is installed
    model = torch.nn.Linear(2, 2)
    assert Engine()._configure_distributed_model(model, flag=True) == 42
    assert calls == [(model, True)]
