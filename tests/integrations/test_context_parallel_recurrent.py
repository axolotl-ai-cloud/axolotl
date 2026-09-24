"""Native recurrent CP preflight and instance-local forward bindings."""

import builtins
from functools import wraps
from types import SimpleNamespace

import pytest
from ringmaster import recurrent


def test_missing_fla_fails_before_cp_setup(monkeypatch):
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name.startswith("fla."):
            raise ImportError("missing native kernels")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    mixer = SimpleNamespace(chunk_gated_delta_rule=lambda: None)
    model = SimpleNamespace(modules=lambda: [mixer])
    with pytest.raises(ValueError, match="importable FLA CP kernels"):
        recurrent.validate_recurrent([model], 4)
    assert not hasattr(mixer, "_cp_instance_wrapped")


def test_unknown_gated_delta_forward_fails_before_mutation(monkeypatch):
    class UnknownGatedDelta:
        def forward(self, x):
            return x

    mixer = UnknownGatedDelta()
    monkeypatch.setattr(recurrent, "require_fla_cp", lambda: None)
    with pytest.raises(ValueError, match="Unsupported gated-delta forward"):
        recurrent.validate_recurrent([SimpleNamespace(modules=lambda: [mixer])], 4)
    assert vars(mixer) == {}


def _kernel(value):
    return value + 1


def test_rebinding_preserves_decorators_and_original_globals():
    events = []

    def decorator(function):
        @wraps(function)
        def wrapper(value):
            events.append(value)
            return function(value)

        return wrapper

    @decorator
    def forward(value):
        return _kernel(value)

    cloned = recurrent._rebind_globals(forward, {"_kernel": lambda value: value * 2})
    assert cloned(3) == 6
    assert forward(3) == 4
    assert events == [3, 3]
