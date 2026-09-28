"""Reject ineffective CP options before creating process groups or patching models."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from axolotl.integrations.context_parallel.args import ContextParallelConfig
from axolotl.integrations.context_parallel.settings import resolve_settings


@pytest.mark.parametrize(
    "settings",
    [
        {"load_balance": "ptrr"},
        {"load_balance": "per_document"},
        {"ulysses_size": 0},
        {"ring_size": -1},
        {"ulysses_size": 3},
        {"backend": "ring", "ulysses_size": 2},
        {"backend": "ulysses", "ring_size": 2},
        {"load_balnce": "none"},
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(ValidationError):
        ContextParallelConfig(size=4, **settings)


def _resolve(u=1, r=4, **kwargs):
    cp = ContextParallelConfig(size=u * r, **kwargs.pop("settings", {}))
    resolved = SimpleNamespace(ulysses_size=u, ring_size=r)
    communication = resolve_settings(cp, resolved, num_kv_heads=8, **kwargs)
    return resolved.load_balance.value, communication


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, ("head_tail", "p2p")),
        ({"u": 4, "r": 1}, ("none", "all_to_all")),
        ({"u": 2, "r": 2}, ("none", "allgather")),
        ({"contiguous_reason": "recurrent state passing"}, ("none", "allgather")),
        ({"sliding_window": True}, ("none", "allgather")),
        ({"settings": {"rotate_method": "alltoall"}}, ("none", "p2p")),
        (
            {
                "settings": {"load_balance": "distflash"},
                "contiguous_reason": "recurrent state passing",
            },
            ("distflash", "p2p"),
        ),
        ({"glm_dsa": True}, ("none", "glm_dsa")),
    ],
)
def test_effective_settings(kwargs, expected):
    pytest.importorskip("ringmaster")
    assert _resolve(**kwargs) == expected


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"u": 4, "r": 1, "settings": {"load_balance": "head_tail"}}, "pure Ring"),
        ({"u": 2, "r": 2, "settings": {"load_balance": "distflash"}}, "pure Ring"),
        (
            {
                "contiguous_reason": "output gathering",
                "settings": {"load_balance": "head_tail"},
            },
            "output gathering",
        ),
        (
            {"settings": {"load_balance": "distflash", "rotate_method": "allgather"}},
            "owns its P2P",
        ),
        (
            {"sliding_window": True, "settings": {"load_balance": "head_tail"}},
            "sliding-window",
        ),
        ({"u": 4, "r": 1, "settings": {"ring_impl": "hf_kernels"}}, "only apply"),
        ({"glm_dsa": True, "settings": {"backend": "ring"}}, "GLM DSA owns"),
    ],
)
def test_ineffective_settings_fail(kwargs, match):
    pytest.importorskip("ringmaster")
    with pytest.raises(ValueError, match=match):
        _resolve(**kwargs)


@pytest.mark.parametrize("descriptor", [None, "missing", "supported", "unsupported"])
def test_model_capability_uses_generic_fallback(monkeypatch, descriptor):
    from axolotl import model_support
    from axolotl.integrations.context_parallel.settings import check_model_capability
    from axolotl.model_support.base import ModelSupport, Supported, Unsupported

    support = ModelSupport() if descriptor else None
    if descriptor in ("supported", "unsupported"):
        support.capabilities = {
            "context_parallel": Supported()
            if descriptor == "supported"
            else Unsupported("unsafe mixer")
        }
    monkeypatch.setattr(model_support, "get_model_support", lambda model_type: support)
    if descriptor == "unsupported":
        with pytest.raises(ValueError, match="unsafe mixer"):
            check_model_capability("future_model")
    else:
        check_model_capability("future_model")


def test_experimental_model_capability_warns(monkeypatch):
    from unittest.mock import Mock

    from axolotl import model_support
    from axolotl.integrations.context_parallel.settings import check_model_capability
    from axolotl.model_support import base

    support = base.ModelSupport()
    support.capabilities = {"context_parallel": base.Experimental("verify this mixer")}
    monkeypatch.setattr(model_support, "get_model_support", lambda model_type: support)
    warning = Mock()
    monkeypatch.setattr(base.LOG, "warning_once", warning)
    check_model_capability("future_model")
    warning.assert_called_once()
    assert "verify this mixer" in str(warning.call_args)
