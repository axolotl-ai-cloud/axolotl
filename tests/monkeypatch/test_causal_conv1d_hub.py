"""transformers' causal convs fall back to the causal-conv1d hub kernel."""

import sys
import types

import pytest

from axolotl.monkeypatch.models import causal_conv1d_hub


@pytest.fixture
def fake_model(monkeypatch):
    module = types.ModuleType("transformers.models.fake_hybrid.modeling_fake_hybrid")

    def reference(*args, **kwargs):
        return "reference"

    module.causal_conv1d_fn = reference
    module.causal_conv1d_update = reference
    monkeypatch.setitem(sys.modules, module.__name__, module)

    calls = []

    def causal_conv1d_fn(x, weight, bias=None, seq_idx=None, activation=None):
        calls.append({"seq_idx": seq_idx, "activation": activation})
        return "hub"

    kernel = types.SimpleNamespace(
        causal_conv1d_fn=causal_conv1d_fn, causal_conv1d_update=causal_conv1d_fn
    )
    import kernels

    monkeypatch.setattr(kernels, "get_kernel", lambda repo, version: kernel)
    causal_conv1d_hub.hub_causal_conv1d.cache_clear()
    yield module, calls
    causal_conv1d_hub.hub_causal_conv1d.cache_clear()


def _package_installed(monkeypatch, installed):
    find_spec = causal_conv1d_hub.importlib.util.find_spec
    monkeypatch.setattr(
        causal_conv1d_hub.importlib.util,
        "find_spec",
        lambda name: (
            (object() if installed else None)
            if name == "causal_conv1d"
            else find_spec(name)
        ),
    )


def test_binds_the_hub_kernel_without_the_package(fake_model, monkeypatch):
    module, calls = fake_model
    _package_installed(monkeypatch, False)
    assert causal_conv1d_hub.patch_causal_conv1d("fake_hybrid")
    # transformers passes the forward's kwargs along; the kernel takes its own.
    out = module.causal_conv1d_fn(
        "x", "w", None, activation="silu", seq_idx="idx", cu_seq_lens_q="unused"
    )
    assert out == "hub" and calls == [{"seq_idx": "idx", "activation": "silu"}]


def test_keeps_the_installed_package(fake_model, monkeypatch):
    module, _ = fake_model
    _package_installed(monkeypatch, True)
    assert not causal_conv1d_hub.patch_causal_conv1d("fake_hybrid")
    assert module.causal_conv1d_fn() == "reference"


def test_skips_models_without_a_causal_conv(fake_model, monkeypatch):
    _package_installed(monkeypatch, False)
    assert not causal_conv1d_hub.patch_causal_conv1d("llama")
    assert not causal_conv1d_hub.patch_causal_conv1d("no_such_model")
