"""Loading, staged or not, must not require torchao, which pyproject excludes on macOS and aarch64."""

import importlib
import sys

import pytest


@pytest.fixture(name="without_torchao")
def _without_torchao(monkeypatch):
    for name in list(sys.modules):
        if name == "torchao" or name.startswith("torchao."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "torchao", None)
    for name in ("axolotl.monkeypatch.quantized_init", "axolotl.loaders.nf4"):
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_quantized_init_guard_imports_without_torchao(without_torchao):
    module = importlib.import_module("axolotl.monkeypatch.quantized_init")
    module.patch_transformers_skip_quantized_init()


def test_model_loader_guard_runs_without_torchao(without_torchao):
    from axolotl.loaders.model import ModelLoader

    ModelLoader._patch_quantized_init()


def test_staged_loader_imports_without_torchao(without_torchao):
    importlib.import_module("axolotl.loaders.nf4")
