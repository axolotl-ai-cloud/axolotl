"""Loading, staged or not, must not require torchao, which pyproject excludes on macOS and aarch64."""

import gc
import importlib
import sys
import weakref

import pytest
import torch

from axolotl.utils.dict import DictDefault


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


def test_zero3_model_build_keeps_config_alive_only_for_zero3(monkeypatch):
    from axolotl.loaders import model as model_module
    from axolotl.loaders.model import ModelLoader

    created = []
    expect_config = True

    class DeepSpeedConfig:
        def __init__(self, _config):
            created.append(weakref.ref(self))

        def fill_match(self, *_args):
            pass

    class Loader:
        @staticmethod
        def from_pretrained(_base_model, **_kwargs):
            gc.collect()
            if expect_config:
                assert created[-1]() is not None
            else:
                assert created[-1]() is None
            return torch.nn.Linear(2, 2)

    def build_loader():
        loader = object.__new__(ModelLoader)
        loader.cfg = DictDefault(
            base_model="stub",
            load_in_4bit=False,
            tensor_parallel_size=1,
            context_parallel_size=1,
            reinit_weights=False,
            torch_dtype=torch.bfloat16,
            deepspeed={},
            micro_batch_size=1,
            gradient_accumulation_steps=1,
        )
        loader.model_kwargs = {"torch_dtype": torch.bfloat16}
        loader.auto_model_loader = Loader
        loader.base_model = "stub"
        loader.model_config = object()
        loader.model_type = None
        return loader

    monkeypatch.setattr(
        model_module.transformers.modeling_utils,
        "is_deepspeed_zero3_enabled",
        model_module.transformers.modeling_utils.is_deepspeed_zero3_enabled,
    )
    monkeypatch.setattr(
        model_module.transformers.integrations.deepspeed,
        "is_deepspeed_zero3_enabled",
        model_module.transformers.integrations.deepspeed.is_deepspeed_zero3_enabled,
    )
    monkeypatch.setattr(model_module, "HfTrainerDeepSpeedConfig", DeepSpeedConfig)
    monkeypatch.setenv("ACCELERATE_DEEPSPEED_ZERO_STAGE", "3")
    zero3_loader = build_loader()
    assert not zero3_loader._build_model()
    assert zero3_loader._native_nvfp4_zero3_loading
    gc.collect()
    assert created[-1]() is None

    monkeypatch.delenv("ACCELERATE_DEEPSPEED_ZERO_STAGE")
    expect_config = False
    ordinary_loader = build_loader()
    assert not ordinary_loader._build_model()
    assert not ordinary_loader._native_nvfp4_zero3_loading
