"""Pin the accelerate behaviour ``ple_cpu_offload`` relies on.

``ModelLoader._keep_no_placement_params_on_cpu`` does not place anything itself. It sets a
single-entry ``hf_device_map`` so that ``Accelerator.prepare_model`` takes its quantized
branch and never reaches ``model.to(self.device)``. Nothing else keeps the 95.4 GiB n-gram
table off the accelerator, so if an accelerate release reorders that logic the table is
pulled into VRAM and the run OOMs instead of failing loudly.

``Accelerator.device`` is patched to ``meta`` so the move is observable without a GPU.
"""

from unittest.mock import patch

import torch
from accelerate import Accelerator

from axolotl.loaders.model import ModelLoader


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offloaded = torch.nn.Parameter(torch.zeros(4))


def _prepare_on_meta(model):
    with patch.object(Accelerator, "device", torch.device("meta")):
        return Accelerator().prepare_model(model, evaluation_mode=False)


def test_quantized_plus_device_map_skips_placement():
    model = _TinyModel()
    model.is_loaded_in_4bit = True
    model.hf_device_map = {"": 0}

    assert _prepare_on_meta(model).offloaded.device.type == "cpu"


def test_placement_happens_without_the_quantized_marker():
    """The negative control: without it the skip is not what kept the tensor on CPU."""
    model = _TinyModel()
    model.hf_device_map = {"": 0}

    assert _prepare_on_meta(model).offloaded.device.type == "meta"


def test_loader_sets_the_device_map_the_skip_needs(tmp_path):
    model = _TinyModel()
    model._no_placement_params = ["offloaded"]

    loader = ModelLoader.__new__(ModelLoader)
    loader.model = model
    loader.cfg = type("Cfg", (), {"model_config_type": "qwen4_exp"})()
    loader._keep_no_placement_params_on_cpu()

    assert model.hf_device_map == {"": 0}


class _TinyTwoParamModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = torch.nn.Linear(2, 2)
        self.table = torch.nn.Embedding(4, 2)
        self.register_buffer("scale", torch.ones(1))
        self._no_placement_params = ["table.weight"]


def test_to_device_except_leaves_kept_params_in_place():
    from axolotl.loaders.utils import (
        host_resident_no_placement_params,
        to_device_except,
    )

    model = _TinyTwoParamModel()
    table = model.table.weight
    order = [name for name, _ in model.named_parameters()]

    to_device_except(model, "meta", host_resident_no_placement_params(model).values())

    assert model.table.weight is table
    assert table.device.type == "cpu"
    assert model.inner.weight.device.type == "meta"
    assert model.scale.device.type == "meta"
    assert [name for name, _ in model.named_parameters()] == order


def test_trainer_skips_placement_for_unquantized_host_resident_model():
    """A CPU-staged NF4 model has no ``is_loaded_in_4bit``, so only the trainer's skip
    keeps accelerate from moving the table."""
    from axolotl.core.trainers.base import AxolotlTrainer

    with patch.object(Accelerator, "device", torch.device("meta")):
        trainer = AxolotlTrainer.__new__(AxolotlTrainer)
        trainer.accelerator = Accelerator()
        trainer._skip_placement_of_host_resident_models()

        offloaded = _TinyTwoParamModel()
        offloaded.hf_device_map = {"": 0}
        trainer.accelerator.prepare_model(offloaded, evaluation_mode=False)
        assert offloaded.table.weight.device.type == "cpu"

        plain = _TinyTwoParamModel()
        plain._no_placement_params = []
        plain = trainer.accelerator.prepare_model(plain, evaluation_mode=False)
        assert plain.table.weight.device.type == "meta"


def test_unquantized_host_resident_model_moves_without_the_trainer_skip():
    model = _TinyTwoParamModel()
    model.hf_device_map = {"": 0}

    assert _prepare_on_meta(model).table.weight.device.type == "meta"
