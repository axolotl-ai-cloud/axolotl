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
