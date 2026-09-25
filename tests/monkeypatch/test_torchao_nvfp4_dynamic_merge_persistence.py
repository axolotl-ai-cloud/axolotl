"""CPU coverage for dynamic native NVFP4 merge-aware export metadata."""

import json

import pytest
import torch
from torch import nn

pytest.importorskip("torchao")
pytest.importorskip("peft")

from peft import LoraConfig
from peft.tuners.lora.layer import Linear
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    QuantizeTensorToNVFP4Kwargs,
    per_tensor_amax_to_scale,
)

from axolotl.monkeypatch.torchao_nvfp4_merge import (
    capture_native_nvfp4_recipe,
    install_native_nvfp4_merge_aware_lora_linears,
)
from axolotl.monkeypatch.torchao_nvfp4_merge_persistence import (
    capture_static_native_metadata,
    write_native_metadata,
)


def _dynamic_model():
    source = torch.randn(32, 32, dtype=torch.bfloat16)
    base = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
    base.weight = nn.Parameter(
        NVFP4Tensor.to_nvfp4(
            source,
            per_tensor_scale=per_tensor_amax_to_scale(source.abs().max()),
            act_per_tensor_scale=per_tensor_amax_to_scale(torch.tensor(4.0)),
            is_swizzled_scales=True,
            act_quant_kwargs=QuantizeTensorToNVFP4Kwargs(
                use_dynamic_per_tensor_scale=False,
                is_swizzled_scales=True,
            ),
        ),
        requires_grad=False,
    )
    layer = Linear(
        base,
        "default",
        LoraConfig(r=2, lora_alpha=4),
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
    )
    return nn.ModuleDict({"q_proj": layer})


def test_dynamic_native_metadata_capture_and_adapter_export(tmp_path):
    model = _dynamic_model()
    assert install_native_nvfp4_merge_aware_lora_linears(model) == 1

    metadata = capture_static_native_metadata(model, start_step=2)
    weight = model["q_proj"].get_base_layer().weight
    assert metadata["backend"] == "native_torchao"
    assert metadata["targets"] == {
        "q_proj.weight": capture_native_nvfp4_recipe(weight).fingerprint()
    }

    config = tmp_path / "adapter_config.json"
    config.write_text(json.dumps({"r": 2}))
    assert write_native_metadata(tmp_path, metadata)
    assert json.loads(config.read_text())["nvfp4_merge_aware"] == metadata
