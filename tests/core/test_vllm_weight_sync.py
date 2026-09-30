"""Tests for pushing trainer weights to a `vllm serve` server."""

import copy
from unittest.mock import MagicMock

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from axolotl.core.trainers.grpo.vllm_weight_sync import (
    FP8_DTYPE,
    init_communicator_lazily,
    load_lora_adapter,
    peft_weights_for_vllm,
    quantize_fp8,
)


class Attention(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)


class TinyModel(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.embed = nn.Embedding(8, dim)
        self.attn = Attention(dim)
        self.lm_head = nn.Linear(dim, 8, bias=False)


def fix_name(name, extra_prefixes=None):
    for prefix in ["_checkpoint_wrapped_module."] + (extra_prefixes or []):
        name = name.replace(prefix, "")
    return name


def _peft_model(**lora_kwargs):
    torch.manual_seed(0)
    config = LoraConfig(
        r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], **lora_kwargs
    )
    model = get_peft_model(TinyModel(), config)
    for module in model.modules():
        if hasattr(module, "lora_B"):
            nn.init.normal_(module.lora_B["default"].weight)
    return model


def _stream(model):
    metadata, named_params = peft_weights_for_vllm(model, fix_name)
    return metadata, list(named_params)


def test_streams_every_base_weight_with_lora_merged():
    model = _peft_model()
    expected = copy.deepcopy(model).merge_and_unload().state_dict()
    base_before = model.base_model.model.attn.q_proj.base_layer.weight.clone()

    metadata, streamed = _stream(model)

    # k_proj and the biases are sent too: vLLM rebuilds partially-loaded fused layers
    # from uninitialized memory.
    assert [name for name, _ in streamed] == list(expected)
    for name, tensor in streamed:
        torch.testing.assert_close(tensor, expected[name])
    torch.testing.assert_close(
        model.base_model.model.attn.q_proj.base_layer.weight, base_before
    )


def test_metadata_matches_stream():
    model = _peft_model()
    metadata, streamed = _stream(model)
    assert metadata == [
        (name, str(t.dtype).removeprefix("torch."), list(t.shape))
        for name, t in streamed
    ]


def test_modules_to_save_use_vllm_name():
    model = _peft_model(modules_to_save=["lm_head"])
    _, streamed = _stream(model)
    names = [name for name, _ in streamed]
    assert names.count("lm_head.weight") == 1
    assert not any("modules_to_save" in n or "original_module" in n for n in names)


def test_dora_is_rejected():
    model = _peft_model(use_dora=True)
    with pytest.raises(NotImplementedError, match="vllm_lora_sync"):
        peft_weights_for_vllm(model, fix_name)


@pytest.fixture
def dequantize_fp8():
    pytest.importorskip("bitsandbytes")
    from axolotl.kernels.quantize import dequantize_fp8

    return dequantize_fp8


@pytest.mark.parametrize(
    "shape, scale_shape",
    [((256, 128), (2, 1)), ((200, 130), (2, 2)), ((64, 64), (1,))],
)
def test_fp8_roundtrip(dequantize_fp8, shape, scale_shape):
    torch.manual_seed(0)
    weight = torch.randn(shape)
    quantized, scale_inv = quantize_fp8(weight, torch.ones(scale_shape))
    assert quantized.dtype == FP8_DTYPE
    assert quantized.shape == weight.shape
    assert scale_inv.shape == scale_shape
    restored = dequantize_fp8(quantized, scale_inv, torch.float32)
    torch.testing.assert_close(restored, weight, atol=0.15, rtol=0.1)


def test_fp8_lora_weight_is_requantized_with_fresh_scales(dequantize_fp8):
    model = _peft_model()
    q_base = model.base_model.model.attn.q_proj.base_layer
    fp8_weight, scale_inv = quantize_fp8(q_base.weight.data, torch.ones(2, 2))
    q_base.weight = nn.Parameter(fp8_weight, requires_grad=False)
    q_base.register_parameter(
        "weight_scale_inv", nn.Parameter(scale_inv, requires_grad=False)
    )
    q_lora = model.base_model.model.attn.q_proj
    delta = (
        q_lora.lora_B["default"].weight @ q_lora.lora_A["default"].weight
    ) * q_lora.scaling["default"]
    expected = dequantize_fp8(fp8_weight, scale_inv, torch.float32) + delta

    metadata, streamed = _stream(model)
    names = [name for name, _ in streamed]

    assert names.count("attn.q_proj.weight_scale_inv") == 1
    idx = names.index("attn.q_proj.weight")
    assert names[idx + 1] == "attn.q_proj.weight_scale_inv"
    weight, new_scale = streamed[idx][1], streamed[idx + 1][1]
    assert weight.dtype == FP8_DTYPE
    torch.testing.assert_close(
        dequantize_fp8(weight, new_scale, torch.float32),
        expected,
        atol=0.15,
        rtol=0.1,
    )
    assert metadata[idx] == ("attn.q_proj.weight", "float8_e4m3fn", [16, 16])


class FakeClient:
    def __init__(self):
        self.communicator = None
        self.init_calls = []
        self.pushed = []

    def init_communicator(self, device=0):
        self.init_calls.append(device)
        self.communicator = object()

    def update_named_params(self, metadata, named_params):
        assert self.communicator is not None
        self.pushed.append(list(named_params))

    def update_named_param(self, name, weights):
        self.update_named_params([], iter([(name, weights)]))


def test_communicator_opens_once_on_first_push():
    client = FakeClient()
    init_communicator_lazily(client, device="cuda:0")
    assert client.init_calls == []

    client.update_named_param("w", torch.zeros(1))
    client.update_named_params([], iter([]))

    assert client.init_calls == ["cuda:0"]
    assert len(client.pushed) == 2


@pytest.mark.parametrize("status, loaded", [(200, True), (404, False)])
def test_load_lora_adapter_uses_served_model_name(status, loaded):
    client = MagicMock()
    client.base_url = "http://vllm:8000"
    client.model = "org/base-model"
    client.session.post.return_value = MagicMock(status_code=status, text="")

    assert load_lora_adapter(client, "/tmp/adapter/v3", timeout=5) is loaded

    url = client.session.post.call_args.args[0]
    body = client.session.post.call_args.kwargs["json"]
    assert url == "http://vllm:8000/v1/load_lora_adapter"
    assert body == {
        "lora_name": "org/base-model",
        "lora_path": "/tmp/adapter/v3",
        "load_inplace": True,
    }
