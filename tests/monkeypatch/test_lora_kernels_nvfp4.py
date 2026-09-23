"""Native NVFP4 base dispatch preserves the projection's quantized forward."""

import copy

import pytest
import torch
from test_lora_fp32_gradients import projection
from torch import nn


@pytest.mark.parametrize("kind", ["linear", "qkv", "swiglu", "geglu"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fp32", [False, True])
def test_native_nvfp4_projection_forward(kind, device, monkeypatch, fp32):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    nvfp4 = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    from axolotl.kernels import lora as kernels

    torch.manual_seed(19)
    model = nn.Module()
    names = {
        "linear": ["o_proj"],
        "qkv": ["q_proj", "k_proj", "v_proj"],
        "swiglu": ["gate_proj", "up_proj", "down_proj"],
        "geglu": ["gate_proj", "up_proj", "down_proj"],
    }[kind]
    for name in names:
        layer = projection(device)
        layer.base_layer.weight = nn.Parameter(
            nvfp4.NVFP4Tensor.to_nvfp4(layer.base_layer.weight.detach()),
            requires_grad=False,
        )
        setattr(model, name, layer)
    model.act_fn = nn.GELU() if kind == "geglu" else nn.SiLU()
    if kind == "geglu" and device == "cpu":
        del model.act_fn
        model.act_fn = lambda value: torch.nn.functional.gelu(value.float()).to(
            value.dtype
        )
    if fp32:
        from axolotl.utils.lora_precision import upcast_lora_parameters

        upcast_lora_parameters(model)
    reference = copy.deepcopy(model)
    calls = []
    for name in names:
        layer = getattr(model, name).base_layer
        original = layer.forward

        def forward(x, original=original, name=name):
            calls.append(name)
            return original(x)

        monkeypatch.setattr(layer, "forward", forward)
    x = torch.randn(2, 7, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    if kind == "linear":
        actual = (kernels.apply_lora_o(model, x),)
        expected = (reference.o_proj(ref_x),)
    elif kind == "qkv":
        actual = kernels.apply_lora_qkv(model, x)
        expected = tuple(getattr(reference, name)(ref_x) for name in names)
    else:
        actual = (getattr(kernels, "apply_lora_mlp_" + kind)(model, x),)
        expected = (
            reference.down_proj(
                reference.act_fn(reference.gate_proj(ref_x)) * reference.up_proj(ref_x)
            ),
        )
    assert calls == names
    assert "LoRA_DeltaBackward" in type(actual[0].grad_fn).__name__
    sum(t.float().square().mean() for t in actual).backward()
    sum(t.float().square().mean() for t in expected).backward()
    for result, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, target, rtol=0.04, atol=0.002)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=0.04, atol=0.002)
    for (_, p), (_, q) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if p.requires_grad:
            assert p.grad.dtype == (torch.float32 if fp32 else torch.bfloat16)
            relative = (
                p.grad.float() - q.grad.float()
            ).norm() / q.grad.float().norm().clamp_min(1e-9)
            assert relative < 0.04


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fp32", [False, True])
def test_native_nvfp4_checkpoint_and_retained_backward(fp32):
    from torch.utils.checkpoint import checkpoint
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    from axolotl.kernels.lora import apply_lora_linear
    from axolotl.utils.lora_precision import upcast_lora_parameters

    torch.manual_seed(95)
    model = projection("cuda")
    model.base_layer.weight = nn.Parameter(
        NVFP4Tensor.to_nvfp4(model.base_layer.weight.detach()), requires_grad=False
    )
    if fp32:
        upcast_lora_parameters(model)
    reference = copy.deepcopy(model)
    x = torch.randn(2, 7, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    actual = checkpoint(
        lambda value: apply_lora_linear(model, value), x, use_reentrant=False
    )
    expected = reference(rx)
    for _ in range(2):
        actual.float().square().mean().backward(retain_graph=True)
        expected.float().square().mean().backward(retain_graph=True)
    assert (x.grad.float() - rx.grad.float()).norm() / rx.grad.float().norm() < 0.03
    for (_, p), (_, q) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if p.requires_grad:
            assert (
                p.grad.float() - q.grad.float()
            ).norm() / q.grad.float().norm() < 0.03


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_nvfp4_nemotron_attention_installer(monkeypatch):
    from peft import LoraConfig, get_peft_model
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from transformers.models.nemotron_h.modeling_nemotron_h import (
        NemotronHAttention,
        NemotronHForCausalLM,
    )

    from axolotl.kernels import lora as kernels
    from axolotl.monkeypatch import lora_kernels
    from axolotl.utils.dict import DictDefault
    from axolotl.utils.lora_precision import upcast_lora_parameters

    torch.manual_seed(99)
    config = NemotronHConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        layers_block_type=["full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        mtp_layers_block_type=[],
    )
    config._attn_implementation = "eager"
    model = (
        get_peft_model(
            NemotronHForCausalLM(config),
            LoraConfig(
                task_type="CAUSAL_LM",
                r=4,
                lora_alpha=8,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
        .cuda()
        .bfloat16()
    )
    for module in model.modules():
        if hasattr(module, "lora_A"):
            module.base_layer.weight = nn.Parameter(
                NVFP4Tensor.to_nvfp4(module.base_layer.weight.detach()),
                requires_grad=False,
            )
            nn.init.normal_(module.lora_B["default"].weight, std=0.02)
    upcast_lora_parameters(model)
    reference = copy.deepcopy(model)
    ids = torch.randint(0, 32, (2, 7), device="cuda")
    expected = reference(ids, use_cache=False).logits
    expected.float().square().mean().backward()
    monkeypatch.setattr(NemotronHAttention, "forward", NemotronHAttention.forward)
    monkeypatch.setattr(
        NemotronHAttention,
        "_original_forward",
        getattr(NemotronHAttention, "_original_forward", NemotronHAttention.forward),
        raising=False,
    )
    monkeypatch.setattr(
        lora_kernels, "get_attention_cls_from_config", lambda cfg: NemotronHAttention
    )
    monkeypatch.delattr(NemotronHAttention, "_original_forward")
    calls = []
    original = kernels._apply_native_quantized_lora

    def record(proj, value):
        calls.append(proj)
        return original(proj, value)

    monkeypatch.setattr(kernels, "_apply_native_quantized_lora", record)
    cfg = DictDefault(lora_qkv_kernel=True, lora_o_kernel=True, lora_mlp_kernel=True)
    lora_kernels.patch_self_attn_lora(cfg)
    lora_kernels.apply_lora_kernel_patches(model, cfg)
    actual = model(ids, use_cache=False).logits
    actual.float().square().mean().backward()
    assert len(calls) == 4
    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.003)
    for (name, p), (_, q) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if p.requires_grad:
            assert p.grad.dtype == torch.float32
            assert (p.grad - q.grad).norm() / q.grad.norm() < 0.04, name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["swiglu", "geglu"])
def test_standalone_activation_retained_backward_preserves_upstream(kind):
    from axolotl.kernels.lora import _GatedActivation

    torch.manual_seed(22)
    gate = torch.randn(
        2, 7, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    up = torch.randn_like(gate, requires_grad=True)
    actual = _GatedActivation.apply(gate, up, kind)
    grad = torch.randn_like(actual)
    saved_grad = grad.clone()
    actual.backward(grad, retain_graph=True)
    first_gate, first_up = gate.grad.clone(), up.grad.clone()
    actual.backward(grad, retain_graph=True)
    torch.testing.assert_close(grad, saved_grad, rtol=0, atol=0)
    torch.testing.assert_close(gate.grad, first_gate * 2, rtol=0, atol=0)
    torch.testing.assert_close(up.grad, first_up * 2, rtol=0, atol=0)


def test_dynamic_nvfp4_activation_quantization_fails_before_training():
    pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    from axolotl.kernels.lora import apply_lora_linear

    model = projection()
    weight = NVFP4Tensor.to_nvfp4(model.base_layer.weight.detach())
    weight.act_quant_kwargs = object()
    model.base_layer.weight = nn.Parameter(weight, requires_grad=False)
    with pytest.raises(NotImplementedError, match="weight-only NVFP4"):
        apply_lora_linear(
            model, torch.randn(2, 32, dtype=torch.bfloat16, requires_grad=True)
        )


def test_axolotl_nvfp4_quantization_produces_weight_only_base():
    pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    from torchao.quantization import quantize_

    from axolotl.utils.quantization import get_quantization_config
    from axolotl.utils.schemas.enums import TorchAOQuantDType

    model = nn.Sequential(nn.Linear(32, 32, bias=False).bfloat16())
    config = get_quantization_config(TorchAOQuantDType.nvfp4)
    quantize_(model, config)
    assert type(model[0].weight).__name__ == "NVFP4Tensor"
    assert model[0].weight.act_quant_kwargs is None


def test_native_nvfp4_quantizer_allows_only_frozen_adapter_training(tmp_path):
    pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments
    from transformers.trainer_utils import validate_quantization_for_training

    from axolotl.monkeypatch.torchao_lora import enable_native_nvfp4_lora_training
    from axolotl.utils.quantization import quantize_model
    from axolotl.utils.schemas.enums import TorchAOQuantDType

    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    ).bfloat16()
    quantize_model(base, TorchAOQuantDType.nvfp4)
    base.is_quantized = True
    assert not enable_native_nvfp4_lora_training(base)
    model = get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    with pytest.raises(ValueError, match="do not support training"):
        validate_quantization_for_training(model)
    original_quantizer = model.hf_quantizer
    native = [p for p in model.parameters() if type(p).__name__ == "NVFP4Tensor"]
    native[0].act_quant_kwargs = object()
    assert not enable_native_nvfp4_lora_training(model)
    native[0].act_quant_kwargs = None
    assert enable_native_nvfp4_lora_training(model)
    assert model.hf_quantizer is original_quantizer
    validate_quantization_for_training(model)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[]),
    )
    assert trainer.model is model
