"""FP32 LoRA accumulation and fused parameter-gradient reduction precision."""

import copy

import pytest
import torch
from peft import LoraConfig
from peft.tuners.lora.layer import Linear
from torch import nn

from axolotl.utils.lora_precision import upcast_lora_parameters


def projection(device="cpu"):
    cfg = LoraConfig(r=4, lora_alpha=8)
    layer = Linear(
        nn.Linear(32, 32, bias=False, dtype=torch.bfloat16, device=device),
        adapter_name="default",
        config=cfg,
        r=4,
        lora_alpha=8,
    ).to(torch.bfloat16)
    layer.base_layer.requires_grad_(False)
    nn.init.normal_(layer.lora_B["default"].weight, std=0.02)
    return layer


def test_upcast_preserves_identity_base_and_unrelated_parameters():
    model = nn.Module()
    model.proj = projection()
    model.extra = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    identities = {name: id(p) for name, p in model.named_parameters()}
    upcast_lora_parameters(model)
    for name, p in model.named_parameters():
        assert id(p) == identities[name]
        assert p.dtype == (torch.float32 if "lora_" in name else torch.bfloat16)
    x = torch.randn(2, 32, dtype=torch.bfloat16)
    model.proj(x).float().square().mean().backward()
    assert all(
        p.grad.dtype == torch.float32
        for n, p in model.named_parameters()
        if "lora_" in n
    )


@pytest.mark.parametrize(
    "extra", [{}, {"fsdp_config": {}}, {"deepspeed": "zero2.json"}]
)
def test_precision_config_accepts_distributed_backends(extra):
    from axolotl.utils.schemas.config import AxolotlConfigWCapabilities

    cfg = dict(adapter="lora", lora_fp32_gradients=True) | extra
    assert AxolotlConfigWCapabilities.check_lora_fp32_gradients(cfg) == cfg
    with pytest.raises(ValueError, match="lora_fp32_gradients"):
        AxolotlConfigWCapabilities.check_lora_fp32_gradients(cfg | {"adapter": None})


def test_distributed_precision_policies():
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from axolotl.utils.lora_precision import (
        configure_deepspeed_lora_precision,
        lora_fsdp2_precision_policy,
    )

    original = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, output_dtype=torch.bfloat16
    )
    policy = lora_fsdp2_precision_policy(original)
    assert policy.param_dtype is None
    assert policy.reduce_dtype == torch.float32
    assert policy.output_dtype == original.output_dtype
    assert original.param_dtype == torch.bfloat16
    ds = {
        "data_types": {"grad_accum_dtype": "bf16"},
        "fp16": {"fp16_master_weights_and_grads": True},
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 3},
    }
    fixed = configure_deepspeed_lora_precision(ds)
    assert ds["data_types"]["grad_accum_dtype"] == "bf16"
    assert fixed["data_types"]["grad_accum_dtype"] == "fp32"
    assert fixed["communication_data_type"] == "fp32"
    assert fixed["bf16"]["bf16_master_weights_and_grads"] is False
    assert fixed["fp16"]["fp16_master_weights_and_grads"] is False
    assert ds["fp16"]["fp16_master_weights_and_grads"] is True
    assert fixed["zero_optimization"]["stage"] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_linear_fp32_reduction_and_microbatch_accumulation():
    from axolotl.kernels.lora import apply_lora_linear

    torch.manual_seed(57)
    model = projection("cuda")
    upcast_lora_parameters(model)
    a = model.lora_A["default"].weight
    b = model.lora_B["default"].weight
    expected_a, expected_b = torch.zeros_like(a), torch.zeros_like(b)
    for _ in range(3):
        x = torch.randn(
            2, 129, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        grad = torch.randn_like(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = apply_lora_linear(model, x)
        assert actual.dtype == torch.bfloat16
        actual.backward(grad)
        xf, gf = x.detach().flatten(0, 1), grad.flatten(0, 1)
        grad_b = gf @ b.detach().bfloat16()
        hidden = a.detach().bfloat16() @ xf.t()
        expected_a += (xf.float().t() @ grad_b.float()).t() * 2
        expected_b += (hidden.float() @ gf.float()).t() * 2
        assert a.grad.dtype == b.grad.dtype == torch.float32
        torch.testing.assert_close(a.grad, expected_a, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(b.grad, expected_b, rtol=1e-5, atol=1e-5)
        assert x.grad.dtype == torch.bfloat16
    # The new result retains precision beyond simply casting a BF16 reduction.
    assert not torch.equal(b.grad, b.grad.bfloat16().float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["qkv", "qk", "swiglu", "geglu", "gdn"])
def test_fused_groups_keep_fp32_parameter_gradients(kind, monkeypatch):
    from axolotl.kernels import lora as kernels

    torch.manual_seed(13)
    model = nn.Module()
    names = {
        "qkv": ["q_proj", "k_proj", "v_proj"],
        "qk": ["q_proj", "k_proj"],
        "swiglu": ["gate_proj", "up_proj", "down_proj"],
        "geglu": ["gate_proj", "up_proj", "down_proj"],
        "gdn": ["in_proj_qkv", "in_proj_z"],
    }[kind]
    for name in names:
        setattr(model, name, projection("cuda"))
    upcast_lora_parameters(model)
    reference = copy.deepcopy(model)
    calls = []
    original = kernels._gradient_mm

    def reduction(left, right, dtype, scale):
        result = original(left, right, dtype, scale)
        assert left.dtype == right.dtype == torch.bfloat16
        assert dtype == result.dtype == torch.float32
        calls.append(result)
        return result

    monkeypatch.setattr(kernels, "_gradient_mm", reduction)
    x = torch.randn(2, 9, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if kind in ("qkv", "qk"):
            actual = getattr(kernels, "apply_lora_" + kind)(model, x)
            expected = tuple(getattr(reference, name)(ref_x) for name in names)
            if kind == "qk":
                expected = (*expected, expected[1])
        elif kind == "gdn":
            actual = tuple(
                kernels.apply_lora_gdn_in_proj(model, x, tuple(names)).values()
            )
            expected = tuple(getattr(reference, name)(ref_x) for name in names)
        else:
            actual = (getattr(kernels, "apply_lora_mlp_" + kind)(model, x),)
            gate, up = reference.gate_proj(ref_x), reference.up_proj(ref_x)
            act = (
                torch.nn.functional.silu
                if kind == "swiglu"
                else torch.nn.functional.gelu
            )
            expected = (reference.down_proj(act(gate) * up),)
    sum(v.float().square().mean() for v in actual).backward()
    sum(v.float().square().mean() for v in expected).backward()
    assert len(calls) == 2 * len(names)
    for value, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, target, rtol=0.03, atol=0.01)
    for (_, p), (_, q) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if p.requires_grad:
            assert p.grad.dtype == torch.float32
            relative_error = (p.grad - q.grad).norm() / q.grad.norm().clamp_min(1e-12)
            assert relative_error < 0.03


@pytest.mark.parametrize("loaded", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_loader_precision_for_new_and_saved_adapters(tmp_path, loaded, enabled):
    from transformers import LlamaConfig, LlamaForCausalLM

    from axolotl.loaders.adapter import load_lora
    from axolotl.utils.dict import DictDefault

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    cfg = DictDefault(
        adapter="lora",
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=["q_proj"],
        peft_autocast_adapter_dtype=False,
        torch_dtype=torch.bfloat16,
        lora_fp32_gradients=enabled,
    )
    if loaded:
        saved, _ = load_lora(LlamaForCausalLM(config).bfloat16(), cfg)
        saved.save_pretrained(tmp_path)
        cfg.lora_model_dir = str(tmp_path)
    model, _ = load_lora(LlamaForCausalLM(config).bfloat16(), cfg)
    for name, parameter in model.named_parameters():
        expected = torch.float32 if enabled and "lora_" in name else torch.bfloat16
        assert parameter.dtype == expected
    model(
        torch.tensor([[1, 2, 3, 4]]), labels=torch.tensor([[1, 2, 3, 4]])
    ).loss.backward()
    assert all(
        p.grad.dtype == (torch.float32 if enabled else torch.bfloat16)
        for p in model.parameters()
        if p.requires_grad
    )


def test_quantized_fsdp_residual_cast_preserves_lora():
    from axolotl.monkeypatch.accelerate.fsdp2_quantized import cast_residual_fp32

    model = nn.Sequential(projection())
    upcast_lora_parameters(model)
    model.extra = nn.Parameter(torch.ones(2, dtype=torch.float32))
    model._axolotl_lora_fp32_gradients = True
    assert cast_residual_fp32(model) == 1
    assert model.extra.dtype == torch.bfloat16
    assert all(
        p.dtype == torch.float32 for n, p in model.named_parameters() if "lora_" in n
    )


@pytest.mark.parametrize("from_file", [False, True])
def test_deepspeed_setup_writes_precision_config(tmp_path, monkeypatch, from_file):
    import json
    import os
    from pathlib import Path

    from axolotl.utils import distributed, trainer
    from axolotl.utils.dict import DictDefault

    original = {"zero_optimization": {"stage": 2}, "bf16": {"enabled": True}}
    source = tmp_path / "original.json"
    source.write_text(json.dumps(original))
    cfg = DictDefault(
        deepspeed=str(source) if from_file else original,
        lora_fp32_gradients=True,
        gradient_accumulation_steps=3,
        use_ray=True,
    )
    monkeypatch.setattr(
        "axolotl.monkeypatch.deepspeed_utils.patch_zero_gradient_accumulation_dtype",
        lambda: None,
    )
    monkeypatch.setattr(distributed, "distributed_state", None)
    monkeypatch.setattr(trainer, "init_distributed_state", lambda: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(
        "transformers.integrations.deepspeed.HfTrainerDeepSpeedConfig",
        lambda config: None,
    )
    for name in (
        "ACCELERATE_USE_DEEPSPEED",
        "ACCELERATE_DEEPSPEED_CONFIG_FILE",
        "ACCELERATE_GRADIENT_ACCUMULATION_STEPS",
    ):
        monkeypatch.setenv(name, "")
    trainer.setup_deepspeed_env(cfg)
    output = Path(cfg.deepspeed)
    try:
        fixed = json.loads(output.read_text())
        assert fixed["data_types"]["grad_accum_dtype"] == "fp32"
        assert fixed["communication_data_type"] == "fp32"
        assert fixed["zero_optimization"] == original["zero_optimization"]
        assert os.environ["ACCELERATE_DEEPSPEED_CONFIG_FILE"] == str(output)
        assert json.loads(source.read_text()) == original
        assert "data_types" not in original
    finally:
        output.unlink()


def test_deepspeed_accumulation_cast_precedes_sum(monkeypatch):
    from types import SimpleNamespace

    pytest.importorskip("deepspeed")
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    from axolotl.monkeypatch.deepspeed_utils import (
        patch_zero_gradient_accumulation_dtype,
    )

    monkeypatch.setattr(
        DeepSpeedZeroOptimizer,
        "get_all_grad_tensors",
        DeepSpeedZeroOptimizer.get_all_grad_tensors,
    )
    patch_zero_gradient_accumulation_dtype()
    patched = DeepSpeedZeroOptimizer.get_all_grad_tensors
    patch_zero_gradient_accumulation_dtype()
    assert DeepSpeedZeroOptimizer.get_all_grad_tensors is patched
    optimizer = SimpleNamespace(get_param_gradient_attribute=lambda p: p.grad)
    parameter = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
    total = torch.zeros(1)
    for value in (1.0, 0.001, -1.0):
        parameter.grad = torch.full_like(parameter, value)
        gradient = patched(optimizer, [parameter], torch.float32)[0]
        assert gradient.dtype == torch.float32
        total.add_(gradient)
    assert total.item() == torch.tensor(0.001, dtype=torch.bfloat16).float().item()


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "nvme"])
def test_deepspeed_rejects_low_precision_offload_accumulation(stage, device):
    from axolotl.utils.lora_precision import configure_deepspeed_lora_precision

    with pytest.raises(ValueError, match="optimizer offload"):
        configure_deepspeed_lora_precision(
            {
                "zero_optimization": {
                    "stage": stage,
                    "offload_optimizer": {"device": device},
                },
            }
        )


@pytest.mark.parametrize(
    "requested,communication",
    [(False, torch.float32), (True, torch.float32), (True, torch.bfloat16)],
)
def test_deepspeed_reduction_preserves_mean_and_restores_dtype(
    monkeypatch, requested, communication
):
    from types import SimpleNamespace

    pytest.importorskip("deepspeed")
    from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    from axolotl.monkeypatch.deepspeed_utils import (
        patch_zero_gradient_accumulation_dtype,
    )

    original = DeepSpeedZeroOptimizer.get_all_grad_tensors
    monkeypatch.setattr(
        DeepSpeedZeroOptimizer,
        "get_all_grad_tensors",
        getattr(original, "__wrapped__", original),
    )
    for cls, names in (
        (
            DeepSpeedZeroOptimizer,
            ("get_gradient_for_reduction", "setup_buckets", "copy_grads_in_partition"),
        ),
        (
            DeepSpeedZeroOptimizer_Stage3,
            (
                "_DeepSpeedZeroOptimizer_Stage3__avg_scatter_grads",
                "_DeepSpeedZeroOptimizer_Stage3__avg_scatter_contiguous_grads",
            ),
        ),
    ):
        for name in names:
            method = getattr(cls, name)
            monkeypatch.setattr(cls, name, getattr(method, "__wrapped__", method))

    active = requested and communication == torch.float32

    def allocation(self, fail=False):
        assert self.dtype == (torch.float32 if active else torch.bfloat16)
        if fail:
            raise RuntimeError("allocation failed")
        return torch.empty(2, dtype=self.dtype)

    def scatter(self, parameters, communication_data_type):
        if getattr(self, "fail", False):
            raise RuntimeError("scatter failed")
        averaged = parameters[0].grad.float().add(1.0078125).div(2)
        return averaged.to(self.dtype)

    monkeypatch.setattr(DeepSpeedZeroOptimizer, "setup_buckets", allocation)
    name = "_DeepSpeedZeroOptimizer_Stage3__avg_scatter_grads"
    monkeypatch.setattr(DeepSpeedZeroOptimizer_Stage3, name, scatter)
    patch_zero_gradient_accumulation_dtype()
    optimizer = SimpleNamespace(
        dtype=torch.bfloat16,
        gradient_accumulation_dtype=torch.float32 if requested else torch.bfloat16,
        communication_data_type=communication,
        use_grad_accum_attribute=False,
    )
    assert DeepSpeedZeroOptimizer.setup_buckets(optimizer).dtype == (
        torch.float32 if active else torch.bfloat16
    )
    with pytest.raises(RuntimeError, match="allocation failed"):
        DeepSpeedZeroOptimizer.setup_buckets(optimizer, fail=True)
    assert optimizer.dtype == torch.bfloat16
    parameter = nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    parameter.grad = torch.ones_like(parameter)
    result = getattr(DeepSpeedZeroOptimizer_Stage3, name)(
        optimizer, [parameter], communication
    )
    assert result.dtype == (torch.float32 if active else torch.bfloat16)
    assert torch.equal(result.float(), torch.full((3,), 1.00390625 if active else 1.0))
    assert optimizer.dtype == parameter.dtype == torch.bfloat16

    optimizer.fail = True
    with pytest.raises(RuntimeError, match="scatter failed"):
        getattr(DeepSpeedZeroOptimizer_Stage3, name)(
            optimizer, [parameter], communication
        )
    assert optimizer.dtype == torch.bfloat16
