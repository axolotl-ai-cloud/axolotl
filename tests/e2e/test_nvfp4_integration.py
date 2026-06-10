"""Schema, gate, and (GPU-gated) end-to-end tests for nvfp4_training."""

import pytest

import axolotl.utils.nvfp4_training as nvfp4_mod
from axolotl.utils.schemas.config import (
    AxolotlConfigWCapabilities,
    AxolotlInputConfig,
)

BASE = {
    "base_model": "Qwen/Qwen2.5-0.5B",
    "datasets": [{"path": "tatsu-lab/alpaca", "type": "alpaca"}],
    "learning_rate": 1e-5,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "num_epochs": 1,
}

# Capability gates live on AxolotlConfigWCapabilities; supply the two capability
# blocks so the gate validator runs without touching real hardware.
CAPS = {
    "capabilities": {
        "bf16": True,
        "fp8": True,
        "n_gpu": 1,
        "compute_capability": "sm_120",
    },
    "env_capabilities": {"torch_version": "2.8.0"},
}


def _supported(monkeypatch, ok, reason=""):
    monkeypatch.setattr(nvfp4_mod, "nvfp4_supported", lambda: (ok, reason))


def test_schema_accepts_valid_nvfp4_config(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(
        **BASE,
        nvfp4_training={
            "enabled": True,
            "stochastic_rounding": True,
            "hadamard": True,
            "exclude_modules": ["lm_head", "embed_tokens"],
            "skip_first_n_blocks": 1,
            "skip_last_n_blocks": 1,
        },
    )
    assert cfg.nvfp4_training.enabled is True
    assert cfg.nvfp4_training.skip_first_n_blocks == 1


def test_schema_backend_defaults_native_and_accepts_te(monkeypatch):
    _supported(monkeypatch, True)
    assert (
        AxolotlInputConfig(
            **BASE, nvfp4_training={"enabled": True}
        ).nvfp4_training.backend
        == "native"
    )
    cfg = AxolotlInputConfig(**BASE, nvfp4_training={"enabled": True, "backend": "te"})
    assert cfg.nvfp4_training.backend == "te"


def test_schema_accepts_shared_lora_base_fprop(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(
        **BASE,
        nvfp4_training={"enabled": True, "shared_lora_base_fprop": True},
    )
    assert cfg.nvfp4_training.shared_lora_base_fprop is True

    cfg = AxolotlInputConfig(**BASE, nvfp4_training={"enabled": True})
    assert cfg.nvfp4_training.shared_lora_base_fprop is None


def test_schema_accepts_fp8_lm_head_eval_knobs(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(
        **BASE,
        nvfp4_training={
            "enabled": True,
            "fp8_lm_head": True,
            "fp8_lm_head_cross_entropy": True,
            "fp8_lm_head_granularity": "rowwise",
        },
    )
    assert cfg.nvfp4_training.fp8_lm_head is True
    assert cfg.nvfp4_training.fp8_lm_head_cross_entropy is True
    assert cfg.nvfp4_training.fp8_lm_head_granularity == "rowwise"


def test_schema_refuses_fp8_lm_head_ce_with_other_ce(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="Only one cross entropy optimization"):
        AxolotlInputConfig(
            **BASE,
            cut_cross_entropy=True,
            nvfp4_training={
                "enabled": True,
                "fp8_lm_head_cross_entropy": True,
            },
        )


def test_disabled_fp8_lm_head_ce_does_not_conflict(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(
        **BASE,
        chunked_cross_entropy=True,
        nvfp4_training={
            "enabled": False,
            "fp8_lm_head_cross_entropy": True,
        },
    )
    assert cfg.chunked_cross_entropy is True
    assert cfg.nvfp4_training.enabled is False


def test_gate_refuses_fp8_lm_head_ce_with_quantized_lm_head(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="fp8_lm_head_cross_entropy"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            nvfp4_training={
                "enabled": True,
                "quantize_lm_head": True,
                "fp8_lm_head_cross_entropy": True,
            },
        )


def test_schema_accepts_bf16_lm_head_ce(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(
        **BASE,
        nvfp4_training={"enabled": True, "bf16_lm_head_cross_entropy": True},
    )
    assert cfg.nvfp4_training.bf16_lm_head_cross_entropy is True


def test_schema_refuses_bf16_lm_head_ce_with_other_ce(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="Only one cross entropy optimization"):
        AxolotlInputConfig(
            **BASE,
            cut_cross_entropy=True,
            nvfp4_training={"enabled": True, "bf16_lm_head_cross_entropy": True},
        )


def test_gate_refuses_bf16_lm_head_ce_with_quantized_lm_head(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="bf16_lm_head_cross_entropy"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            nvfp4_training={
                "enabled": True,
                "quantize_lm_head": True,
                "bf16_lm_head_cross_entropy": True,
            },
        )


def test_gate_refuses_unsupported_hardware(monkeypatch):
    _supported(monkeypatch, False, "no Blackwell here")
    with pytest.raises(ValueError, match="no Blackwell here"):
        AxolotlConfigWCapabilities(**BASE, **CAPS, nvfp4_training={"enabled": True})


def test_gate_allows_lora(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlConfigWCapabilities(
        **BASE, **CAPS, adapter="lora", nvfp4_training={"enabled": True}
    )
    assert cfg.adapter == "lora"
    assert cfg.nvfp4_training.quantize_base is False


def test_gate_allows_lora_quantize_base(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlConfigWCapabilities(
        **BASE,
        **CAPS,
        adapter="lora",
        nvfp4_training={"enabled": True, "quantize_base": True},
    )
    assert cfg.nvfp4_training.quantize_base is True


def test_gate_refuses_unsupported_adapter(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="lora, qlora"):
        AxolotlConfigWCapabilities(
            **BASE, **CAPS, adapter="llama-adapter", nvfp4_training={"enabled": True}
        )


def test_gate_refuses_load_in_4bit_with_quantize_base(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="load_in_4bit"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            adapter="lora",
            load_in_4bit=True,
            nvfp4_training={"enabled": True, "quantize_base": True},
        )


def test_gate_refuses_deepspeed(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="DeepSpeed"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            deepspeed="deepspeed_configs/zero3.json",
            nvfp4_training={"enabled": True},
        )


def test_gate_refuses_fp16(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="does not support fp16"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            fp16=True,
            bf16=False,
            nvfp4_training={"enabled": True},
        )


def test_disabled_nvfp4_skips_gate(monkeypatch):
    _supported(monkeypatch, False, "should not be raised")
    cfg = AxolotlConfigWCapabilities(**BASE, **CAPS, nvfp4_training={"enabled": False})
    assert cfg.nvfp4_training.enabled is False


def _tiny_lora_model():
    """A 2-layer toy model wrapped with a PEFT LoRA adapter (CPU-friendly)."""
    from peft import LoraConfig, get_peft_model
    from torch import nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32, bias=False)
            self.v_proj = nn.Linear(32, 32, bias=False)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(), Block()])

        def forward(self, x):
            for blk in self.layers:
                x = blk.v_proj(blk.q_proj(x))
            return x

    base = Net()
    lcfg = LoraConfig(r=4, target_modules=["q_proj", "v_proj"], lora_alpha=8)
    return get_peft_model(base, lcfg)


def _patch_manager(cfg_dict):
    from axolotl.loaders.patch_manager import PatchManager
    from axolotl.utils.dict import DictDefault

    return PatchManager(DictDefault(cfg_dict), model_config=DictDefault({}))


def test_qwen3_5_packing_patch_forwards_fla_compile_boundary(monkeypatch):
    _supported(monkeypatch, True)
    from axolotl.loaders import patch_manager
    from axolotl.monkeypatch.models.qwen3_5 import modeling

    captured = []

    def fake_patch(**kwargs):
        captured.append(kwargs)

    monkeypatch.setattr(patch_manager.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(modeling, "patch_qwen3_5_modeling_packing", fake_patch)

    pm = _patch_manager(
        {
            "model_config_type": "qwen3_5",
            "sample_packing": True,
            "bf16": True,
        }
    )

    pm._apply_model_specific_patches()

    pm = _patch_manager(
        {
            "model_config_type": "qwen3_5",
            "sample_packing": True,
            "nvfp4_training": {
                "enabled": True,
                "fla_causal_conv_compile_boundary": True,
            },
        }
    )

    pm._apply_model_specific_patches()

    assert captured[0]["fla_causal_conv_compile_boundary"] is False
    assert captured[1]["fla_causal_conv_compile_boundary"] is True


def test_apply_selects_lora_compute_mode(monkeypatch):
    """adapter=lora default -> base_layer becomes an FP4 compute base (the
    pre-quantized FP4-compute base, the recommended default)."""
    _supported(monkeypatch, True)
    from peft.tuners.lora import Linear as LoraLinear

    from axolotl.utils.nvfp4_training import (
        NVFP4ComputeBaseLinear,
        NVFP4FastComputeBaseLinear,
        NVFP4FastFrozenBaseLinear,
        NVFP4FrozenBaseLinear,
    )

    model = _tiny_lora_model()
    pm = _patch_manager({"adapter": "lora", "nvfp4_training": {"enabled": True}})
    pm._apply_nvfp4_training(model)

    bases = [m.base_layer for m in model.modules() if isinstance(m, LoraLinear)]
    assert bases and all(
        isinstance(b, (NVFP4ComputeBaseLinear, NVFP4FastComputeBaseLinear))
        for b in bases
    )
    assert not any(
        isinstance(b, (NVFP4FrozenBaseLinear, NVFP4FastFrozenBaseLinear)) for b in bases
    )


def test_apply_sets_shared_lora_base_fprop_flag(monkeypatch):
    _supported(monkeypatch, True)
    from axolotl.kernels import lora as lora_mod
    from axolotl.loaders.patch_manager import PatchManager
    from axolotl.utils import nvfp4_training as nvfp4_training_mod

    monkeypatch.setattr(
        nvfp4_training_mod,
        "convert_lora_base_to_nvfp4",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        PatchManager, "_nvfp4_apply_tied_or_lm_head", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        PatchManager, "_nvfp4_load_packed_sidecar", lambda *args, **kwargs: None
    )

    pm = _patch_manager(
        {
            "adapter": "lora",
            "nvfp4_training": {
                "enabled": True,
                "shared_lora_base_fprop": True,
                "fuse_rmsnorm": False,
            },
        }
    )
    pm._apply_nvfp4_training(object())
    assert lora_mod._NVFP4_SHARED_BASE_FPROP is True

    pm = _patch_manager(
        {
            "adapter": "lora",
            "nvfp4_training": {
                "enabled": True,
                "shared_lora_base_fprop": False,
                "fuse_rmsnorm": False,
            },
        }
    )
    pm._apply_nvfp4_training(object())
    assert lora_mod._NVFP4_SHARED_BASE_FPROP is False


def test_apply_selects_hp_mode_when_requested(monkeypatch):
    """base_mode: hp -> NVFP4Linear (HP frozen base, re-quantized each step)."""
    _supported(monkeypatch, True)
    from peft.tuners.lora import Linear as LoraLinear

    from axolotl.utils.nvfp4_training import NVFP4Linear

    model = _tiny_lora_model()
    pm = _patch_manager(
        {"adapter": "lora", "nvfp4_training": {"enabled": True, "base_mode": "hp"}}
    )
    pm._apply_nvfp4_training(model)
    bases = [m.base_layer for m in model.modules() if isinstance(m, LoraLinear)]
    assert bases and all(isinstance(b, NVFP4Linear) for b in bases)
    assert all(not b.weight.requires_grad for b in bases)


def test_apply_selects_fft_mode_when_no_adapter(monkeypatch):
    """No adapter -> raw nn.Linear swapped to NVFP4Linear (full fine-tune)."""
    _supported(monkeypatch, True)
    from torch import nn

    from axolotl.utils.nvfp4_training import NVFP4Linear

    model = nn.Sequential(nn.Linear(32, 32, bias=False), nn.Linear(32, 32, bias=False))
    pm = _patch_manager({"nvfp4_training": {"enabled": True}})
    pm._apply_nvfp4_training(model)
    assert any(isinstance(m, NVFP4Linear) for m in model.modules())


def test_apply_honors_block_exclusions(monkeypatch):
    """skip_last_n_blocks keeps the tail block's base_layer un-swapped."""
    _supported(monkeypatch, True)
    from peft.tuners.lora import Linear as LoraLinear

    from axolotl.utils.nvfp4_training import (
        NVFP4ComputeBaseLinear,
        NVFP4FastComputeBaseLinear,
    )

    model = _tiny_lora_model()
    pm = _patch_manager(
        {
            "adapter": "lora",
            "nvfp4_training": {"enabled": True, "skip_last_n_blocks": 1},
        }
    )
    pm._apply_nvfp4_training(model)

    swapped = {
        name: isinstance(
            m.base_layer, (NVFP4ComputeBaseLinear, NVFP4FastComputeBaseLinear)
        )
        for name, m in model.named_modules()
        if isinstance(m, LoraLinear)
    }
    # layer 0 swapped, layer 1 (last block) excluded
    assert any("layers.0." in n and v for n, v in swapped.items())
    assert all(not v for n, v in swapped.items() if "layers.1." in n)


_ok, _ = nvfp4_mod.nvfp4_supported()


@pytest.mark.skipif(not _ok, reason="requires Blackwell FP4 tensor cores")
def test_e2e_swap_and_train_step(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    from axolotl.utils.nvfp4_training import (
        NVFP4Linear,
        NVFP4Recipe,
        convert_to_nvfp4_training,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "axolotl-ai-co/tiny-qwen2-129m", torch_dtype=torch.bfloat16
    ).cuda()
    count = convert_to_nvfp4_training(model, NVFP4Recipe())
    assert count > 0
    assert any(isinstance(m, NVFP4Linear) for m in model.modules())

    ids = torch.randint(0, model.config.vocab_size, (1, 64), device="cuda")
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    assert torch.isfinite(out.loss).item()


@pytest.mark.skipif(not _ok, reason="requires Blackwell FP4 tensor cores")
@pytest.mark.parametrize("quantize_base", [False, True])
def test_e2e_lora_swap_and_train_step(quantize_base):
    """Both adapter modes: right NVFP4 class appears under PEFT and a step is finite."""
    import torch
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import Linear as LoraLinear
    from transformers import AutoModelForCausalLM

    from axolotl.utils.nvfp4_training import (
        NVFP4FastComputeBaseLinear,
        NVFP4FastFrozenBaseLinear,
        NVFP4FrozenBaseLinear,
        NVFP4Linear,
        NVFP4Recipe,
        convert_lora_base_to_nvfp4,
    )

    base = AutoModelForCausalLM.from_pretrained(
        "axolotl-ai-co/tiny-qwen2-129m", torch_dtype=torch.bfloat16
    ).cuda()
    model = get_peft_model(
        base, LoraConfig(r=8, target_modules=["q_proj", "v_proj"], lora_alpha=16)
    )
    count = convert_lora_base_to_nvfp4(
        model, NVFP4Recipe(), quantized_storage=quantize_base
    )
    assert count > 0

    expected = (
        (NVFP4FrozenBaseLinear, NVFP4FastFrozenBaseLinear)
        if quantize_base
        else (NVFP4Linear, NVFP4FastComputeBaseLinear)
    )
    bases = [m.base_layer for m in model.modules() if isinstance(m, LoraLinear)]
    assert any(isinstance(b, expected) for b in bases)

    ids = torch.randint(0, base.config.vocab_size, (1, 64), device="cuda")
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    assert torch.isfinite(out.loss).item()


def test_fsdp_nvfp4_class_is_picklable():
    """FSDP2's FULL_STATE_DICT save pickles the frozen NVFP4 params, so the
    lazily-built FSDP-hooked subclass must live at module scope (not be a
    ``<locals>`` class) or the checkpoint save crashes with
    ``Can't get local object '_fsdp_nvfp4_class.<locals>.FSDPNVFP4Tensor'``.
    Regression guard for that FSDP checkpoint-save bug."""
    import pickle

    pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    from axolotl.utils.nvfp4_training import _fsdp_nvfp4_class

    cls = _fsdp_nvfp4_class()
    assert cls.__qualname__ == "FSDPNVFP4Tensor"
    assert cls.__module__ == "axolotl.utils.nvfp4_training"
    # the exact operation that failed during the FSDP checkpoint save:
    assert pickle.loads(pickle.dumps(cls)) is cls
