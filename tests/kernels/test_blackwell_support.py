"""CPU-safe tests for the B200 factored LoRA MLP kernel's gating and env setup.

Covers the hard arch gate (mocked compute capabilities), config validation,
per-module eligibility, and the CUBLAS_WORKSPACE_CONFIG contract (applies when
unset in a fresh process, never clobbers an explicit value).
"""

import os
import subprocess
import sys
import warnings
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from axolotl.kernels.blackwell.support import (
    CUBLAS_WORKSPACE_CONFIG_RECOMMENDED,
    is_sm100,
    lora_mlp_b200_config_eligible,
    lora_mlp_b200_module_supported,
    maybe_set_cublas_workspace_config,
)
from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


def _lora_config(lora_dropout=0.0, use_dora=False):
    return SimpleNamespace(lora_dropout=lora_dropout, use_dora=use_dora)


def _mock_sm(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)


class TestArchGate:
    def test_sm100_detected(self, monkeypatch):
        _mock_sm(monkeypatch, (10, 0))
        assert is_sm100()
        eligible, reason = lora_mlp_b200_config_eligible(_lora_config(), "silu")
        assert eligible, reason

    @pytest.mark.parametrize("capability", [(12, 0), (9, 0), (8, 0), (10, 3)])
    def test_non_sm100_falls_back(self, monkeypatch, capability):
        _mock_sm(monkeypatch, capability)
        assert not is_sm100()
        eligible, reason = lora_mlp_b200_config_eligible(_lora_config(), "silu")
        assert not eligible
        assert str(capability) in reason

    def test_no_cuda_falls_back(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert not is_sm100()
        eligible, _ = lora_mlp_b200_config_eligible(_lora_config(), "silu")
        assert not eligible


class TestConfigEligibility:
    def test_dropout_rejected(self, monkeypatch):
        _mock_sm(monkeypatch, (10, 0))
        eligible, reason = lora_mlp_b200_config_eligible(
            _lora_config(lora_dropout=0.05), "silu"
        )
        assert not eligible
        assert "dropout" in reason

    def test_dora_rejected(self, monkeypatch):
        _mock_sm(monkeypatch, (10, 0))
        eligible, reason = lora_mlp_b200_config_eligible(
            _lora_config(use_dora=True), "silu"
        )
        assert not eligible
        assert "DoRA" in reason

    def test_non_silu_rejected(self, monkeypatch):
        _mock_sm(monkeypatch, (10, 0))
        eligible, reason = lora_mlp_b200_config_eligible(_lora_config(), "gelu")
        assert not eligible
        assert "gelu" in reason


class _MockLoRAProj(nn.Module):
    def __init__(self, in_features=32, out_features=64, rank=4, dtype=torch.bfloat16):
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features, bias=False, dtype=dtype)
        self.lora_A = nn.ModuleDict(
            {"default": nn.Linear(in_features, rank, bias=False, dtype=dtype)}
        )
        self.lora_B = nn.ModuleDict(
            {"default": nn.Linear(rank, out_features, bias=False, dtype=dtype)}
        )
        self.scaling = {"default": 0.5}
        self.active_adapter = "default"
        self.disable_adapters = False
        self.merged = False


class TestModuleEligibility:
    def _projs(self):
        return _MockLoRAProj(), _MockLoRAProj(), _MockLoRAProj(64, 32)

    def test_supported(self):
        supported, reason = lora_mlp_b200_module_supported(*self._projs())
        assert supported, reason

    def test_missing_adapter_rejected(self):
        gate, up, down = self._projs()
        del gate.lora_A
        supported, reason = lora_mlp_b200_module_supported(gate, up, down)
        assert not supported
        assert "no LoRA adapter" in reason

    def test_fp32_weight_rejected(self):
        gate, up, down = self._projs()
        up.base_layer.weight.data = up.base_layer.weight.data.float()
        supported, reason = lora_mlp_b200_module_supported(gate, up, down)
        assert not supported
        assert "bfloat16" in reason

    def test_base_bias_rejected(self):
        gate, up, down = self._projs()
        down.base_layer = nn.Linear(64, 32, bias=True, dtype=torch.bfloat16)
        supported, reason = lora_mlp_b200_module_supported(gate, up, down)
        assert not supported
        assert "bias" in reason

    def test_quantized_base_rejected(self):
        gate, up, down = self._projs()
        gate.base_layer.weight.quant_state = object()
        supported, reason = lora_mlp_b200_module_supported(gate, up, down)
        assert not supported
        assert "quantized" in reason


def _cfg(**extra):
    base = {
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "learning_rate": 1e-3,
        "datasets": [{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "sequence_len": 2048,
        "adapter": "lora",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "lora_target_linear": True,
        "lora_mlp_kernel_b200": True,
    }
    base.update(extra)
    return DictDefault(base)


def _validate(cfg, compute_capability="sm_100"):
    # The b200 validator lives on AxolotlConfigWCapabilities, the class the
    # real CLI path always uses -- so validate with capabilities like it does.
    return validate_config(
        cfg,
        capabilities={
            "bf16": True,
            "n_gpu": 1,
            "n_node": 1,
            "compute_capability": compute_capability,
        },
        env_capabilities={"torch_version": str(torch.__version__).split("+")[0]},
    )


@pytest.fixture(autouse=True)
def _preserve_cublas_env():
    saved = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    yield
    if saved is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = saved


class TestConfigValidation:
    def test_valid_config_enables_mlp_kernel(self):
        cfg = _validate(_cfg())
        assert cfg.lora_mlp_kernel is True
        assert cfg.lora_mlp_kernel_b200 is True

    def test_requires_lora_mlp_kernel(self):
        with pytest.raises(ValueError, match="requires lora_mlp_kernel"):
            _validate(_cfg(lora_mlp_kernel=False))

    def test_qlora_rejected(self):
        with pytest.raises(ValueError, match="adapter: lora"):
            _validate(_cfg(adapter="qlora", load_in_4bit=True))

    def test_load_in_4bit_rejected(self):
        with pytest.raises(ValueError, match="load_in_4bit"):
            _validate(_cfg(load_in_4bit=True))

    def test_lora_dropout_rejected(self):
        with pytest.raises(ValueError, match="dropout"):
            _validate(_cfg(lora_dropout=0.05))

    def test_dora_rejected(self):
        with pytest.raises(ValueError, match="DoRA"):
            _validate(_cfg(peft_use_dora=True))

    def test_fsdp_rejected(self):
        with pytest.raises(ValueError, match="FSDP"):
            _validate(
                _cfg(
                    fsdp_version=2,
                    fsdp_config={"transformer_layer_cls_to_wrap": "LlamaDecoderLayer"},
                )
            )

    def test_non_sm100_capability_warns_but_validates(self, caplog):
        cfg = _validate(_cfg(), compute_capability="sm_120")
        assert cfg.lora_mlp_kernel_b200 is True
        assert any("not sm_100" in r.message for r in caplog.records)

    def test_non_sm100_capability_does_not_set_cublas_env(self):
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        _validate(_cfg(), compute_capability="sm_120")
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ

    def test_sm100_capability_sets_cublas_env(self):
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        _validate(_cfg())
        assert (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            == CUBLAS_WORKSPACE_CONFIG_RECOMMENDED
        )

    def test_missing_bf16_warns(self, caplog):
        _validate(_cfg())
        assert any("bf16" in r.message for r in caplog.records)

    def test_bf16_enabled_does_not_warn(self, caplog):
        _validate(_cfg(bf16=True))
        assert not any("requires bf16" in r.message for r in caplog.records)


class TestCublasWorkspaceConfig:
    """The env var is read at cuBLAS handle creation, so the applies-when-unset
    behavior must be verified in a fresh subprocess."""

    def _run(self, code, env_overrides=None):
        env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
        if env_overrides:
            env.update(env_overrides)
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            check=False,
        )

    def test_applied_in_fresh_process_when_unset(self):
        result = self._run(
            "import os\n"
            "from axolotl.kernels.blackwell.support import maybe_set_cublas_workspace_config\n"
            "maybe_set_cublas_workspace_config()\n"
            "print(os.environ.get('CUBLAS_WORKSPACE_CONFIG'))"
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == CUBLAS_WORKSPACE_CONFIG_RECOMMENDED

    def test_never_clobbers_explicit_value(self):
        result = self._run(
            "import os, warnings\n"
            "from axolotl.kernels.blackwell.support import maybe_set_cublas_workspace_config\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    maybe_set_cublas_workspace_config()\n"
            "    print(os.environ.get('CUBLAS_WORKSPACE_CONFIG'))\n"
            "    print(len(w) > 0)",
            env_overrides={"CUBLAS_WORKSPACE_CONFIG": ":9999:1"},
        )
        assert result.returncode == 0, result.stderr
        lines = result.stdout.strip().splitlines()
        assert lines[0] == ":9999:1", "explicit user choice was clobbered"
        assert lines[1] == "True", "expected a warning for a differing explicit value"

    def test_noop_when_already_recommended(self):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG_RECOMMENDED
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            maybe_set_cublas_workspace_config()
        assert not w
        assert (
            os.environ["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG_RECOMMENDED
        )
