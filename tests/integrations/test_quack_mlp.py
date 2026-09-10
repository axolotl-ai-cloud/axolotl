"""CPU-side tests for the quack fused-MLP plugin (wiring, eligibility, fallback).

The quack kernel itself is Hopper+/GPU-only; numeric correctness is covered by
`tests/e2e/kernels/test_quack_mlp.py`. Here we only verify that the plumbing is
correct and that patching is a no-op (identical output) when the kernel is
ineligible, e.g. on CPU.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from axolotl.integrations.quack_kernels.args import QuackKernelsArgs
from axolotl.integrations.quack_kernels.mlp import (
    apply_quack_mlp,
    mlp_is_eligible,
    resolve_gated_activation,
)
from axolotl.utils.config import prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault


def _minimal_cfg(**overrides):
    return DictDefault(
        {
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
            "learning_rate": 0.000001,
            "datasets": [{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "plugins": ["axolotl.integrations.quack_kernels.QuackKernelsPlugin"],
        }
        | overrides
    )


class RefMLP(nn.Module):
    def __init__(self, hidden: int, inter: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=bias)
        self.up_proj = nn.Linear(hidden, inter, bias=bias)
        self.down_proj = nn.Linear(inter, hidden, bias=bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Layer(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.mlp = RefMLP(hidden, inter)


class TinyModel(nn.Module):
    def __init__(
        self, hidden: int = 32, inter: int = 64, n_layers: int = 2, act: str = "silu"
    ):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(hidden, inter) for _ in range(n_layers)])
        self.config = SimpleNamespace(hidden_act=act)


# --- config validation ---------------------------------------------------------


@pytest.mark.parametrize(
    "conflict", ["liger_glu_activation", "liger_swiglu", "lora_mlp_kernel", "tiled_mlp"]
)
def test_conflicting_mlp_kernels_rejected(conflict):
    with pytest.raises(ValueError, match="only one MLP kernel"):
        QuackKernelsArgs(**{"quack_mlp_kernel": True, conflict: True})


def test_tiled_mlp_allowed_with_original_mlp():
    QuackKernelsArgs(
        quack_mlp_kernel=True, tiled_mlp=True, tiled_mlp_use_original_mlp=True
    )


def test_no_conflict_when_disabled():
    QuackKernelsArgs(quack_mlp_kernel=None, liger_glu_activation=True)


# --- activation + eligibility ---------------------------------------------------


@pytest.mark.parametrize(
    "act,expected",
    [
        ("silu", "swiglu"),
        ("gelu_pytorch_tanh", "geglu"),
        ("gelu_new", "geglu"),
        ("gelu", None),  # erf gelu: quack geglu is tanh-approx only, so skip
        ("relu", "reglu"),
        ("some_unknown_act", None),
    ],
)
def test_resolve_gated_activation(act, expected):
    assert resolve_gated_activation(SimpleNamespace(hidden_act=act)) == expected


def test_resolve_activation_nested_text_config():
    cfg = SimpleNamespace(text_config=SimpleNamespace(hidden_act="silu"))
    assert resolve_gated_activation(cfg) == "swiglu"


def test_eligible_dense_mlp():
    mlp = RefMLP(32, 64).to(torch.bfloat16)
    assert mlp_is_eligible(mlp)


def test_bias_mlp_not_eligible():
    assert not mlp_is_eligible(RefMLP(32, 64, bias=True).to(torch.bfloat16))


def test_fp32_mlp_not_eligible():
    assert not mlp_is_eligible(RefMLP(32, 64))  # fp32 default


def test_non_divisible_mlp_not_eligible():
    assert not mlp_is_eligible(RefMLP(32, 60).to(torch.bfloat16))  # inter % 8 != 0


def test_lora_like_projection_not_eligible():
    class LoRALinear(nn.Linear):  # PEFT-style subclass; base .weight drops the delta
        pass

    mlp = RefMLP(32, 64).to(torch.bfloat16)
    mlp.gate_proj = LoRALinear(32, 64, bias=False).to(torch.bfloat16)
    assert not mlp_is_eligible(mlp)


# --- patching is a safe no-op on CPU (kernel ineligible -> fallback) -------------


def test_cpu_patch_is_identity():
    torch.manual_seed(0)
    # bf16 so the MLP is kernel-eligible and actually gets patched; on CPU the patched
    # forward then falls back to the original (is_cuda is False), so output is unchanged.
    model = TinyModel().to(torch.bfloat16)
    x = torch.randn(4, 8, 32, dtype=torch.bfloat16, requires_grad=True)

    before = model.layers[0].mlp(x.detach())

    n = apply_quack_mlp(model, cfg=SimpleNamespace(quack_mlp_kernel=True))
    assert n == 2  # both layers' MLPs matched

    after = model.layers[0].mlp(x)
    torch.testing.assert_close(after, before)

    # autograd still flows through the patched (fallback) forward
    after.sum().backward()
    assert model.layers[0].mlp.gate_proj.weight.grad is not None


def test_unsupported_activation_skips_patching():
    model = TinyModel(act="tanh")  # not a gated activation
    n = apply_quack_mlp(model, cfg=SimpleNamespace(quack_mlp_kernel=True))
    assert n == 0


# --- plugin merges through the real config machinery ----------------------------


def test_plugin_exposes_flag_through_validate_config():
    cfg = _minimal_cfg(quack_mlp_kernel=True)
    prepare_plugins(cfg)
    updated = validate_config(cfg)
    assert updated.quack_mlp_kernel is True


def test_plugin_conflict_raised_through_validate_config():
    cfg = _minimal_cfg(quack_mlp_kernel=True, liger_glu_activation=True)
    with pytest.raises(ValueError, match="only one MLP kernel"):
        prepare_plugins(cfg)
        validate_config(cfg)
