"""LoRA factors are drawn from the configured seed, not from load-path RNG state."""

import pytest
import torch
from peft import LoraConfig
from torch import nn

from axolotl.loaders import adapter as adapter_module
from axolotl.utils.dict import DictDefault


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])


def _lora_factors(seed: int, *, pre_draws: int) -> dict[str, torch.Tensor]:
    """Adapter init after `pre_draws` unrelated RNG draws, as a load path would make."""
    torch.manual_seed(1234)
    model = _TinyModel()
    for _ in range(pre_draws):
        torch.rand(3)
    cfg = DictDefault(seed=seed, lora_fp32_gradients=False, deepspeed=None)
    peft_model, _ = adapter_module.load_lora(model, cfg, inference=False)
    factors = {
        name: param.detach().clone()
        for name, param in peft_model.named_parameters()
        if "lora_A" in name
    }
    assert len(factors) == 4, sorted(factors)
    return factors


@pytest.fixture(autouse=True)
def _tiny_lora_config(monkeypatch):
    monkeypatch.setattr(
        adapter_module,
        "_build_peft_lora_config",
        lambda model, cfg: LoraConfig(r=2, target_modules=["q_proj", "v_proj"]),
    )


def test_same_seed_gives_identical_adapters_despite_different_rng_history():
    first = _lora_factors(7, pre_draws=0)
    second = _lora_factors(7, pre_draws=5)

    assert first.keys() == second.keys()
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def test_different_seeds_give_different_adapters():
    first = _lora_factors(7, pre_draws=0)
    second = _lora_factors(8, pre_draws=0)

    assert any(not torch.equal(first[name], second[name]) for name in first)


def test_unseeded_config_leaves_rng_history_in_charge():
    first = _lora_factors(None, pre_draws=0)
    second = _lora_factors(None, pre_draws=5)

    assert any(not torch.equal(first[name], second[name]) for name in first)
