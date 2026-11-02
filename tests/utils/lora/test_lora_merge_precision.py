"""Merge-rounding diagnostic for merged LoRA weights."""

import logging

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from axolotl.utils import lora_merge_precision
from axolotl.utils.lora_merge_precision import (
    MergeRetention,
    accumulate_lora_linear,
    log_merge_retention,
    lora_delta,
    measure_lora_merge_retention,
    merged_lora_weight,
)


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(64, 48, bias=False)
        self.o = nn.Linear(48, 64, bias=False)

    def forward(self, x):
        return self.o(self.q(x))


def _peft(base_dtype=torch.bfloat16, b_std=1e-4, seed=0):
    torch.manual_seed(seed)
    model = get_peft_model(
        _Toy().to(base_dtype), LoraConfig(r=4, lora_alpha=8, target_modules=["q", "o"])
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, b_std)
    return model


def _reference(weight, lora_a, lora_b, scaling, dtype):
    delta = (lora_b.float() @ lora_a.float()) * scaling
    base = weight.float()
    change = (base + delta).to(dtype).float() - base.to(dtype).float()
    return delta, change


def test_retained_is_projection_onto_delta():
    stats = MergeRetention()
    delta = torch.tensor([1.0, 2.0, 0.0, -1.0])
    stats.update(delta, torch.tensor([1.0, 0.0, 0.5, -1.0]))
    assert stats.retained == pytest.approx((1.0 + 0.0 + 0.0 + 1.0) / 6.0)
    assert stats.changed_fraction == pytest.approx(3 / 4)


def test_empty_delta_has_no_retention():
    stats = MergeRetention()
    stats.update(torch.zeros(3), torch.zeros(3))
    assert stats.retained is None


@pytest.mark.parametrize(
    "scale, low, high",
    [(1e-6, 0.0, 0.05), (1e-1, 0.95, 1.05)],
)
def test_sub_ulp_deltas_round_away_and_large_ones_survive(scale, low, high):
    torch.manual_seed(0)
    weight = torch.randn(128, 64).to(torch.bfloat16)
    lora_a = torch.randn(8, 64)
    lora_b = torch.randn(128, 8) * scale
    stats = MergeRetention()
    accumulate_lora_linear(stats, weight, lora_a, lora_b, 2.0, torch.bfloat16)
    assert low <= stats.retained <= high


def test_chunked_accumulation_matches_reference(monkeypatch):
    torch.manual_seed(0)
    weight = torch.randn(37, 16).to(torch.bfloat16)
    lora_a = torch.randn(4, 16)
    lora_b = torch.randn(37, 4) * 3e-3
    monkeypatch.setattr(lora_merge_precision, "_CHUNK_ELEMENTS", 16 * 5)
    stats = MergeRetention()
    accumulate_lora_linear(stats, weight, lora_a, lora_b, 2.0, torch.bfloat16)

    delta, change = _reference(weight, lora_a, lora_b, 2.0, torch.bfloat16)
    expected = (delta * change).sum() / (delta * delta).sum()
    assert stats.retained == pytest.approx(expected.item(), rel=1e-5)
    assert stats.changed == int(torch.count_nonzero(change))
    assert stats.numel == weight.numel()


def test_merged_weight_rounds_once():
    torch.manual_seed(0)
    weight = torch.randn(32, 16).to(torch.bfloat16)
    lora_a, lora_b = torch.randn(4, 16), torch.randn(32, 4) * 1e-2
    merged = merged_lora_weight(weight, lora_a, lora_b, 0.5, torch.bfloat16)
    expected = (weight.float() + (lora_b @ lora_a) * 0.5).to(torch.bfloat16)
    assert torch.equal(merged, expected)
    assert lora_delta(lora_a, lora_b, 0.5).dtype == torch.float32


@pytest.mark.parametrize("base_dtype", [torch.bfloat16, torch.float32])
def test_measure_matches_peft_merge(base_dtype):
    model = _peft(base_dtype, b_std=3e-4)
    stats = measure_lora_merge_retention(model)
    assert stats.modules == 2 and stats.skipped == 0

    expected = MergeRetention()
    for module in (model.base_model.model.q, model.base_model.model.o):
        base = module.base_layer.weight.detach().clone()
        before = base.to(torch.bfloat16).float()
        module.merge()
        served = module.base_layer.weight.detach().to(torch.bfloat16).float()
        module.unmerge()
        expected.update(module.get_delta_weight("default"), served - before)
    assert stats.retained == pytest.approx(expected.retained, rel=1e-5)
    assert stats.changed == expected.changed


def test_measure_skips_merged_disabled_and_merge_aware_modules():
    model = _peft()
    model.base_model.model.q._axolotl_lora_merge_aware_dtype = torch.bfloat16
    stats = measure_lora_merge_retention(model)
    assert stats.modules == 1

    model.base_model.model.o.merge()
    assert measure_lora_merge_retention(model).modules == 0
    model.base_model.model.o.unmerge()

    with model.disable_adapter():
        assert measure_lora_merge_retention(model).modules == 0


def test_measure_skips_tensor_subclass_weights():
    class Sharded(nn.Parameter):
        pass

    model = _peft()
    q = model.base_model.model.q.base_layer
    q.weight = Sharded(q.weight.data, requires_grad=False)
    stats = measure_lora_merge_retention(model)
    assert stats.modules == 1 and stats.skipped == 1


def test_log_warns_only_below_threshold(caplog):
    low = MergeRetention(dot=0.1, delta_sq=1.0, changed=1, numel=10, modules=1)
    high = MergeRetention(dot=0.9, delta_sq=1.0, changed=9, numel=10, modules=1)
    with caplog.at_level(logging.INFO, logger=lora_merge_precision.LOG.name):
        log_merge_retention(low, "ctx", "do X.")
        log_merge_retention(high, "ctx", "do X.")
        log_merge_retention(low, "ctx", "do X.", warn=False)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "10.0%" in warnings[0].getMessage()
    assert "do X." in warnings[0].getMessage()
