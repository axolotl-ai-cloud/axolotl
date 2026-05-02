"""Unit tests for FIM-guided automatic LoRA rank allocation."""

import warnings

import pytest
import torch
from torch import nn

from axolotl.utils.fim_rank import (
    _allocate_ranks,
    _get_lora_b_params,
    apply_fim_ranks,
)

# ---------------------------------------------------------------------------
# Minimal model helpers
# ---------------------------------------------------------------------------


class TinyMLP(nn.Module):
    def __init__(self, in_f: int = 16, hidden: int = 8, out_f: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_f, hidden)
        self.fc2 = nn.Linear(hidden, out_f)

    def forward(self, input_ids, labels=None):
        x = torch.relu(self.fc1(input_ids.float()))
        logits = self.fc2(x)
        loss = (
            nn.functional.mse_loss(logits, labels.float())
            if labels is not None
            else None
        )
        return type("Out", (), {"loss": loss, "logits": logits})()


def _make_peft_model(r: int = 4):
    from peft import LoraConfig, get_peft_model

    base = TinyMLP()
    cfg = LoraConfig(r=r, lora_alpha=r * 2, target_modules=["fc1", "fc2"], bias="none")
    return get_peft_model(base, cfg)


def _make_dataloader(n: int = 4, bs: int = 2, in_f: int = 16, out_f: int = 4):
    batches = [
        {
            "input_ids": torch.randn(bs, in_f),
            "labels": torch.randn(bs, out_f),
        }
        for _ in range(n)
    ]
    return batches


# ---------------------------------------------------------------------------
# _allocate_ranks
# ---------------------------------------------------------------------------


class TestAllocateRanks:
    def test_budget_preserved(self):
        imp = {"a": 3.0, "b": 1.0, "c": 1.0}
        ranks = _allocate_ranks(imp, base_r=4, r_min=1, r_max=8)
        assert abs(sum(ranks.values()) - 4 * 3) <= len(ranks)

    def test_higher_importance_gets_higher_rank(self):
        imp = {"high": 10.0, "low": 0.1}
        ranks = _allocate_ranks(imp, base_r=4, r_min=1, r_max=16)
        assert ranks["high"] > ranks["low"]

    def test_r_min_enforced(self):
        imp = {"a": 0.0, "b": 100.0}
        ranks = _allocate_ranks(imp, base_r=4, r_min=2, r_max=8)
        assert ranks["a"] >= 2

    def test_r_max_enforced(self):
        imp = {"a": 1000.0, "b": 0.001}
        ranks = _allocate_ranks(imp, base_r=4, r_min=1, r_max=5)
        assert ranks["a"] <= 5

    def test_empty_returns_empty(self):
        assert _allocate_ranks({}, base_r=4, r_min=1, r_max=8) == {}


# ---------------------------------------------------------------------------
# _get_lora_b_params
# ---------------------------------------------------------------------------


@pytest.mark.requires("peft")
class TestGetLoraBParams:
    def test_finds_params(self):
        model = _make_peft_model(r=4)
        params = _get_lora_b_params(model, "default")
        assert len(params) > 0

    def test_all_require_grad(self):
        model = _make_peft_model(r=4)
        params = _get_lora_b_params(model, "default")
        assert all(p.requires_grad for p in params.values())

    def test_no_params_for_wrong_adapter(self):
        model = _make_peft_model(r=4)
        with warnings.catch_warnings(record=True):
            params = _get_lora_b_params(model, "nonexistent_adapter")
        assert len(params) == 0


# ---------------------------------------------------------------------------
# apply_fim_ranks — end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.requires("peft")
class TestApplyFimRanks:
    def test_runs_without_error(self):
        model = _make_peft_model(r=4)
        dl = _make_dataloader(n=2)
        apply_fim_ranks(model, dl, base_r=4, n_batches=2)

    def test_returns_rank_mapping(self):
        model = _make_peft_model(r=4)
        dl = _make_dataloader(n=2)
        result = apply_fim_ranks(model, dl, base_r=4, n_batches=2)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_all_ranks_in_range(self):
        model = _make_peft_model(r=8)
        dl = _make_dataloader(n=4)
        result = apply_fim_ranks(model, dl, base_r=8, n_batches=4, r_min=1, r_max=16)
        for r in result.values():
            assert 1 <= r <= 16

    def test_adapter_shapes_updated(self):
        from peft.tuners.lora.layer import Linear as LoraLinear

        model = _make_peft_model(r=4)
        dl = _make_dataloader(n=2)
        rank_pattern = apply_fim_ranks(model, dl, base_r=4, n_batches=2)

        for name, module in model.named_modules():
            if isinstance(module, LoraLinear) and name in rank_pattern:
                expected_r = rank_pattern[name]
                assert module.lora_A["default"].weight.shape[0] == expected_r
                assert module.lora_B["default"].weight.shape[1] == expected_r

    def test_empty_dataloader_returns_empty(self):
        model = _make_peft_model(r=4)
        with warnings.catch_warnings(record=True):
            result = apply_fim_ranks(model, [], base_r=4, n_batches=4)
        assert result == {}
