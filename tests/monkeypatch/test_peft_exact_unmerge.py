"""PEFT LoRA merge -> unmerge must restore floating-point base weights exactly."""

import copy
import inspect
from unittest.mock import patch

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Linear as LoraLinear
from torch import nn

from axolotl.monkeypatch.peft.exact_unmerge import patch_peft_lora_exact_unmerge
from axolotl.utils.dict import DictDefault


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(256, 192, bias=False)
        self.o = nn.Linear(192, 256, bias=True)

    def forward(self, x):
        return self.o(self.q(x))


@pytest.fixture(autouse=True)
def _patched():
    patch_peft_lora_exact_unmerge()


def _peft(dtype=torch.bfloat16, seed=0, **lora_kwargs):
    torch.manual_seed(seed)
    base = _Toy()
    with torch.no_grad():
        # small weights next to the delta, where unmerge rounding is most visible
        base.q.weight.mul_(0.05)
    model = get_peft_model(
        base.to(dtype),
        LoraConfig(r=8, lora_alpha=16, target_modules=["q", "o"], **lora_kwargs),
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 2e-3)
    return model


def _base_weights(model):
    return {
        n: p.detach().clone() for n, p in model.named_parameters() if "base_layer" in n
    }


def _step(model, scale=1e-4):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.add_(torch.sign(torch.randn_like(param)) * scale)


def _assert_restored(model, base):
    for name, param in model.named_parameters():
        if name in base:
            assert torch.equal(param, base[name]), name


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_repeated_syncs_leave_base_weights_bitwise_unchanged(dtype):
    model = _peft(dtype)
    base = _base_weights(model)
    for _ in range(20):
        _step(model)
        model.merge_adapter()
        model.unmerge_adapter()
    _assert_restored(model, base)
    assert all(not m.merged for m in model.modules() if isinstance(m, LoraLinear))


def test_unpatched_peft_drifts_in_bf16():
    """Guards that the toy setup exercises the rounding the patch fixes."""
    model = _peft()
    base = _base_weights(model)
    merge, unmerge = (
        inspect.unwrap(LoraLinear.merge),
        inspect.unwrap(LoraLinear.unmerge),
    )
    modules = [m for m in model.modules() if isinstance(m, LoraLinear)]
    for _ in range(20):
        _step(model)
        for module in modules:
            merge(module)
        for module in modules:
            unmerge(module)
    drifted = sum(
        int((p != base[n]).sum()) for n, p in model.named_parameters() if n in base
    )
    assert drifted > 0


def test_merged_weights_are_unchanged():
    model = _peft()
    reference = copy.deepcopy(model)
    model.merge_adapter()
    for module in reference.modules():
        if isinstance(module, LoraLinear):
            inspect.unwrap(LoraLinear.merge)(module)
    for (name, got), (_, expected) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert torch.equal(got, expected), name


@pytest.mark.parametrize("safe_merge", [False, True])
def test_merge_and_unmerge_module_directly(safe_merge):
    model = _peft()
    q = model.base_model.model.q
    before = q.base_layer.weight.detach().clone()
    q.merge(safe_merge=safe_merge)
    assert q.merged
    assert not torch.equal(q.base_layer.weight, before)
    q.unmerge()
    assert torch.equal(q.base_layer.weight, before)


def test_multiple_adapters_merged_together_and_incrementally():
    model = _peft()
    model.add_adapter("other", LoraConfig(r=4, lora_alpha=8, target_modules=["q", "o"]))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B.other" in name:
                param.normal_(0, 2e-3)
    q = model.base_model.model.q
    before = q.base_layer.weight.detach().clone()

    q.merge(adapter_names=["default", "other"])
    q.unmerge()
    assert torch.equal(q.base_layer.weight, before)

    q.merge(adapter_names=["default"])
    q.merge(adapter_names=["other"])
    assert q.merged_adapters == ["default", "other"]
    q.unmerge()
    assert q.merged_adapters == []
    assert torch.equal(q.base_layer.weight, before)


def test_forward_with_disabled_adapter_unmerges_exactly():
    model = _peft()
    base = _base_weights(model)
    model.merge_adapter()
    with model.disable_adapter(), torch.no_grad():
        model(torch.randn(2, 256, dtype=torch.bfloat16))
    _assert_restored(model, base)


def test_dora_keeps_peft_behaviour():
    model = _peft(use_dora=True)
    q = model.base_model.model.q
    q.merge()
    assert q._axolotl_merge_groups == [(1, None)]
    q.unmerge()
    assert not q.merged and q._axolotl_merge_groups == []


def test_unmerge_when_not_merged_still_warns():
    model = _peft()
    with pytest.warns(UserWarning, match="Already unmerged"):
        model.base_model.model.q.unmerge()


def test_patch_is_idempotent():
    merge, unmerge = LoraLinear.merge, LoraLinear.unmerge
    patch_peft_lora_exact_unmerge()
    assert LoraLinear.merge is merge and LoraLinear.unmerge is unmerge


@pytest.mark.parametrize(
    "inference, merge_lora, installed",
    [(False, None, True), (True, None, False), (False, True, False)],
)
def test_load_lora_installs_patch_for_training(inference, merge_lora, installed):
    from axolotl.loaders.adapter import load_lora

    cfg = DictDefault(
        {
            "adapter": "lora",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_target_modules": ["q"],
            "merge_lora": merge_lora,
        }
    )
    with patch(
        "axolotl.monkeypatch.peft.exact_unmerge.patch_peft_lora_exact_unmerge"
    ) as install:
        load_lora(_Toy(), cfg, inference=inference, config_only=True)
    assert install.called is installed
