"""Resuming a routed-expert LoRA on an EP-sliced model cuts the saved (all-experts) adapter to
this rank's experts before PEFT loads it."""

import copy

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model

from axolotl.integrations.expert_parallel.shard import ep_local_adapter_dir
from axolotl.utils.dict import DictDefault

E_GLOBAL, EP_SIZE, HIDDEN, RANK = 8, 2, 6, 2


class _Experts(torch.nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(
            torch.randn(num_experts, 2 * HIDDEN, HIDDEN)
        )
        self.down_proj = torch.nn.Parameter(torch.randn(num_experts, HIDDEN, HIDDEN))


class _Model(torch.nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.q_proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.experts = _Experts(num_experts)


def _lora_config():
    return LoraConfig(
        r=RANK,
        lora_alpha=4,
        target_modules=["q_proj"],
        target_parameters=["experts.gate_up_proj", "experts.down_proj"],
    )


def _ep_sliced(model, ep_rank):
    local = copy.deepcopy(model)
    e_local = E_GLOBAL // EP_SIZE
    start = ep_rank * e_local
    for name in ("gate_up_proj", "down_proj"):
        full = getattr(local.experts, name)
        setattr(
            local.experts,
            name,
            torch.nn.Parameter(full.data[start : start + e_local].clone()),
        )
    local.experts.num_experts_global = E_GLOBAL
    local.experts.num_local_experts = e_local
    local.experts.local_expert_offset = start
    return local


@pytest.fixture
def saved_adapter(tmp_path):
    torch.manual_seed(0)
    base = _Model(E_GLOBAL)
    peft = get_peft_model(copy.deepcopy(base), _lora_config())
    with torch.no_grad():
        for p in peft.parameters():
            if p.requires_grad:
                p.copy_(torch.randn_like(p))
    peft.save_pretrained(str(tmp_path / "adapter"))
    return base, peft, str(tmp_path / "adapter")


def test_unsliced_model_returns_the_directory_untouched(saved_adapter):
    base, _, adapter_dir = saved_adapter
    assert ep_local_adapter_dir(base, adapter_dir) == adapter_dir


@pytest.mark.parametrize("ep_rank", [0, 1])
def test_ep_sliced_model_loads_its_own_experts(saved_adapter, ep_rank):
    base, peft, adapter_dir = saved_adapter
    local = _ep_sliced(base, ep_rank)
    e_local = E_GLOBAL // EP_SIZE
    start = ep_rank * e_local

    with pytest.raises((RuntimeError, ValueError)):
        PeftModel.from_pretrained(copy.deepcopy(local), adapter_dir)

    resumed = PeftModel.from_pretrained(local, ep_local_adapter_dir(local, adapter_dir))

    full = dict(peft.named_parameters())
    got = dict(resumed.named_parameters())
    lora_keys = [k for k in full if "lora_" in k]
    assert lora_keys and set(lora_keys) == {k for k in got if "lora_" in k}
    for key in lora_keys:
        expected = full[key]
        if "experts" not in key:
            torch.testing.assert_close(got[key], expected, msg=key)
        elif "lora_A" in key:  # [E*r, in] expert-major rows
            torch.testing.assert_close(
                got[key], expected[start * RANK : (start + e_local) * RANK], msg=key
            )
        else:  # lora_B [out, r*E]: experts on the last axis of the [out, r, E] view
            out_dim = expected.shape[0]
            view = expected.reshape(out_dim, RANK, E_GLOBAL)[
                :, :, start : start + e_local
            ]
            torch.testing.assert_close(
                got[key], view.reshape(out_dim, RANK * e_local), msg=key
            )


def test_already_local_adapter_is_left_alone(saved_adapter, tmp_path):
    base, _, adapter_dir = saved_adapter
    local = _ep_sliced(base, 1)
    once = ep_local_adapter_dir(local, adapter_dir)
    twice = ep_local_adapter_dir(local, once)
    from safetensors.torch import load_file

    a = load_file(f"{once}/adapter_model.safetensors")
    b = load_file(f"{twice}/adapter_model.safetensors")
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key])


def test_load_lora_removes_the_rank_local_directory(saved_adapter, monkeypatch):
    import os

    from axolotl.integrations.expert_parallel import shard
    from axolotl.loaders import adapter as adapter_module

    base, _, adapter_dir = saved_adapter
    local = _ep_sliced(base, 0)
    created = []
    original = shard.ep_local_adapter_dir

    def tracking(model, directory):
        out = original(model, directory)
        created.append(out)
        return out

    monkeypatch.setattr(shard, "ep_local_adapter_dir", tracking)
    cfg = DictDefault(
        {
            "lora_model_dir": adapter_dir,
            "lora_r": RANK,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "lora_target_modules": ["q_proj"],
            "lora_target_parameters": ["experts.gate_up_proj", "experts.down_proj"],
        }
    )
    model, _ = adapter_module.load_lora(local, cfg)
    assert created and created[0] != adapter_dir
    assert not os.path.exists(created[0])
    assert any("lora_" in name for name, _ in model.named_parameters())
