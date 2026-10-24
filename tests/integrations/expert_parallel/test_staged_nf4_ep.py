"""CPU checks for expert parallelism over the CPU-staged NF4 loader."""

from types import SimpleNamespace

import torch
from torch import nn

from axolotl.integrations.expert_parallel.bnb import (
    _adopt_staged_quant_state,
    expert_storage,
)
from axolotl.loaders.nf4 import _align_parameter_dtypes, _parameter_dtypes
from axolotl.monkeypatch.accelerate.fsdp2_nf4 import (
    _expert_parallel_local_names,
    _restore_local_state,
    expert_parallel_local_state,
)


class BnbNF4Parametrization(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.nested = True
        self.register_buffer("absmax", torch.zeros(8))
        self.register_buffer("nested_absmax", torch.zeros(2))
        self.register_buffer("nested_code", torch.zeros(2))
        self.register_buffer("code", torch.zeros(16))

    @property
    def quant_state(self):
        return SimpleNamespace(shape=self.shape)

    def forward(self, data):
        return data


def _parametrized_experts(shape=(4, 3, 2)):
    module = nn.Module()
    module.gate_up_proj = nn.Parameter(
        torch.zeros(12, 1, dtype=torch.uint8), requires_grad=False
    )
    nn.utils.parametrize.register_parametrization(
        module, "gate_up_proj", BnbNF4Parametrization(shape), unsafe=True
    )
    return module


def test_expert_storage_reads_staged_parametrization():
    module = _parametrized_experts()
    original, shape, parametrization = expert_storage(module, "gate_up_proj")
    assert original.dtype == torch.uint8
    assert shape == (4, 3, 2)
    assert isinstance(parametrization, BnbNF4Parametrization)


def test_adopt_sliced_quant_state_updates_buffers():
    parametrization = BnbNF4Parametrization((4, 3, 2))
    sliced = SimpleNamespace(shape=(1, 3, 2), absmax=torch.ones(2), nested=False)
    _adopt_staged_quant_state(parametrization, sliced)
    assert parametrization.shape == (1, 3, 2)
    assert torch.equal(parametrization.absmax, torch.ones(2))
    assert not parametrization.nested
    assert "nested_absmax" not in dict(parametrization.named_buffers())
    assert "code" in dict(parametrization.named_buffers())


def test_peers_adopt_rank_zero_parameter_dtypes():
    with torch.device("meta"):
        rank_zero = nn.ModuleDict({"gate": nn.Linear(2, 2), "up": nn.Linear(2, 2)})
        peer = nn.ModuleDict({"gate": nn.Linear(2, 2), "up": nn.Linear(2, 2)})
    # buffers are real on the meta build (init_empty_weights keeps them), params are meta
    for model in (rank_zero, peer):
        model["gate"].register_buffer("bias_correction", torch.zeros(2))
    rank_zero.to(torch.bfloat16)
    rank_zero["gate"].float()
    peer.to(torch.bfloat16)
    assert _align_parameter_dtypes(peer, _parameter_dtypes(rank_zero)) == 3
    assert peer["gate"].weight.dtype == torch.float32
    assert peer["gate"].weight.is_meta
    assert peer["gate"].bias_correction.dtype == torch.float32
    assert peer["up"].weight.dtype == torch.bfloat16


def test_expert_parallel_local_names_cover_sliced_experts_only():
    model = nn.Module()
    model.experts = _parametrized_experts()
    model.experts.num_local_experts = 1
    model.experts.num_experts_global = 4
    model.experts.local_expert_offset = 0
    model.dense = nn.Linear(2, 2)
    targets = model.state_dict()
    local = _expert_parallel_local_names(model, targets)
    assert local == {
        "experts.parametrizations.gate_up_proj.original",
        "experts.parametrizations.gate_up_proj.0.absmax",
        "experts.parametrizations.gate_up_proj.0.nested_absmax",
    }
    model.experts.num_local_experts = 4
    assert _expert_parallel_local_names(model, targets) == set()


def test_local_expert_state_survives_the_move_to_meta():
    model = nn.Module()
    model.experts = _parametrized_experts()
    model.experts.num_local_experts = 1
    model.experts.num_experts_global = 4
    model.experts.local_expert_offset = 0
    model.dense = nn.Linear(2, 2)
    kept = expert_parallel_local_state(model)
    assert set(kept) == {
        "experts.parametrizations.gate_up_proj.original",
        "experts.parametrizations.gate_up_proj.0.absmax",
        "experts.parametrizations.gate_up_proj.0.nested_absmax",
    }
    model.to(torch.device("meta"))
    targets = model.state_dict()
    local = _expert_parallel_local_names(model, targets)
    restored = _restore_local_state(targets, local, kept, torch.device("cpu"), False)
    assert all(not value.is_meta for value in restored.values())
    model.load_state_dict({**targets, **restored}, assign=True, strict=True)
    assert not model.experts.parametrizations.gate_up_proj.original.is_meta
    # the shared codebook is rank zero's to broadcast, like any replicated buffer
    assert model.experts.parametrizations.gate_up_proj[0].code.is_meta
    assert model.dense.weight.is_meta


def test_expert_param_leaf_sees_through_parametrizations():
    from axolotl.monkeypatch.accelerate.fsdp2 import _expert_param_leaf

    assert _expert_param_leaf("m.layers.3.mlp.experts.gate_up_proj") == "gate_up_proj"
    assert (
        _expert_param_leaf(
            "m.layers.3.mlp.experts.base_layer.parametrizations.down_proj.original"
        )
        == "down_proj"
    )
    assert (
        _expert_param_leaf("m.layers.3.mlp.experts.lora_A.default.weight") == "weight"
    )
