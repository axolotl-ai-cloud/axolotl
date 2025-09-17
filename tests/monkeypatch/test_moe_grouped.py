import sys
import types

import torch
import torch.nn as nn

from axolotl.kernels.moe import (
    backends as moe_backends,
    torch_grouped as torch_grouped_module,
)
from axolotl.monkeypatch import moe_grouped


class DummyExperts(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.num_experts = len(layers)

    def __getitem__(self, idx):
        return self.layers[idx]


class DummyQwenMLP(nn.Module):
    def __init__(self, idx: int, hidden: int, intermediate: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden, 2 * intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        nn.init.constant_(self.gate_up_proj.weight, float(idx + 1))
        nn.init.constant_(self.down_proj.weight, float((idx + 1) * 10))


class DummyQwenExpert(nn.Module):
    def __init__(self, idx: int, hidden: int, intermediate: int):
        super().__init__()
        self.mlp = DummyQwenMLP(idx, hidden, intermediate)


def _make_transformers_stub(monkeypatch, block_cls):
    # ensure we start from the original forward for each test
    if block_cls is DummyMixtralBlock:
        DummyMixtralBlock.forward = _DUMMY_MIXTRAL_ORIG_FORWARD

    transformers_mod = types.ModuleType("transformers")
    models_mod = types.ModuleType("transformers.models")
    mixtral_mod = types.ModuleType("transformers.models.mixtral")
    modeling_mixtral = types.ModuleType("transformers.models.mixtral.modeling_mixtral")
    modeling_mixtral.MixtralSparseMoeBlock = block_cls

    transformers_mod.models = models_mod
    models_mod.mixtral = mixtral_mod
    mixtral_mod.modeling_mixtral = modeling_mixtral

    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "transformers.models", models_mod)
    monkeypatch.setitem(sys.modules, "transformers.models.mixtral", mixtral_mod)
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.mixtral.modeling_mixtral",
        modeling_mixtral,
    )


def test_grouped_uses_per_expert_nested_modules(monkeypatch):
    hidden = 4
    intermediate = 2
    num_experts = 2

    experts = DummyExperts(
        [DummyQwenExpert(i, hidden, intermediate) for i in range(num_experts)]
    )

    gate = nn.Linear(hidden, num_experts, bias=False)
    nn.init.zeros_(gate.weight)

    captured = []

    def fake_grouped_mm(As, Bs):
        captured.append([b.detach().clone() for b in Bs])
        return [
            torch.zeros(a.shape[0], b.shape[-1], device=a.device, dtype=a.dtype)
            for a, b in zip(As, Bs, strict=False)
        ]

    monkeypatch.setattr(torch_grouped_module, "_call_grouped_mm", fake_grouped_mm)

    hidden_states = torch.randn(1, 2, hidden)
    y, router_logits = torch_grouped_module.moe_ffn_forward_grouped(
        hidden_states, gate, experts, top_k=2
    )

    assert y is not None
    assert router_logits is not None
    assert captured, "Grouped GEMM path should have been invoked"
    first_call = captured[0]
    expected0 = experts[0].mlp.gate_up_proj.weight.t()
    expected1 = experts[1].mlp.gate_up_proj.weight.t()
    assert torch.equal(first_call[0], expected0)
    assert torch.equal(first_call[1], expected1)
    assert not torch.equal(first_call[0], first_call[1])


class _DummyCfg:
    moe_backend = "torch_grouped"


class DummyMixtralBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.top_k = 1
        self.gate = lambda x: x
        self.experts = object()
        self._calls = []

    def forward(self, hidden_states: torch.Tensor, attention_mask=None):
        self._calls.append((hidden_states, attention_mask))
        tokens = hidden_states.shape[0] * hidden_states.shape[1]
        router = torch.ones(
            tokens, 2, device=hidden_states.device, dtype=hidden_states.dtype
        )
        return hidden_states + 5, router


_DUMMY_MIXTRAL_ORIG_FORWARD = DummyMixtralBlock.forward


def test_apply_grouped_forward_handles_args(monkeypatch):
    _make_transformers_stub(monkeypatch, DummyMixtralBlock)
    import axolotl.common.architectures as arch

    original_map = arch.MOE_ARCH_BLOCK.copy()
    monkeypatch.setitem(arch.MOE_ARCH_BLOCK, "mixtral", "MixtralSparseMoeBlock")
    for key in list(original_map.keys()):
        if key != "mixtral":
            monkeypatch.setitem(arch.MOE_ARCH_BLOCK, key, None)

    monkeypatch.setattr(
        moe_grouped,
        "get_moe_backend_name",
        lambda preferred=None: moe_backends.MOEBackend.TORCH_GROUPED,
    )

    results = {}

    def fake_grouped_forward(hidden_states, gate, experts, top_k):
        results["called"] = True
        router = torch.zeros(
            hidden_states.shape[0] * hidden_states.shape[1],
            2,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return hidden_states + 1, router

    monkeypatch.setattr(torch_grouped_module, "available", lambda: True)
    monkeypatch.setattr(
        torch_grouped_module,
        "moe_ffn_forward_grouped",
        fake_grouped_forward,
    )

    cfg = _DummyCfg()
    moe_grouped.apply_grouped_to_moe_blocks(cfg)

    block = DummyMixtralBlock()
    hidden_states = torch.ones(1, 2, 3)
    mask = torch.zeros(1, 2)
    out, router = block.forward(hidden_states, attention_mask=mask)

    assert results.get("called") is True
    assert torch.equal(out, hidden_states + 1)
    assert router.shape[0] == hidden_states.shape[0] * hidden_states.shape[1]


def test_apply_grouped_forward_fallback(monkeypatch):
    _make_transformers_stub(monkeypatch, DummyMixtralBlock)
    import axolotl.common.architectures as arch

    original_map = arch.MOE_ARCH_BLOCK.copy()
    monkeypatch.setitem(arch.MOE_ARCH_BLOCK, "mixtral", "MixtralSparseMoeBlock")
    for key in list(original_map.keys()):
        if key != "mixtral":
            monkeypatch.setitem(arch.MOE_ARCH_BLOCK, key, None)

    monkeypatch.setattr(
        moe_grouped,
        "get_moe_backend_name",
        lambda preferred=None: moe_backends.MOEBackend.TORCH_GROUPED,
    )
    monkeypatch.setattr(torch_grouped_module, "available", lambda: True)
    monkeypatch.setattr(
        torch_grouped_module,
        "moe_ffn_forward_grouped",
        lambda *args, **kwargs: (None, None),
    )

    cfg = _DummyCfg()
    moe_grouped.apply_grouped_to_moe_blocks(cfg)

    block = DummyMixtralBlock()
    hidden_states = torch.ones(1, 2, 3)
    mask = torch.zeros(1, 2)
    out, router = block.forward(hidden_states, attention_mask=mask)

    assert torch.equal(out, hidden_states + 5)
    assert router.shape[0] == hidden_states.shape[0] * hidden_states.shape[1]
    assert block._calls, "Original forward should have been invoked"
    call_hidden, call_mask = block._calls[-1]
    assert torch.equal(call_hidden, hidden_states)
    assert torch.equal(call_mask, mask)


def test_get_moe_backend_name_prefers_probe(monkeypatch):
    monkeypatch.setattr(moe_backends, "_probe_torch_grouped", lambda: True)
    assert moe_backends.get_moe_backend_name() == moe_backends.MOEBackend.TORCH_GROUPED


def test_get_moe_backend_name_falls_back(monkeypatch):
    warnings_captured = []

    def fake_warn(msg):
        warnings_captured.append(msg)

    monkeypatch.setattr(moe_backends, "_probe_torch_grouped", lambda: False)
    monkeypatch.setattr(moe_backends.warnings, "warn", fake_warn)
    backend = moe_backends.get_moe_backend_name("torch_grouped")
    assert backend == moe_backends.MOEBackend.NAIVE
    assert warnings_captured, "Expected warning when torch_grouped unavailable"
