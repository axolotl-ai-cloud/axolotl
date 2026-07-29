"""Tests for the B200 factored LoRA MLP kernel (axolotl.kernels.blackwell).

Ported from the B200 optimization campaign's regression suite. Correctness
tests (fp32-reference parity, live-gradients canary, scalar-fusion guard,
equivalence vs the standard LoRA_MLP kernel) run on any CUDA device -- the
math is architecture-portable. Launch-budget and perf tests are keyed by
compute capability in blackwell_launch_budgets.json and only run where a
golden exists (sm_100).
"""

import json
import pathlib
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from axolotl.kernels.blackwell.lora_mlp_factored import (
    LoRA_MLP_B200_Factored,
    apply_lora_mlp_swiglu_b200,
)
from axolotl.kernels.blackwell.support import lora_mlp_b200_module_supported
from axolotl.kernels.lora import apply_lora_mlp_swiglu

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.b200,
]

_BUDGETS = json.loads(
    (pathlib.Path(__file__).parent / "blackwell_launch_budgets.json").read_text()
)

# The campaign's certified shape grid (Llama-3-8B MLP, 4096 tokens/micro-batch)
HIDDEN, INTER = 4096, 14336
BATCH, SEQ = 8, 512
RANKS = [8, 16, 32, 64]
RANK = 16


def _device_key():
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def make_params(rank, dtype=torch.bfloat16, seed=0, scale=2.0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape, req=False, std=1.0):
        t = torch.randn(*shape, device="cuda", dtype=dtype, generator=g) * std
        t.requires_grad_(req)
        return t

    p = {}
    p["gate_W"] = rnd(INTER, HIDDEN, std=HIDDEN**-0.5)
    p["up_W"] = rnd(INTER, HIDDEN, std=HIDDEN**-0.5)
    p["down_W"] = rnd(HIDDEN, INTER, std=INTER**-0.5)
    for proj, (fin, fout) in {
        "gate": (HIDDEN, INTER),
        "up": (HIDDEN, INTER),
        "down": (INTER, HIDDEN),
    }.items():
        p[f"{proj}_A"] = rnd(rank, fin, req=True, std=fin**-0.5)
        p[f"{proj}_B"] = rnd(fout, rank, req=True, std=0.5)
        p[f"{proj}_s"] = scale / rank
    return p


def _grad_tensors(p):
    names, tensors = [], []
    for proj in ("gate", "up", "down"):
        for suf in ("A", "B"):
            names.append(f"{proj}_{suf}")
            tensors.append(p[f"{proj}_{suf}"])
    return names, tensors


def frob(a, b):
    return (
        torch.linalg.norm((a.float() - b.float()).flatten())
        / (torch.linalg.norm(b.float().flatten()) + 1e-9)
    ).item()


def _apply_factored(X, p):
    return LoRA_MLP_B200_Factored.apply(
        X,
        p["gate_W"],
        p["gate_A"],
        p["gate_B"],
        p["gate_s"],
        p["up_W"],
        p["up_A"],
        p["up_B"],
        p["up_s"],
        p["down_W"],
        p["down_A"],
        p["down_B"],
        p["down_s"],
    )


def _make_case(rank=RANK):
    torch.manual_seed(0)
    X = torch.randn(
        BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    grad_out = torch.randn(BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.bfloat16)
    p = make_params(rank)
    names, ts = _grad_tensors(p)
    return X, grad_out, p, names, ts


@pytest.mark.parametrize("rank", RANKS)
def test_fp32_reference_parity_factored_math(rank):
    """Frobenius relative error vs an fp32 reference computing the same
    factored algebraic form -- the campaign's standing correctness bar."""
    X, grad_out, p, names, ts = _make_case(rank)

    P = {
        k: (
            v.detach().float().requires_grad_(True)
            if torch.is_tensor(v) and v.requires_grad
            else (v.detach().float() if torch.is_tensor(v) else v)
        )
        for k, v in p.items()
    }
    Xf = X.detach().float().requires_grad_(True)

    def factored_proj(x, W, A, B, s):
        return x @ W.t() + s * ((x @ A.t()) @ B.t())

    e = factored_proj(Xf, P["gate_W"], P["gate_A"], P["gate_B"], P["gate_s"])
    g = factored_proj(Xf, P["up_W"], P["up_A"], P["up_B"], P["up_s"])
    h = e * torch.sigmoid(e) * g
    out_ref = factored_proj(h, P["down_W"], P["down_A"], P["down_B"], P["down_s"])
    ref_ts = [P[n] for n in names]
    grads_ref = torch.autograd.grad(
        out_ref, [Xf] + ref_ts, grad_outputs=grad_out.float()
    )

    out = _apply_factored(X, p)
    grads = torch.autograd.grad(out, [X] + ts, grad_outputs=grad_out)

    tol = 2e-2
    assert frob(out, out_ref) < tol, f"rank={rank}: forward output outside tolerance"
    assert frob(grads[0], grads_ref[0]) < tol, f"rank={rank}: dX outside tolerance"
    for name, g_, g_ref in zip(names, grads[1:], grads_ref[1:], strict=False):
        assert frob(g_, g_ref) < tol, f"rank={rank}: {name} gradient outside tolerance"


def test_live_outputs_canary():
    """Every parameter must actually receive a finite, non-zero gradient --
    guards against dead-code-elimination making a broken kernel look fast."""
    X, grad_out, p, names, ts = _make_case()
    out = _apply_factored(X, p)
    grads = torch.autograd.grad(out, [X] + ts, grad_outputs=grad_out)

    dX = grads[0]
    assert dX is not None and torch.isfinite(dX).all() and dX.abs().sum() > 0, (
        "dX is None/non-finite/all-zero"
    )
    for name, g in zip(names, grads[1:], strict=False):
        assert g is not None, f"{name}'s gradient is None"
        assert torch.isfinite(g).all(), f"{name}'s gradient contains NaN/Inf"
        assert g.abs().sum() > 0, f"{name}'s gradient is all-zero"


def test_no_unfused_scalar_multiply_kernel():
    """The six scalar multiplies must stay fused into their GEMM's addmm
    alpha -- a separate MulFunctor elementwise kernel means a regression."""
    X, grad_out, p, names, ts = _make_case()

    def step():
        out = _apply_factored(X, p)
        torch.autograd.grad(out, [X] + ts, grad_outputs=grad_out, retain_graph=False)

    for _ in range(10):
        step()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()

    events = prof.key_averages()
    names_seen = [
        e.key
        for e in events
        if e.device_type == torch.autograd.DeviceType.CUDA and e.count > 0
    ]
    assert not any("MulFunctor" in name for name in names_seen), (
        "found a separate elementwise-multiply kernel (MulFunctor) -- "
        "scalar-multiply fusion via addmm alpha appears to have regressed"
    )


def test_launch_count_budget():
    """Golden launch-count/kernel-name budget, keyed by compute capability.
    Skips on devices without a certified golden (currently sm_100 only)."""
    device_key = _device_key()
    budget_key = "lora_mlp_factored_fwdbwd_r16_M4096"
    if device_key not in _BUDGETS or budget_key not in _BUDGETS[device_key]:
        pytest.skip(f"no launch budget recorded for {device_key}")
    budget = _BUDGETS[device_key][budget_key]

    X, grad_out, p, names, ts = _make_case()

    def step():
        out = _apply_factored(X, p)
        torch.autograd.grad(out, [X] + ts, grad_outputs=grad_out, retain_graph=False)

    for _ in range(10):
        step()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()

    events = prof.key_averages()
    cuda_events = [
        e
        for e in events
        if e.device_type == torch.autograd.DeviceType.CUDA and e.count > 0
    ]
    total_launches = sum(e.count for e in cuda_events)
    names_seen = [e.key for e in cuda_events]

    if total_launches > budget["launches"]:
        pytest.fail(
            f"LAUNCH COUNT REGRESSION: measured {total_launches}, budget "
            f"{budget['launches']} (locked {budget['locked']}). See "
            f"blackwell_launch_budgets.json _meta.how_to_update."
        )
    if total_launches < budget["launches"]:
        pytest.fail(
            f"BUDGET CAN BE LOWERED: measured {total_launches}, budget "
            f"{budget['launches']} (locked {budget['locked']}) -- re-certify "
            f"deliberately per blackwell_launch_budgets.json _meta.how_to_update."
        )
    for required in budget["required_kernel_substrings"]:
        assert any(required in name for name in names_seen), (
            f"required kernel '{required}' not found -- kernel set may have drifted"
        )
    for forbidden in budget["forbidden_kernel_substrings"]:
        assert not any(forbidden in name for name in names_seen), (
            f"forbidden kernel '{forbidden}' found -- "
            f"{budget.get('forbidden_kernel_note', '')}"
        )


class MockLoRAProj(nn.Module):
    """Minimal PEFT-lora-shaped projection for get_lora_parameters."""

    def __init__(self, in_features, out_features, rank, dtype=torch.bfloat16):
        super().__init__()
        self.base_layer = nn.Linear(
            in_features, out_features, bias=False, dtype=dtype, device="cuda"
        )
        self.lora_A = nn.ModuleDict(
            {
                "default": nn.Linear(
                    in_features, rank, bias=False, dtype=dtype, device="cuda"
                )
            }
        )
        self.lora_B = nn.ModuleDict(
            {
                "default": nn.Linear(
                    rank, out_features, bias=False, dtype=dtype, device="cuda"
                )
            }
        )
        self.scaling = {"default": 2.0 / rank}
        self.active_adapter = "default"
        self.disable_adapters = False
        self.merged = False
        self.lora_dropout = nn.ModuleDict({"default": nn.Identity()})


class MockMLP(nn.Module):
    """gate/up/down container matching what the MLP patch binds against."""

    def __init__(self, hidden=512, inter=1024, rank=8):
        super().__init__()
        self.gate_proj = MockLoRAProj(hidden, inter, rank)
        self.up_proj = MockLoRAProj(hidden, inter, rank)
        self.down_proj = MockLoRAProj(inter, hidden, rank)


def test_matches_standard_lora_mlp_kernel():
    """The B200 wrapper and axolotl's standard LoRA_MLP path must agree on
    the same weights within the campaign's fp32-parity bar."""
    torch.manual_seed(0)
    mlp = MockMLP()
    X1 = torch.randn(
        2, 128, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    X2 = X1.detach().clone().requires_grad_(True)

    supported, reason = lora_mlp_b200_module_supported(
        mlp.gate_proj, mlp.up_proj, mlp.down_proj
    )
    assert supported, reason

    # inplace=False: both paths' backward would otherwise write dX into their
    # own X storage, which is fine in training but makes comparison awkward
    out_std = apply_lora_mlp_swiglu(mlp, X1, inplace=False)
    out_b200 = apply_lora_mlp_swiglu_b200(mlp, X2, inplace=False)
    assert frob(out_b200, out_std) < 2e-2

    grad_out = torch.randn_like(out_std)
    lora_params = [
        proj.lora_A["default"].weight
        for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    ] + [
        proj.lora_B["default"].weight
        for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    ]
    grads_std = torch.autograd.grad(
        out_std, [X1] + lora_params, grad_outputs=grad_out, retain_graph=False
    )
    grads_b200 = torch.autograd.grad(
        out_b200, [X2] + lora_params, grad_outputs=grad_out, retain_graph=False
    )
    for g_std, g_b200 in zip(grads_std, grads_b200, strict=True):
        assert frob(g_b200, g_std) < 2e-2


def test_wrapper_falls_back_when_module_unsupported():
    """A post-patch state change (here: active dropout) must degrade to the
    standard kernel, not produce silently wrong numbers."""
    torch.manual_seed(0)
    mlp = MockMLP()
    mlp.gate_proj.lora_dropout = nn.ModuleDict({"default": nn.Dropout(0.5)})
    mlp.train()
    X = torch.randn(
        2, 128, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    with patch(
        "axolotl.kernels.lora.apply_lora_mlp_swiglu",
        wraps=apply_lora_mlp_swiglu,
    ) as spy:
        apply_lora_mlp_swiglu_b200(mlp, X)
    assert spy.called, "expected fallback to the standard LoRA MLP kernel"


def test_module_supported_rejects_non_bf16():
    mlp = MockMLP()
    mlp.gate_proj.base_layer.weight.data = mlp.gate_proj.base_layer.weight.data.float()
    supported, reason = lora_mlp_b200_module_supported(
        mlp.gate_proj, mlp.up_proj, mlp.down_proj
    )
    assert not supported
    assert "bfloat16" in reason


def _tiny_peft_llama():
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=512,
    )
    model = LlamaForCausalLM(config)
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["gate_proj", "up_proj", "down_proj"],
        ),
    )
    return peft_model.to(device="cuda", dtype=torch.bfloat16)


def _patch_and_get_mlp_forward(cfg_overrides=None):
    from axolotl.monkeypatch.lora_kernels import (
        apply_lora_kernel_patches,
        get_layers,
    )
    from axolotl.utils.dict import DictDefault

    cfg = DictDefault(
        {
            "base_model": "tiny-llama",
            "lora_mlp_kernel": True,
            "lora_mlp_kernel_b200": True,
            **(cfg_overrides or {}),
        }
    )
    model = _tiny_peft_llama()
    apply_lora_kernel_patches(model, cfg)
    return get_layers(model)[0].mlp.forward


@pytest.mark.skipif(
    torch.cuda.get_device_capability() == (10, 0), reason="requires a non-B200 device"
)
def test_patching_falls_back_on_non_sm100():
    """G2: with the flag on but on a non-sm_100 device, the patcher must
    select the standard kernel -- no exception, no silent wrong path."""
    forward = _patch_and_get_mlp_forward()
    assert forward.__func__ is apply_lora_mlp_swiglu


def test_patching_selects_b200_on_sm100(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    forward = _patch_and_get_mlp_forward()
    assert forward.__func__ is apply_lora_mlp_swiglu_b200


def test_patching_respects_flag_off(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    forward = _patch_and_get_mlp_forward({"lora_mlp_kernel_b200": None})
    assert forward.__func__ is apply_lora_mlp_swiglu


def test_perf_vs_axolotl_lora_mlp():
    """B200-only relative perf gate vs axolotl's standard LoRA_MLP kernel.

    Skips (reporting the measured ratio) until a threshold is certified in
    blackwell_launch_budgets.json -- the campaign never certified against
    axolotl's own path, and fabricating a threshold would make this gate
    flaky or toothless.
    """
    device_key = _device_key()
    cert = _BUDGETS["perf_vs_axolotl_lora_mlp"].get(device_key)
    if cert is None:
        pytest.skip(f"no perf golden structure for {device_key}")

    torch.manual_seed(0)
    mlp = MockMLP(hidden=HIDDEN, inter=INTER, rank=RANK)
    X = torch.randn(
        BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    grad_out = torch.randn(BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.bfloat16)
    lora_params = [
        proj.lora_A["default"].weight
        for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    ] + [
        proj.lora_B["default"].weight
        for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    ]

    def std_step(K=8):
        for _ in range(K):
            out = apply_lora_mlp_swiglu(mlp, X, inplace=False)
            torch.autograd.grad(out, [X] + lora_params, grad_outputs=grad_out)

    def b200_step(K=8):
        for _ in range(K):
            out = apply_lora_mlp_swiglu_b200(mlp, X, inplace=False)
            torch.autograd.grad(out, [X] + lora_params, grad_outputs=grad_out)

    # Interleaved warmup + interleaved min-of-3: sequential measurement lets
    # clock/thermal drift bias one side.
    for _ in range(3):
        std_step()
        b200_step()
    torch.cuda.synchronize()

    std_samples, b200_samples = [], []
    for _ in range(3):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        std_step()
        e1.record()
        torch.cuda.synchronize()
        std_samples.append(e0.elapsed_time(e1))
        e2 = torch.cuda.Event(enable_timing=True)
        e3 = torch.cuda.Event(enable_timing=True)
        e2.record()
        b200_step()
        e3.record()
        torch.cuda.synchronize()
        b200_samples.append(e2.elapsed_time(e3))
    ratio = min(std_samples) / min(b200_samples)

    threshold = cert.get("threshold_min_ratio")
    if threshold is None:
        pytest.skip(
            f"perf threshold not yet certified for {device_key}; measured "
            f"ratio standard/b200 = {ratio:.4f} -- see "
            f"blackwell_launch_budgets.json perf_vs_axolotl_lora_mlp.note"
        )
    assert ratio >= threshold, (
        f"PERF REGRESSION vs standard LoRA_MLP: measured ratio={ratio:.4f}, "
        f"below certified threshold {threshold:.4f}"
    )
