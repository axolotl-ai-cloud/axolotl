# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0
"""
Single-GPU correctness tests for the TiledMLP autograd function under both
dense and MoE block forwards. Synthetic-shape modules only; no real
transformers checkpoints are loaded.
"""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def _requires_cuda():
    return pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )


pytestmark = _requires_cuda()

DEVICE = "cuda"


# ────────────────────────────── Helpers ──────────────────────────────


class TinyDenseMLP(nn.Module):
    """LlamaMLP-shape: gate * up -> down, no bias, silu activation."""

    def __init__(self, hidden, intermediate, dtype=torch.float32):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, dtype=dtype)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TinyMoEBlock(nn.Module):
    """Hand-rolled MoE: top-k softmax router + per-expert SwiGLU MLPs.

    Stays intentionally simple so the test exercises sharding semantics
    without dragging in transformers, peft, or kernel libs.
    """

    def __init__(self, hidden, intermediate, num_experts, top_k, dtype=torch.float32):
        super().__init__()
        self.hidden = hidden
        self.intermediate = intermediate
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden, num_experts, bias=False, dtype=dtype)
        # Per-expert SwiGLU weights packed as 3D tensors.
        self.gate_proj = nn.Parameter(
            torch.randn(num_experts, hidden, intermediate, dtype=dtype) * 0.02
        )
        self.up_proj = nn.Parameter(
            torch.randn(num_experts, hidden, intermediate, dtype=dtype) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, intermediate, hidden, dtype=dtype) * 0.02
        )

    def forward(self, x):
        bsz, seq, h = x.shape
        flat = x.reshape(-1, h)
        logits = self.gate(flat)
        weights = F.softmax(logits, dim=-1, dtype=torch.float32)
        top_w, top_i = torch.topk(weights, self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        top_w = top_w.to(flat.dtype)

        out = torch.zeros_like(flat)
        for e in range(self.num_experts):
            mask = top_i == e
            if not mask.any():
                continue
            # tokens routed to expert e (with their per-slot weight)
            token_rows, slot_idx = mask.nonzero(as_tuple=True)
            xe = flat[token_rows]
            we = top_w[token_rows, slot_idx].unsqueeze(-1)
            gate = xe @ self.gate_proj[e]
            up = xe @ self.up_proj[e]
            h_e = F.silu(gate) * up
            ye = h_e @ self.down_proj[e]
            out.index_add_(0, token_rows, we * ye)
        return out.reshape(bsz, seq, h)


def _clone_module(mod):
    """Deep copy + detach + re-attach to autograd to compare two runs."""
    cloned = copy.deepcopy(mod)
    return cloned


def _grad_dict(mod):
    return {
        n: p.grad.detach().clone()
        for n, p in mod.named_parameters()
        if p.grad is not None
    }


def _run_untiled(mod, x):
    x = x.clone().detach().requires_grad_(True)
    y = mod(x)
    g = torch.randn_like(y)
    y.backward(g)
    return y.detach().clone(), x.grad.detach().clone(), _grad_dict(mod), g


def _run_tiled(mod, x, upstream_grad, shards):
    """Re-run forward+backward but routed through TiledMLP."""
    from axolotl.monkeypatch.tiled_mlp.base import TiledMLP

    # Re-fetch fn that takes (self, x) — matches what the patcher passes.
    forward_fn = type(mod).forward

    x = x.clone().detach().requires_grad_(True)
    compute_params = [p for p in mod.parameters() if p.requires_grad]
    y = TiledMLP.apply(forward_fn, mod, x, shards, compute_params)
    if isinstance(y, tuple):  # MoE block forwards may return tuples
        y = y[0]
    y.backward(upstream_grad)
    return y.detach().clone(), x.grad.detach().clone(), _grad_dict(mod)


# ────────────────────────────── Dense parity ──────────────────────────────


def test_tiled_dense_mlp_parity_fp32():
    """Dense LlamaMLP-shape: tiled vs un-tiled must match closely."""
    torch.manual_seed(0)
    hidden, intermediate, seq = 64, 128, 64
    mlp_ref = TinyDenseMLP(hidden, intermediate).to(DEVICE)
    mlp_tile = _clone_module(mlp_ref)

    # ``TiledMLP``'s backward narrows into a flattened ``x_grad`` buffer
    # using offsets along dim 1 only — sequence-packed inputs (batch=1)
    # are the supported shape; multi-batch tensors aren't contiguous in
    # the way the narrow assumes. Production inputs from transformers
    # are batch=1 after sequence packing, so this matches reality.
    x = torch.randn(1, seq, hidden, device=DEVICE)
    y_ref, dx_ref, gp_ref, upstream = _run_untiled(mlp_ref, x)
    y_tile, dx_tile, gp_tile = _run_tiled(mlp_tile, x, upstream, shards=4)

    # FMA reordering across shards introduces sub-eps noise in fp32; allow
    # a small tolerance on outputs and grads.
    assert torch.allclose(y_ref, y_tile, atol=1e-5, rtol=1e-5), (
        f"forward mismatch max={((y_ref - y_tile).abs().max()).item()}"
    )
    assert torch.allclose(dx_ref, dx_tile, atol=1e-5, rtol=1e-5), (
        f"dX mismatch max={((dx_ref - dx_tile).abs().max()).item()}"
    )
    for name in gp_ref:
        diff = (gp_ref[name] - gp_tile[name]).abs().max().item()
        assert diff < 1e-5, f"param-grad mismatch {name}: max={diff}"


# ────────────────────────────── MoE parity ──────────────────────────────


def test_tiled_moe_block_parity_fp32():
    """Hand-rolled MoE block: tiled vs un-tiled fp32 parity."""
    torch.manual_seed(1)
    hidden, intermediate, seq = 64, 128, 64
    moe_ref = TinyMoEBlock(hidden, intermediate, num_experts=8, top_k=2).to(DEVICE)
    moe_tile = _clone_module(moe_ref)

    x = torch.randn(1, seq, hidden, device=DEVICE)
    y_ref, dx_ref, gp_ref, upstream = _run_untiled(moe_ref, x)
    y_tile, dx_tile, gp_tile = _run_tiled(moe_tile, x, upstream, shards=4)

    # The MoE forward involves index_add and routing, which is not
    # numerically deterministic across different batch sizes for fp32 in
    # all setups — but at the synthetic small scale we expect tight match.
    assert torch.allclose(y_ref, y_tile, atol=1e-5, rtol=1e-5), (
        f"MoE forward mismatch max={((y_ref - y_tile).abs().max()).item()}"
    )
    assert torch.allclose(dx_ref, dx_tile, atol=1e-5, rtol=1e-5), (
        f"MoE dX mismatch max={((dx_ref - dx_tile).abs().max()).item()}"
    )
    for name in gp_ref:
        diff = (gp_ref[name] - gp_tile[name]).abs().max().item()
        assert diff < 1e-5, f"MoE param-grad mismatch {name}: max={diff}"


# ─────────────────────── scattermoe-lora + tiled parity ───────────────────


def _build_scattermoe_block(hidden, intermediate, num_experts, top_k, dtype, device):
    """Build a minimal :class:`ScatterMoEGatedMLP`-compatible module.

    Attributes are populated to match what :meth:`ScatterMoEGatedMLP.forward`
    expects (``router``, ``input_linear``, ``output_linear``, ``activation``).
    """
    try:
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            ScatterMoEGatedMLP,
        )
    except ImportError:
        pytest.skip("scattermoe_lora kernels not available")

    block = ScatterMoEGatedMLP()
    # router: a Linear-shaped router with top_k / num_experts attrs.
    router = SimpleNamespace()
    router.layer = nn.Linear(hidden, num_experts, bias=False, dtype=dtype).to(device)
    router.top_k = top_k
    router.num_experts = num_experts
    block.router = router
    # input_linear and output_linear store 3D weights: [E, *, *].
    in_weight = nn.Parameter(
        torch.randn(num_experts, 2 * intermediate, hidden, dtype=dtype, device=device)
        * 0.02
    )
    out_weight = nn.Parameter(
        torch.randn(num_experts, hidden, intermediate, dtype=dtype, device=device)
        * 0.02
    )
    block.input_linear = nn.Module()
    block.input_linear.weight = in_weight
    block.output_linear = nn.Module()
    block.output_linear.weight = out_weight
    block.activation = nn.SiLU()
    # Register the params so .parameters() walks them.
    block.input_linear.register_parameter("weight", in_weight)
    block.output_linear.register_parameter("weight", out_weight)
    return block


def test_tiled_scattermoe_gated_mlp_parity():
    """ScatterMoEGatedMLP: tiled vs un-tiled fwd+bwd parity in bf16.

    Uses the same tolerance scale as ``tests/integrations/test_scattermoe_lora_kernels.py``
    (norm-relative error < 1% for weight grads is the established bar there
    given the bf16 + tiled reduction order differences).
    """
    pytest.importorskip("triton")
    torch.manual_seed(2)
    hidden, intermediate = 64, 128
    num_experts, top_k = 8, 2
    seq = 64
    dtype = torch.bfloat16

    block_ref = _build_scattermoe_block(
        hidden, intermediate, num_experts, top_k, dtype, DEVICE
    )
    block_tile = copy.deepcopy(block_ref)

    x = torch.randn(1, seq, hidden, device=DEVICE, dtype=dtype)
    y_ref, dx_ref, gp_ref, upstream = _run_untiled(block_ref, x)
    y_tile, dx_tile, gp_tile = _run_tiled(block_tile, x, upstream, shards=4)

    def _rel(a, b):
        return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-6)).item()

    assert _rel(y_tile, y_ref) < 1e-2, (
        f"scattermoe forward rel_err={_rel(y_tile, y_ref)}"
    )
    assert _rel(dx_tile, dx_ref) < 1e-2, (
        f"scattermoe dX rel_err={_rel(dx_tile, dx_ref)}"
    )
    for name in gp_ref:
        if name not in gp_tile:
            continue
        rel = _rel(gp_tile[name], gp_ref[name])
        assert rel < 1e-2, f"scattermoe param-grad {name} rel_err={rel}"


# ─────────────────── Patcher: MoE block discovery & dispatch ────────────────


def test_resolve_moe_block_cls_picks_first_available():
    """The patcher walks the suffix list in order; the first hit wins."""
    from axolotl.monkeypatch.tiled_mlp.patch import _resolve_moe_block_cls

    module = SimpleNamespace(
        FooMoE=object,
        FooMoeMLP=object,  # would also match but later in the list
    )
    cls = _resolve_moe_block_cls(module, "Foo")
    assert cls is module.FooMoeMLP, "MoeMLP should be preferred over MoE"


def test_resolve_moe_block_cls_returns_none_for_dense_model():
    from axolotl.monkeypatch.tiled_mlp.patch import _resolve_moe_block_cls

    module = SimpleNamespace(FooMLP=object)
    assert _resolve_moe_block_cls(module, "Foo") is None


# ─────────────── Grad parity under non-uniform per-token loss weights ─────────
#
# Sequence-dim sharding makes per-shard parameter-grads additive: the full
# batch gradient is the SUM of shard contributions, not the mean. If the
# tiled backward ever scaled by ``1/total_shards`` (the historical
# ``GradientAccumulator.gradient_scale``), per-shard non-uniform weights
# would make the mean visibly diverge from the un-tiled reference — uniform
# loss weights can mask the bug because the per-shard means happen to add
# up to a related-magnitude value. These tests exercise multiple
# ``shards ∈ {1, 2, 4}`` with deliberately non-uniform per-token weights
# so a regression in the scaling semantics fails loudly.


def _run_untiled_with_upstream(mod, x, upstream):
    """Un-tiled fwd+bwd given a fixed upstream grad."""
    x = x.clone().detach().requires_grad_(True)
    y = mod(x)
    y.backward(upstream)
    return y.detach().clone(), x.grad.detach().clone(), _grad_dict(mod)


@pytest.mark.parametrize("shards", [1, 2, 4])
def test_tiled_dense_mlp_grad_parity_nonuniform_weights(shards):
    """Dense MLP: tiled vs un-tiled grad parity with non-uniform per-token weights.

    The upstream grad's magnitude varies per token (non-uniform loss weights),
    so each shard contributes a distinct fraction of the total parameter grad.
    A mean-vs-sum bug shows up as a ``shards``-dependent scaling error.
    """
    torch.manual_seed(100 + shards)
    hidden, intermediate = 64, 128
    seq = 128
    mlp_ref = TinyDenseMLP(hidden, intermediate).to(DEVICE)
    mlp_tile = _clone_module(mlp_ref)

    x = torch.randn(1, seq, hidden, device=DEVICE)
    # Non-uniform per-token weights make per-shard grad contributions
    # distinct, exposing any incorrect averaging.
    per_token_w = torch.linspace(0.1, 3.0, seq, device=DEVICE).view(1, seq, 1)
    upstream = torch.randn(1, seq, hidden, device=DEVICE) * per_token_w

    y_ref, dx_ref, gp_ref = _run_untiled_with_upstream(mlp_ref, x, upstream)
    y_tile, dx_tile, gp_tile = _run_tiled(mlp_tile, x, upstream, shards=shards)

    assert torch.allclose(y_ref, y_tile, atol=1e-5, rtol=1e-5), (
        f"shards={shards}: forward mismatch max={((y_ref - y_tile).abs().max()).item()}"
    )
    assert torch.allclose(dx_ref, dx_tile, atol=1e-5, rtol=1e-5), (
        f"shards={shards}: dX mismatch max={((dx_ref - dx_tile).abs().max()).item()}"
    )
    for name in gp_ref:
        diff = (gp_ref[name] - gp_tile[name]).abs().max().item()
        ref_norm = gp_ref[name].abs().max().item() + 1e-8
        # Tight bound in fp32; rel error must be tiny so a 1/N or N
        # scaling error (which would give 25%-400% relative drift) is
        # impossible to miss.
        assert diff / ref_norm < 1e-4, (
            f"shards={shards}: param-grad {name} rel_err={diff / ref_norm}"
        )


@pytest.mark.parametrize("shards", [1, 2, 4])
def test_tiled_moe_grad_parity_nonuniform_weights(shards):
    """MoE block: tiled vs un-tiled grad parity with non-uniform per-token weights."""
    torch.manual_seed(200 + shards)
    hidden, intermediate = 64, 128
    seq = 128
    moe_ref = TinyMoEBlock(hidden, intermediate, num_experts=8, top_k=2).to(DEVICE)
    moe_tile = _clone_module(moe_ref)

    x = torch.randn(1, seq, hidden, device=DEVICE)
    per_token_w = torch.linspace(0.1, 3.0, seq, device=DEVICE).view(1, seq, 1)
    upstream = torch.randn(1, seq, hidden, device=DEVICE) * per_token_w

    y_ref, dx_ref, gp_ref = _run_untiled_with_upstream(moe_ref, x, upstream)
    y_tile, dx_tile, gp_tile = _run_tiled(moe_tile, x, upstream, shards=shards)

    assert torch.allclose(y_ref, y_tile, atol=1e-5, rtol=1e-5), (
        f"shards={shards}: MoE forward mismatch "
        f"max={((y_ref - y_tile).abs().max()).item()}"
    )
    assert torch.allclose(dx_ref, dx_tile, atol=1e-5, rtol=1e-5), (
        f"shards={shards}: MoE dX mismatch "
        f"max={((dx_ref - dx_tile).abs().max()).item()}"
    )
    for name in gp_ref:
        diff = (gp_ref[name] - gp_tile[name]).abs().max().item()
        ref_norm = gp_ref[name].abs().max().item() + 1e-8
        assert diff / ref_norm < 1e-4, (
            f"shards={shards}: MoE param-grad {name} rel_err={diff / ref_norm}"
        )


@pytest.mark.parametrize("shards", [1, 2, 4])
def test_tiled_dense_mlp_grad_parity_bf16(shards):
    """Dense MLP: bf16 grad parity at the param dtype (no fp32 accumulator).

    Guards the default param-dtype accumulator path against regression.
    bf16 reduction order across shards means we use a relative tolerance
    rather than bitwise equality.
    """
    torch.manual_seed(300 + shards)
    hidden, intermediate = 64, 128
    seq = 128
    dtype = torch.bfloat16
    mlp_ref = TinyDenseMLP(hidden, intermediate, dtype=dtype).to(DEVICE)
    mlp_tile = _clone_module(mlp_ref)

    x = torch.randn(1, seq, hidden, device=DEVICE, dtype=dtype)
    per_token_w = torch.linspace(0.1, 3.0, seq, device=DEVICE).view(1, seq, 1).to(dtype)
    upstream = (torch.randn(1, seq, hidden, device=DEVICE) * per_token_w).to(dtype)

    y_ref, dx_ref, gp_ref = _run_untiled_with_upstream(mlp_ref, x, upstream)
    y_tile, dx_tile, gp_tile = _run_tiled(mlp_tile, x, upstream, shards=shards)

    def _rel(a, b):
        return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-6)).item()

    assert _rel(y_tile, y_ref) < 5e-3, (
        f"shards={shards}: bf16 forward rel_err={_rel(y_tile, y_ref)}"
    )
    assert _rel(dx_tile, dx_ref) < 5e-3, (
        f"shards={shards}: bf16 dX rel_err={_rel(dx_tile, dx_ref)}"
    )
    for name in gp_ref:
        rel = _rel(gp_tile[name], gp_ref[name])
        # Tight bound — a 1/N scaling bug would put rel_err ≈ (N-1)/N,
        # which is far above this threshold for any N ≥ 2.
        assert rel < 5e-3, f"shards={shards}: bf16 param-grad {name} rel_err={rel}"


def test_tiled_grad_accumulator_dtype_matches_param_dtype():
    """Regression guard: TiledMLP backward should accumulate at param dtype
    by default (not fp32), so the on-the-fly accumulator does not double
    the parameter-side memory footprint in bf16 training.

    We snapshot the accumulator dtype by patching ``torch.zeros_like`` to
    record the dtype of zero-tensors allocated for compute params during
    backward. The assertion is that none of those allocations request
    fp32 when the params are bf16.
    """
    from axolotl.monkeypatch.tiled_mlp.base import TiledMLP

    torch.manual_seed(42)
    dtype = torch.bfloat16
    mlp = TinyDenseMLP(64, 128, dtype=dtype).to(DEVICE)
    compute_params = [p for p in mlp.parameters() if p.requires_grad]
    param_ids = {id(p) for p in compute_params}

    x = torch.randn(1, 64, 64, device=DEVICE, dtype=dtype)
    upstream = torch.randn(1, 64, 64, device=DEVICE, dtype=dtype)

    real_zeros_like = torch.zeros_like
    allocated_dtypes: list[torch.dtype] = []

    def spy_zeros_like(t, *args, **kwargs):
        # Only record allocations whose shape matches one of the
        # compute params (the accumulator buffers we care about).
        if id(t) in param_ids:
            allocated_dtypes.append(kwargs.get("dtype", t.dtype))
        return real_zeros_like(t, *args, **kwargs)

    x_req = x.clone().detach().requires_grad_(True)
    torch.zeros_like = spy_zeros_like
    try:
        y = TiledMLP.apply(type(mlp).forward, mlp, x_req, 4, compute_params)
        y.backward(upstream)
    finally:
        torch.zeros_like = real_zeros_like

    assert allocated_dtypes, "expected accumulator allocations to be observed"
    # Default path must NOT pre-allocate fp32 buffers when params are bf16.
    assert all(d == dtype for d in allocated_dtypes), (
        f"expected accumulator dtype == {dtype}, got {allocated_dtypes}"
    )
