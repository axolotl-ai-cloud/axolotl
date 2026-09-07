# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0
"""
FSDP2 + TiledMLP multi-rank correctness tests.

Parity guard for the tiled MLP under FSDP2: tiled forward+backward
produces gradients within bf16 tolerance of the un-tiled FSDP2
reference. The companion fix in
``axolotl.monkeypatch.tiled_mlp.base._defer_fsdp2_reshard`` is a
defensive measure that wraps the tile loop in
``FSDPModule.set_reshard_after_backward(False)`` — under the most
common setups (FSDP2 wraps the decoder layer; the post-backward
RegisterPostBackwardFunction fires only when the outer backward
reaches the layer's input, not mid-tile) the reshard does not fire
inside the tile loop, but the helper protects against setups where
the tile loop would otherwise race with FSDP2's per-module reshard.

Run with::

    torchrun --nproc-per-node=2 -m pytest tests/e2e/multigpu/test_tiled_mlp_fsdp2.py

On a 1-GPU executor the tests skip with a clear reason.
"""

import copy
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

_TORCHRUN_LOCAL_RANK = os.environ.get("LOCAL_RANK")
_TORCHRUN_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    ),
    pytest.mark.skipif(
        torch.cuda.device_count() < 2,
        reason="Need >=2 GPUs for FSDP2 multi-rank tests",
    ),
    pytest.mark.skipif(
        _TORCHRUN_LOCAL_RANK is None or _TORCHRUN_WORLD_SIZE < 2,
        reason=(
            "Multi-rank tests must be launched via "
            "`torchrun --nproc-per-node=2 -m pytest <file>`"
        ),
    ),
]


# ──────────────────────────── Process group ──────────────────────────────


@pytest.fixture(scope="module")
def dist_pg():
    """Initialize the default process group exactly once per worker."""
    if not dist.is_initialized():
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dist.init_process_group(backend="nccl")
    yield
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ──────────────────────────── Helpers ────────────────────────────────────


class TinyDenseMLP(nn.Module):
    def __init__(self, hidden, intermediate, dtype=torch.bfloat16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, dtype=dtype)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def _full_state(mod):
    """Materialize an FSDP2 module to full (unsharded) parameters."""
    from torch.distributed.tensor import DTensor

    out = {}
    for name, p in mod.named_parameters():
        if isinstance(p, DTensor):
            out[name] = p.full_tensor().detach().clone()
        else:
            out[name] = p.detach().clone()
    return out


def _full_grads(mod):
    """Gather param grads to full (unsharded) tensors."""
    from torch.distributed.tensor import DTensor

    out = {}
    for name, p in mod.named_parameters():
        if p.grad is None:
            continue
        if isinstance(p.grad, DTensor):
            out[name] = p.grad.full_tensor().detach().clone()
        else:
            out[name] = p.grad.detach().clone()
    return out


def _make_seeded_mlp(hidden, intermediate, dtype, device):
    torch.manual_seed(42)
    mlp = TinyDenseMLP(hidden, intermediate, dtype=dtype).to(device)
    return mlp


# ─────────────────────────── Dense regression guard ──────────────────────


def _install_tiled_forward(module, shards):
    """Bind a tiled-MLP forward at the instance level.

    Mirrors what the patcher does in production: the FSDPModule's
    ``__call__`` triggers FSDP2's pre-forward hooks (which unshard
    parameters) before ``forward`` runs. By going through the wrapped
    module's ``__call__`` rather than calling ``TiledMLP.apply`` directly
    on sharded DTensor params, the tiling and FSDP2's parameter
    materialization compose correctly.
    """
    from types import MethodType

    from axolotl.monkeypatch.tiled_mlp.base import TiledMLP

    original_forward = type(module).forward
    module._compute_params = []  # type: ignore[attr-defined]

    def tiled_forward(self, x):
        if not self._compute_params:
            self._compute_params = [p for p in self.parameters() if p.requires_grad]
        return TiledMLP.apply(
            original_forward,
            self,
            x,
            shards,
            self._compute_params,
        )

    module.forward = MethodType(tiled_forward, module)


def test_fsdp2_tiled_dense_mlp_parity(dist_pg):
    """FSDP2 + tiled MLP must match FSDP2 + un-tiled MLP within bf16 tolerance.

    Wraps the MLP with ``fully_shard`` so it is itself an ``FSDPModule``
    — this is the scenario most likely to hit the post-backward race
    that ``_defer_fsdp2_reshard`` protects against (per-tile inner
    backwards would otherwise fire the FSDPModule's post-backward hook
    mid-loop). The test passes whether or not the helper is in place
    on the current PyTorch (2.11) release; treat it as a parity guard
    that will catch breakage if FSDP2 ever shortens its reshard timing.
    """
    from torch.distributed.fsdp import fully_shard

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    hidden, intermediate = 64, 128
    seq = 64
    dtype = torch.bfloat16

    # Two identical MLPs — one wrapped with FSDP2 only, one wrapped with
    # FSDP2 *and* run through TiledMLP. Same initial weights.
    mlp_ref = _make_seeded_mlp(hidden, intermediate, dtype, device)
    mlp_tile = copy.deepcopy(mlp_ref)
    fully_shard(mlp_ref)
    fully_shard(mlp_tile)
    _install_tiled_forward(mlp_tile, shards=4)

    torch.manual_seed(7 + dist.get_rank())
    x = torch.randn(1, seq, hidden, device=device, dtype=dtype)
    g = torch.randn(1, seq, hidden, device=device, dtype=dtype)

    # Un-tiled reference
    xr = x.clone().detach().requires_grad_(True)
    yr = mlp_ref(xr)
    yr.backward(g)
    ref_grads = _full_grads(mlp_ref)
    ref_dx = xr.grad.detach().clone()

    # Tiled run — must not corrupt gradients on FSDP2.
    xt = x.clone().detach().requires_grad_(True)
    yt = mlp_tile(xt)
    yt.backward(g)
    tile_grads = _full_grads(mlp_tile)
    tile_dx = xt.grad.detach().clone()

    # Outputs match (this is just forward — should be tight).
    assert torch.allclose(yr.detach(), yt.detach(), atol=1e-3, rtol=1e-2), (
        f"FSDP2 forward mismatch max={((yr - yt).abs().max()).item()}"
    )
    # dX should match within bf16 tolerance.
    assert torch.allclose(ref_dx, tile_dx, atol=1e-2, rtol=1e-2), (
        f"FSDP2 dX mismatch max={((ref_dx - tile_dx).abs().max()).item()}"
    )
    # Param grads — the headline check for the reshard fix.
    for name, gref in ref_grads.items():
        gtile = tile_grads[name]
        rel = (
            (gref.float() - gtile.float()).norm() / (gref.float().norm() + 1e-6)
        ).item()
        assert rel < 5e-2, f"FSDP2 + tiled param-grad mismatch {name}: rel_err={rel}"


# ─────────────────────── scattermoe-lora regression guard ────────────────


def test_fsdp2_tiled_scattermoe_block_parity(dist_pg):
    """FSDP2 + tiled ScatterMoEGatedMLP block parity guard.

    Same shape of test as the dense case but routes through the
    ScatterMoE forward. Skips if scattermoe_lora kernels are not
    available in this env.
    """
    try:
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            ScatterMoEGatedMLP,
        )
    except ImportError:
        pytest.skip("scattermoe_lora kernels not available")

    pytest.importorskip("triton")

    from torch.distributed.fsdp import fully_shard

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    hidden, intermediate = 64, 128
    num_experts, top_k = 8, 2
    seq = 64
    dtype = torch.bfloat16

    def _make_block():
        torch.manual_seed(42)
        block = ScatterMoEGatedMLP()
        router = SimpleNamespace()
        router.layer = nn.Linear(hidden, num_experts, bias=False, dtype=dtype).to(
            device
        )
        router.top_k = top_k
        router.num_experts = num_experts
        block.router = router
        in_w = nn.Parameter(
            torch.randn(
                num_experts, 2 * intermediate, hidden, dtype=dtype, device=device
            )
            * 0.02
        )
        out_w = nn.Parameter(
            torch.randn(num_experts, hidden, intermediate, dtype=dtype, device=device)
            * 0.02
        )
        block.input_linear = nn.Module()
        block.input_linear.register_parameter("weight", in_w)
        block.output_linear = nn.Module()
        block.output_linear.register_parameter("weight", out_w)
        block.activation = nn.SiLU()
        return block

    block_ref = _make_block()
    block_tile = _make_block()
    fully_shard(block_ref)
    fully_shard(block_tile)
    _install_tiled_forward(block_tile, shards=4)

    torch.manual_seed(7 + dist.get_rank())
    x = torch.randn(1, seq, hidden, device=device, dtype=dtype)
    g = torch.randn(1, seq, hidden, device=device, dtype=dtype)

    xr = x.clone().detach().requires_grad_(True)
    yr = block_ref(xr)
    yr.backward(g)
    ref_grads = _full_grads(block_ref)
    ref_dx = xr.grad.detach().clone()

    xt = x.clone().detach().requires_grad_(True)
    yt = block_tile(xt)
    yt.backward(g)
    tile_grads = _full_grads(block_tile)
    tile_dx = xt.grad.detach().clone()

    def _rel(a, b):
        return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-6)).item()

    assert _rel(yt.detach(), yr.detach()) < 5e-2, (
        f"FSDP2 + tiled scattermoe forward rel_err={_rel(yt, yr)}"
    )
    assert _rel(tile_dx, ref_dx) < 5e-2, (
        f"FSDP2 + tiled scattermoe dX rel_err={_rel(tile_dx, ref_dx)}"
    )
    for name, gref in ref_grads.items():
        if name not in tile_grads:
            continue
        rel = _rel(tile_grads[name], gref)
        assert rel < 5e-2, f"FSDP2 + tiled scattermoe param-grad {name} rel_err={rel}"
