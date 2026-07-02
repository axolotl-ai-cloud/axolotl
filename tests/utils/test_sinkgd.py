"""Unit tests for the SinkGD optimizer's SR-Sinkhorn, including 3D fused-MoE shapes."""

import torch

from axolotl.utils.optimizers.sinkgd import SinkGD, sr_sinkhorn


def test_sr_sinkhorn_2d_fixed_point():
    """Rows converge to L2 norm sqrt(n), columns to sqrt(m)."""
    torch.manual_seed(0)
    m, n = 16, 24
    x = sr_sinkhorn(torch.randn(m, n), iters=20, eps=1e-8)
    assert torch.allclose(x.norm(dim=1), torch.full((m,), n**0.5), atol=1e-3)
    assert torch.allclose(x.norm(dim=0), torch.full((n,), m**0.5), atol=1e-3)


def test_sr_sinkhorn_3d_per_expert():
    """A fused MoE weight [E, M, N] is normalized independently per expert."""
    torch.manual_seed(0)
    E, m, n = 4, 16, 24
    x = sr_sinkhorn(torch.randn(E, m, n), iters=20, eps=1e-8)
    # each expert independently reaches the doubly-balanced fixed point
    assert torch.allclose(x.norm(dim=-1), torch.full((E, m), n**0.5), atol=1e-3)
    assert torch.allclose(x.norm(dim=-2), torch.full((E, n), m**0.5), atol=1e-3)


def test_sr_sinkhorn_3d_experts_decoupled():
    """Scaling one expert's gradient must not change the other experts' updates."""
    torch.manual_seed(0)
    g = torch.randn(4, 16, 24)
    x = sr_sinkhorn(g, iters=20, eps=1e-8)
    g2 = g.clone()
    g2[0] *= 100.0
    x2 = sr_sinkhorn(g2, iters=20, eps=1e-8)
    assert torch.allclose(x[1:], x2[1:], atol=1e-4)


def test_sinkgd_step_3d_stateless():
    """SinkGD updates a 3D expert weight and stores no optimizer state for it."""
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(4, 16, 24))
    opt = SinkGD(
        [{"params": [w], "use_sinkgd": True, "weight_decay": 0.0}],
        lr=1e-2,
        sinkgd_lr_scale=1.0,
    )
    before = w.detach().clone()
    w.grad = torch.randn_like(w)
    opt.step()
    assert not torch.allclose(before, w.detach())
    assert opt.state[w] == {}  # SR-Sinkhorn is stateless
