"""Math property tests for SAR spectral projection."""

import torch

from axolotl.integrations.sar.core import sar_project_matrix


def test_full_rank_identity():
    torch.manual_seed(0)
    dim = 32
    w_base = torch.randn(dim, dim)
    delta = torch.randn(dim, dim)

    result = sar_project_matrix(w_base, w_base + delta, rank=dim)

    assert result.rank == dim
    assert result.delta_rank == dim
    assert result.m is not None
    assert result.m.shape == (dim, dim)
    torch.testing.assert_close(result.delta_star, delta, atol=1e-5, rtol=0)


def test_orthogonal_annihilation():
    torch.manual_seed(1)
    dout, din, k = 24, 20, 6
    q = torch.linalg.qr(torch.randn(dout, dout)).Q
    p = torch.linalg.qr(torch.randn(din, din)).Q
    s = torch.linspace(5.0, 1.0, k)
    w_base = (q[:, :k] * s) @ p[:, :k].T
    delta = torch.outer(q[:, k], p[:, k])

    result = sar_project_matrix(w_base, w_base + delta, rank=k, delta_rank=1)

    assert result.m is not None
    assert result.m.shape == (k, k)
    assert result.m.abs().max().item() < 1e-6
    assert result.delta_star.norm().item() < 1e-5


def test_rewiring_identity_and_masks():
    torch.manual_seed(2)
    dout, din, k = 20, 28, 5
    w_base = torch.randn(dout, din)
    w_trained = w_base + 0.1 * torch.randn(dout, din)

    full = sar_project_matrix(w_base, w_trained, rank=k)
    assert full.m is not None
    assert full.m.shape == (k, k)
    assert full.m.dtype == torch.float32

    u, s, vh = torch.linalg.svd(w_base.to(torch.float32), full_matrices=False)
    u_k, s_k, vh_k = u[:, :k], s[:k], vh[:k, :]
    base32 = w_base.to(torch.float32)
    lhs = base32 + full.delta_star
    rhs = u_k @ (torch.diag(s_k) + full.m) @ vh_k + (base32 - (u_k * s_k) @ vh_k)
    torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=0)

    diag = sar_project_matrix(w_base, w_trained, rank=k, rewiring="diagonal")
    off = sar_project_matrix(w_base, w_trained, rank=k, rewiring="off_diagonal")
    off_mask = ~torch.eye(k, dtype=torch.bool)
    assert torch.all(diag.m[off_mask] == 0)
    torch.testing.assert_close(torch.diagonal(diag.m), torch.diagonal(full.m))
    assert torch.all(torch.diagonal(off.m) == 0)
    torch.testing.assert_close(off.m[off_mask], full.m[off_mask])
    torch.testing.assert_close(diag.delta_star + off.delta_star, full.delta_star)


def test_no_projection_control():
    torch.manual_seed(3)
    dout, din, k, delta_rank = 26, 18, 9, 4
    w_base = torch.randn(dout, din)
    w_trained = w_base + torch.randn(dout, din)

    result = sar_project_matrix(
        w_base, w_trained, rank=k, delta_rank=delta_rank, projection="none"
    )

    assert result.m is None
    assert result.rank == k
    assert result.delta_rank == delta_rank
    delta = (w_trained - w_base).to(torch.float32)
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    expected = (u[:, :delta_rank] * s[:delta_rank]) @ vh[:delta_rank, :]
    torch.testing.assert_close(result.delta_star, expected)
    tail = torch.linalg.svdvals(result.delta_star)[delta_rank:]
    assert tail.max().item() < 1e-5


def test_rectangular_edge_ranks_and_bf16_upcast():
    torch.manual_seed(4)
    dout, din = 40, 24
    w_base = torch.randn(dout, din)
    delta = 0.1 * (w_base @ torch.randn(din, din))

    full = sar_project_matrix(w_base, w_base + delta, rank=din)
    assert full.delta_star.shape == (dout, din)
    assert full.m.shape == (din, din)
    torch.testing.assert_close(full.delta_star, delta, atol=1e-4, rtol=0)

    wide_base = torch.randn(16, 40, dtype=torch.bfloat16)
    wide_trained = (wide_base.to(torch.float32) + 0.05 * torch.randn(16, 40)).to(
        torch.bfloat16
    )
    rank_one = sar_project_matrix(wide_base, wide_trained, rank=1)
    assert rank_one.delta_star.shape == (16, 40)
    assert rank_one.delta_star.dtype == torch.float32
    assert rank_one.m.shape == (1, 1)
    assert rank_one.m.dtype == torch.float32
    assert torch.linalg.svdvals(rank_one.delta_star)[1:].max().item() < 1e-6

    fp32 = sar_project_matrix(
        wide_base.to(torch.float32), wide_trained.to(torch.float32), rank=1
    )
    torch.testing.assert_close(rank_one.delta_star, fp32.delta_star)
    torch.testing.assert_close(rank_one.m, fp32.m)


def test_idempotence():
    torch.manual_seed(5)
    dout, din, k = 30, 22, 7
    w_base = torch.randn(dout, din)
    w_trained = w_base + 0.2 * torch.randn(dout, din)

    first = sar_project_matrix(w_base, w_trained, rank=k)
    second = sar_project_matrix(w_base, w_base + first.delta_star, rank=k)

    torch.testing.assert_close(second.delta_star, first.delta_star, atol=1e-5, rtol=0)
    torch.testing.assert_close(second.m, first.m, atol=1e-5, rtol=0)
