"""Unit tests for the SinkGD optimizer's SR-Sinkhorn, including 3D fused-MoE shapes."""

import pytest
import torch

from axolotl.utils.optimizers.sinkgd import (
    SinkGD,
    SinkGDMD,
    _pop_sinkgd_extra_kwargs,
    _specnorm_gram_cols,
    _specnorm_gram_rows,
    single_param_sinkgd_specnorm,
    sr_sinkhorn,
)


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


# ---- Feature A: width-aware scaling -------------------------------------------------


def _mk(**kw):
    w = torch.nn.Parameter(torch.randn(24, 16))
    opt = SinkGD(
        [{"params": [w], "use_sinkgd": True, "weight_decay": 0.0}],
        lr=1e-2,
        sinkgd_lr_scale=0.05,
        **kw,
    )
    return w, opt


def test_alpha_eff_no_base_width_is_plain_scalar():
    """Backward-compat: without base_width, alpha_eff == sinkgd_lr_scale exactly."""
    w, opt = _mk()
    assert opt._alpha_eff(w) == pytest.approx(0.05)


def test_alpha_eff_width_aware_1_over_din():
    """alpha_eff = sinkgd_lr_scale * (base_width / d_in) ** exponent; d_in = shape[-1]."""
    w, opt = _mk(sinkgd_base_width=16)  # d_in == base_width -> unchanged
    assert opt._alpha_eff(w) == pytest.approx(0.05)
    w, opt = _mk(sinkgd_base_width=8)  # base < d_in(16) -> halved
    assert opt._alpha_eff(w) == pytest.approx(0.025)
    w, opt = _mk(sinkgd_base_width=8, sinkgd_lr_width_exponent=0.0)  # exponent 0 -> off
    assert opt._alpha_eff(w) == pytest.approx(0.05)


def test_alpha_eff_uses_input_dim_for_fused_layouts():
    """For a fused [d_out, d_in] weight the width factor keys on d_in (shared input)."""
    w_qkv = torch.nn.Parameter(torch.randn(3 * 64, 32))  # fused QKV: d_in = 32
    opt = SinkGD(
        [{"params": [w_qkv], "use_sinkgd": True, "weight_decay": 0.0}],
        lr=1e-2,
        sinkgd_lr_scale=0.05,
        sinkgd_base_width=16,
    )
    assert opt._alpha_eff(w_qkv) == pytest.approx(0.05 * 16 / 32)


# ---- Feature A: spectral norm -------------------------------------------------------


def test_spectral_norm_off_is_unchanged():
    """spectral_norm=False must reproduce the default update byte-for-byte and add no state."""
    torch.manual_seed(0)
    w1 = torch.nn.Parameter(torch.randn(24, 16))
    w2 = torch.nn.Parameter(w1.detach().clone())
    g = torch.randn(24, 16)
    o1 = SinkGD(
        [{"params": [w1], "use_sinkgd": True, "weight_decay": 0.0}],
        lr=1e-2,
        sinkgd_lr_scale=0.3,
    )
    o2 = SinkGD(
        [{"params": [w2], "use_sinkgd": True, "weight_decay": 0.0}],
        lr=1e-2,
        sinkgd_lr_scale=0.3,
        sinkgd_spectral_norm=False,
    )
    w1.grad = g.clone()
    w2.grad = g.clone()
    o1.step()
    o2.step()
    assert torch.equal(w1.detach(), w2.detach())
    assert o2.state[w2] == {}


@pytest.mark.parametrize("shape", [(64, 64), (96, 32), (32, 96)])
def test_power_iteration_matches_matrix_2norm(shape):
    """The persisted power iteration converges to torch.linalg.matrix_norm(U, 2)."""
    torch.manual_seed(0)
    m, n = shape
    g = torch.randn(m, n)
    u = sr_sinkhorn(g, iters=5, eps=1e-8)
    true = torch.linalg.matrix_norm(u, 2).item()
    vec = torch.randn(n)
    vec /= vec.norm()
    for _ in range(30):  # warm-started convergence over many steps
        v = u @ vec
        v /= v.norm()
        vec = u.t() @ v
        sig = vec.norm()
        vec /= sig
    assert sig.item() == pytest.approx(true, rel=1e-2)


def test_spectral_norm_step_pins_operator_norm_and_persists_u():
    """spectral_norm on: step runs, changes the param, persists the u vector as state."""
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(24, 16))
    opt = SinkGD(
        [{"params": [w], "use_sinkgd": True, "weight_decay": 0.0}],
        lr=1.0,
        sinkgd_lr_scale=1.0,
        sinkgd_spectral_norm=True,
        sinkgd_spectral_norm_iters=3,
    )
    before = w.detach().clone()
    for _ in range(4):
        w.grad = torch.randn(24, 16)
        opt.step()
    assert not torch.allclose(before, w.detach())
    assert opt.state[w]["specnorm_u"].shape == (16,)


def test_spectral_specnorm_fn_rescales_to_target():
    """The compiled spectral update drives ||U||_2 toward the requested target norm."""
    torch.manual_seed(0)
    m, n = 48, 32
    p = torch.zeros(m, n)
    u = torch.randn(n)
    u /= u.norm()
    # accumulate the update alone (p starts 0, lr=1, wd=0) over steps with a fixed grad so u warms
    g = torch.randn(m, n)
    target = (m / n) ** 0.5
    for _ in range(20):
        p.zero_()
        single_param_sinkgd_specnorm(
            p, g, u, torch.tensor(1.0), 0.0, 5, 1e-8, target, 2, False
        )
    # p = -lr * U_spectral ; its operator norm should be ~target
    assert torch.linalg.matrix_norm(p, 2).item() == pytest.approx(target, rel=0.1)


# ---- Feature A: config validation ---------------------------------------------------


def test_pop_kwargs_casts_strings():
    out = _pop_sinkgd_extra_kwargs(
        {
            "sinkgd_spectral_norm": "true",
            "sinkgd_base_width": "256",
            "sinkgd_spectral_norm_iters": "2",
            "sinkgd_lr_width_exponent": "1.0",
        }
    )
    assert out["sinkgd_spectral_norm"] is True
    assert out["sinkgd_base_width"] == 256
    assert out["sinkgd_spectral_norm_iters"] == 2
    assert out["sinkgd_lr_width_exponent"] == 1.0


def test_pop_kwargs_rejects_width_double_count():
    """base_width (1/d_in) and spectral_target=muon both correct for width -> rejected."""
    with pytest.raises(ValueError):
        _pop_sinkgd_extra_kwargs(
            {
                "sinkgd_spectral_norm": True,
                "sinkgd_spectral_target": "muon",
                "sinkgd_base_width": 256,
            }
        )


def test_pop_kwargs_rejects_bad_target():
    with pytest.raises(ValueError):
        _pop_sinkgd_extra_kwargs({"sinkgd_spectral_target": "bogus"})


# ---- Feature A: sharded (FSDP2) spectral power iteration -----------------------------


@pytest.mark.parametrize("shape", [(64, 48), (128, 32), (32, 128), (96, 96)])
@pytest.mark.parametrize("shard_dim", [-2, -1])
def test_sharded_gram_power_iteration_matches_spectral_norm(shape, shard_dim):
    """The Gram-matrix partials (summed across shards, as the all-reduce does) recover the
    global ``||U||_2`` — the math behind DistSinkGD's sharded spectral norm, verified here
    without a real process group by emulating all_reduce as a local sum over row/col shards."""
    torch.manual_seed(0)
    m, n = shape
    u_mat = sr_sinkhorn(torch.randn(m, n), iters=5, eps=1e-8)
    true = torch.linalg.matrix_norm(u_mat, 2).item()
    if shard_dim == -2:  # rows sharded -> iterate right vector in R^n
        shards = list(u_mat.chunk(2, dim=0))
        vec = torch.randn(n)
        vec /= vec.norm()
        helper = _specnorm_gram_rows
    else:  # cols sharded -> iterate left vector in R^m
        shards = list(u_mat.chunk(2, dim=1))
        vec = torch.randn(m)
        vec /= vec.norm()
        helper = _specnorm_gram_cols
    nrm = None
    for _ in range(
        30
    ):  # cold start; warm-started across steps in real use it converges fast
        partial = sum(
            helper(s, vec) for s in shards
        )  # all_reduce == sum of shard partials
        nrm = partial.norm()
        vec = partial / nrm
    assert nrm.sqrt().item() == pytest.approx(true, rel=1e-2)


# ---- A+B: SinkGDMD (MD Frobenius sphere) --------------------------------------------


def test_sinkgdmd_keeps_weight_on_sphere():
    """Every step reprojects the 2D weight onto its enable-time Frobenius sphere."""
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(64, 48))
    tn0 = w.detach().float().norm().item()
    opt = SinkGDMD(
        [{"params": [w], "use_sinkgd": True, "weight_decay": 0.0}],
        lr=1.0,
        sinkgd_lr_scale=0.05,
    )
    for _ in range(8):
        w.grad = torch.randn(64, 48)
        opt.step()
    assert w.detach().float().norm().item() == pytest.approx(tn0, rel=1e-4)
    assert set(opt.state[w]) == {"md_target_norm", "specnorm_u"}


def test_sinkgdmd_per_expert_sphere():
    """3D fused-MoE weight: each expert stays on its own Frobenius sphere."""
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(4, 32, 24))
    tn = w.detach().float().norm(dim=(-2, -1)).clone()
    opt = SinkGDMD(
        [{"params": [w], "use_sinkgd": True, "weight_decay": 0.0}],
        lr=1.0,
        sinkgd_lr_scale=0.1,
    )
    for _ in range(5):
        w.grad = torch.randn(4, 32, 24)
        opt.step()
    after = w.detach().float().norm(dim=(-2, -1))
    assert torch.allclose(after, tn, atol=1e-3)


def test_sinkgdmd_moves_weight_and_uses_adam_fallback():
    """MD changes the matrix; a 1D param still goes to the 8-bit AdamW fallback."""
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(32, 16))
    b = torch.nn.Parameter(torch.randn(32))
    before = w.detach().clone()
    opt = SinkGDMD(
        [
            {"params": [w], "use_sinkgd": True, "weight_decay": 0.0},
            {"params": [b], "use_sinkgd": False, "weight_decay": 0.0},
        ],
        lr=1e-2,
        sinkgd_lr_scale=1.0,
    )
    w.grad = torch.randn(32, 16)
    b.grad = torch.randn(32)
    opt.step()
    assert not torch.allclose(before, w.detach())
    assert "exp_avg" in opt.state[b]  # 1D param uses the Adam fallback


# ---- fused Triton kernel path (sinkgd_fused_kernel) -----------------------------------


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused kernels need CUDA"
)


@requires_cuda
@pytest.mark.parametrize("shape", [(256, 192), (1024, 8192), (4, 96, 64)])
def test_fused_base_matches_compiled(shape):
    """sinkgd_fused_kernel=True reproduces the compiled base update to bf16 tolerance,
    including a wide-short shape that exercises the column-split kernels."""
    torch.manual_seed(0)
    w0 = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    g = torch.randn_like(w0)
    results = []
    for fused in (False, True):
        w = torch.nn.Parameter(w0.clone())
        opt = SinkGD(
            [{"params": [w], "use_sinkgd": True, "weight_decay": 0.01}],
            lr=1e-2,
            sinkgd_lr_scale=0.5,
            weight_decay=0.01,
            sinkgd_fused_kernel=fused,
        )
        w.grad = g.clone()
        opt.step()
        results.append(w.detach().float())
    rel = (results[0] - results[1]).abs().max() / results[0].abs().max()
    assert rel.item() < 3e-2


@requires_cuda
@pytest.mark.parametrize("target", ["unit", "muon"])
def test_fused_spectral_matches_compiled(target):
    torch.manual_seed(0)
    w0 = torch.randn(96, 64, device="cuda", dtype=torch.bfloat16)
    g = torch.randn_like(w0)
    results = []
    for fused in (False, True):
        w = torch.nn.Parameter(w0.clone())
        opt = SinkGD(
            [{"params": [w], "use_sinkgd": True, "weight_decay": 0.0}],
            lr=1e-2,
            sinkgd_lr_scale=0.5,
            sinkgd_spectral_norm=True,
            sinkgd_spectral_norm_iters=2,
            sinkgd_spectral_target=target,
            sinkgd_fused_kernel=fused,
            sinkgd_fused_min_numel=0,
        )
        torch.manual_seed(7)  # same u init both paths
        for _ in range(3):
            w.grad = g.clone()
            opt.step()
        results.append(w.detach().float())
    rel = (results[0] - results[1]).abs().max() / results[0].abs().max()
    assert rel.item() < 3e-2


@requires_cuda
def test_fused_md_matches_compiled_and_stays_on_sphere():
    torch.manual_seed(0)
    w0 = torch.randn(96, 64, device="cuda", dtype=torch.bfloat16)
    tn0 = w0.float().norm().item()
    g = torch.randn_like(w0)
    results = []
    for fused in (False, True):
        w = torch.nn.Parameter(w0.clone())
        opt = SinkGDMD(
            [{"params": [w], "use_sinkgd": True, "weight_decay": 0.0}],
            lr=1e-2,
            sinkgd_lr_scale=0.5,
            sinkgd_fused_kernel=fused,
            sinkgd_fused_min_numel=0,
        )
        torch.manual_seed(7)
        for _ in range(3):
            w.grad = g.clone()
            opt.step()
        assert w.detach().float().norm().item() == pytest.approx(tn0, rel=1e-3)
        results.append(w.detach().float())
    rel = (results[0] - results[1]).abs().max() / results[0].abs().max()
    assert rel.item() < 3e-2


@requires_cuda
def test_fused_falls_back_for_stochastic_round():
    """bf16 stochastic rounding is not implemented in the fused kernels -> compiled path."""
    w = torch.nn.Parameter(torch.randn(32, 16, device="cuda", dtype=torch.bfloat16))
    opt = SinkGD(
        [{"params": [w], "use_sinkgd": True, "weight_decay": 0.0}],
        lr=1e-2,
        sinkgd_fused_kernel=True,
        bf16_stochastic_round=True,
    )
    assert not opt._fused_ok(w)
    w2 = torch.nn.Parameter(torch.randn(32, 16, device="cuda", dtype=torch.bfloat16))
    opt2 = SinkGD(
        [{"params": [w2], "use_sinkgd": True, "weight_decay": 0.0}],
        lr=1e-2,
        sinkgd_fused_kernel=True,
    )
    assert opt2._fused_ok(w2)


def test_pop_kwargs_casts_fused_flag():
    out = _pop_sinkgd_extra_kwargs(
        {"sinkgd_fused_kernel": "true", "sinkgd_fused_min_numel": "1024"}
    )
    assert out["sinkgd_fused_kernel"] is True
    assert out["sinkgd_fused_min_numel"] == 1024


@requires_cuda
def test_fused_epilogue_size_gate():
    """Single-device spec/md fused routing is size-gated (small matrices lose to the eager
    epilogue overhead); base mode is not."""
    w = torch.nn.Parameter(torch.randn(32, 16, device="cuda", dtype=torch.bfloat16))
    opt = SinkGD(
        [{"params": [w], "use_sinkgd": True, "weight_decay": 0.0}],
        lr=1e-2,
        sinkgd_fused_kernel=True,
        sinkgd_spectral_norm=True,
    )
    assert opt._fused_ok(w) and not opt._fused_ok(w, epilogue=True)
    opt0 = SinkGD(
        [{"params": [w], "use_sinkgd": True, "weight_decay": 0.0}],
        lr=1e-2,
        sinkgd_fused_kernel=True,
        sinkgd_spectral_norm=True,
        sinkgd_fused_min_numel=0,
    )
    assert opt0._fused_ok(w, epilogue=True)
