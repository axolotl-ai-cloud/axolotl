"""Tests for transformers' local expert kernels under the EP sentinel routing contract."""

import copy

import pytest
import torch
from transformers.integrations.moe import (
    batched_mm_experts_forward,
    grouped_mm_experts_forward,
)

from axolotl.integrations.expert_parallel import experts_fn
from axolotl.integrations.expert_parallel.experts_fn import (
    _eager_local,
    _normalize_sentinels,
)

E, K, H, INTER = 4, 2, 8, 16
KINDS = ("qwen3_moe", "mixtral")
KERNELS = {
    "grouped_mm": grouped_mm_experts_forward,
    "batched_mm": batched_mm_experts_forward,
}


def _build_experts(kind):
    torch.manual_seed(0)
    if kind == "qwen3_moe":
        from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

        m = Qwen3MoeExperts(
            Qwen3MoeConfig(
                hidden_size=H,
                moe_intermediate_size=INTER,
                num_experts=E,
                num_experts_per_tok=K,
                norm_topk_prob=True,
            )
        )
    else:
        from transformers.models.mixtral.configuration_mixtral import MixtralConfig
        from transformers.models.mixtral.modeling_mixtral import MixtralExperts

        m = MixtralExperts(
            MixtralConfig(
                hidden_size=H,
                intermediate_size=INTER,
                num_local_experts=E,
                num_experts_per_tok=K,
            )
        )
    with torch.no_grad():
        m.gate_up_proj.normal_(0, 0.2)
        m.down_proj.normal_(0, 0.2)
    return m


def _routing(num_local):
    g = torch.Generator().manual_seed(7)
    x = torch.randn(6, H, generator=g)
    w_raw = torch.rand(6, K, generator=g)
    if num_local == 4:
        idx_raw = torch.tensor([[0, 1], [-1, 2], [-1, -1], [3, 0], [1, -1], [2, 3]])
    else:
        idx_raw = torch.tensor([[0, 1], [-1, 1], [-1, -1], [1, 0], [0, -1], [1, 1]])
    return x, idx_raw, w_raw


def _prepare(kind, sharded, fake_ep_sharder=None):
    m = _build_experts(kind)
    if sharded:
        holder = torch.nn.Module()
        holder.experts = m
        assert fake_ep_sharder(holder, rank=0) == 1
        assert m._is_expert_parallel is True
    num_local = getattr(m, "num_local_experts", m.num_experts)
    x, idx_raw, w_raw = _routing(num_local)
    idx, w = _normalize_sentinels(idx_raw, w_raw, num_local)
    return m, num_local, x, idx, w


@pytest.fixture(autouse=True)
def _restore_experts_fn_globals():
    prev = experts_fn.get_backend()
    yield
    experts_fn.set_backend(prev)
    experts_fn.set_dispatch_chunks(1)


class TestSentinelKernelsWithFlag:
    @pytest.mark.parametrize("sharded", (False, True))
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("kernel", KERNELS)
    def test_matches_eager_and_grads_finite(
        self, fake_ep_sharder, kernel, kind, sharded
    ):
        m, num_local, x, idx, w = _prepare(kind, sharded, fake_ep_sharder)
        if not sharded:
            m._is_expert_parallel = True
        assert m._is_expert_parallel is True
        sentinel = idx == num_local
        assert int(idx.max()) == num_local
        assert (w[sentinel] == 0).all()

        ref = copy.deepcopy(m)
        xr = x.clone().requires_grad_(True)
        wr = w.clone().requires_grad_(True)
        yr = _eager_local(ref, xr, idx, wr)

        xk = x.clone().requires_grad_(True)
        wk = w.clone().requires_grad_(True)
        yk = KERNELS[kernel](m, xk, idx, wk)

        assert torch.isfinite(yk).all()
        assert torch.allclose(yk, yr, atol=1e-6, rtol=1e-5)
        assert torch.equal(yk[2], torch.zeros(H))

        gout = torch.randn(6, H, generator=torch.Generator().manual_seed(21))
        (yk * gout).sum().backward()
        (yr * gout).sum().backward()
        pairs = [
            (xk.grad, xr.grad),
            (wk.grad, wr.grad),
            (m.gate_up_proj.grad, ref.gate_up_proj.grad),
            (m.down_proj.grad, ref.down_proj.grad),
        ]
        for got, want in pairs:
            assert got is not None and want is not None
            assert torch.isfinite(got).all()
            assert torch.allclose(got, want, atol=1e-6)
        assert torch.equal(wk.grad[sentinel], torch.zeros_like(wk.grad[sentinel]))

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("kernel", KERNELS)
    def test_caller_routing_not_mutated(self, kernel, kind):
        m, _, x, idx, w = _prepare(kind, sharded=False)
        m._is_expert_parallel = True
        idx_copy = idx.clone()
        w_copy = w.clone()
        KERNELS[kernel](m, x, idx, w)
        assert torch.equal(idx, idx_copy)
        assert torch.equal(w, w_copy)


class TestSentinelKernelsWithoutFlag:
    """grouped_mm without the flag leaves its sentinel-tail rows uninitialized upstream, so
    its output is not asserted either way."""

    @pytest.mark.parametrize("kind", KINDS)
    def test_batched_mm_rejects_sentinel_without_flag(self, kind):
        m, _, x, idx, w = _prepare(kind, sharded=False)
        m._is_expert_parallel = False
        with pytest.raises(IndexError):
            batched_mm_experts_forward(m, x, idx, w)

    @pytest.mark.parametrize("kind", KINDS)
    def test_ep_forward_sets_flag_before_local_kernel(self, kind):
        m = _build_experts(kind)
        assert m._is_expert_parallel is False
        experts_fn.set_backend("torch")
        x, idx_raw, w_raw = _routing(E)
        y = experts_fn._ep_forward(
            m, x, idx_raw, w_raw, local="batched_mm", backend="torch"
        )
        assert m._is_expert_parallel is True
        y_eager = experts_fn._ep_forward(
            m, x, idx_raw, w_raw, local="eager", backend="torch"
        )
        assert torch.allclose(y, y_eager, atol=1e-6)
        y_grouped = experts_fn._ep_forward(
            m, x, idx_raw, w_raw, local="grouped_mm", backend="torch"
        )
        assert torch.allclose(y_grouped, y_eager, atol=1e-6)
