"""Unit tests for the sonic-moe EP sentinel backward runtime patch."""

import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

from axolotl.integrations.kernels.libs.sonicmoe.ep_backward_patch import (
    _BUG_SIGNATURE,
    _make_fixed_down_projection_backward_act,
    apply_sonicmoe_ep_backward_patch,
)


class TestFixedDownProjectionBackwardAct:
    """The replacement must null the uninitialized grouped tail: sentinel lanes get
    exactly zero score-grad and lane 0 is never corrupted by the aliased tail writes."""

    def _run(self, ds_scattered_tail):
        num_lanes = 8
        n_valid = 5
        # Grouped rows 0..4 map to lanes [3, 1, 4, 0, 2]; tail rows alias lane 0.
        s_scatter_idx = torch.tensor([3, 1, 4, 0, 2, 0, 0, 0], dtype=torch.int32)
        expert_frequency_offset = torch.tensor([0, 2, n_valid], dtype=torch.int32)

        valid_vals = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        ds_scattered = torch.cat([valid_vals, ds_scattered_tail])

        captured = {}

        def fake_gemm_dgated(*args, **kwargs):
            captured["colvec_scale"] = kwargs["colvec_scale"]
            return None, None, ds_scattered

        bwd_mod = SimpleNamespace(gemm_dgated=fake_gemm_dgated)
        fixed = _make_fixed_down_projection_backward_act(bwd_mod)

        hidden, inter, num_experts = 4, 3, 2
        ds = torch.full((num_lanes,), float("nan"))
        fixed(
            dout=torch.zeros(2, hidden),
            h=torch.zeros(num_lanes, 2 * inter),
            w2=torch.zeros(hidden, inter, num_experts),
            dh=torch.zeros(num_lanes, 2 * inter),
            ds=ds,
            b2=None,
            db2=None,
            a_prime=torch.zeros(num_lanes, inter),
            topk_scores=torch.arange(num_lanes, dtype=torch.float32),
            expert_frequency_offset=expert_frequency_offset,
            x_gather_idx=torch.zeros(num_lanes, dtype=torch.int32),
            s_scatter_idx=s_scatter_idx,
            activation_type="swiglu",
        )
        return ds

    def test_sentinel_tail_zeroed_and_lane0_uncorrupted(self):
        ds = self._run(torch.full((3,), float("nan")))
        assert torch.isfinite(ds).all()
        # Valid grouped rows land on their lanes; lane 0 keeps its true grad despite
        # the tail's aliased writes targeting it.
        assert ds[3] == 10.0 and ds[1] == 20.0 and ds[4] == 30.0
        assert ds[0] == 40.0 and ds[2] == 50.0
        # Sentinel lanes (never a valid scatter target) get exactly zero.
        assert ds[5] == 0.0 and ds[6] == 0.0 and ds[7] == 0.0

    def test_no_sentinels_matches_plain_scatter(self):
        num_lanes = 4
        s_scatter_idx = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
        ds_scattered = torch.tensor([1.0, 2.0, 3.0, 4.0])
        bwd_mod = SimpleNamespace(
            gemm_dgated=lambda *a, **k: (None, None, ds_scattered)
        )
        fixed = _make_fixed_down_projection_backward_act(bwd_mod)

        ds = torch.full((num_lanes,), float("nan"))
        fixed(
            dout=torch.zeros(2, 4),
            h=torch.zeros(num_lanes, 6),
            w2=torch.zeros(4, 3, 2),
            dh=torch.zeros(num_lanes, 6),
            ds=ds,
            b2=None,
            db2=None,
            a_prime=torch.zeros(num_lanes, 3),
            topk_scores=torch.ones(num_lanes),
            expert_frequency_offset=torch.tensor([0, 2, num_lanes], dtype=torch.int32),
            x_gather_idx=torch.zeros(num_lanes, dtype=torch.int32),
            s_scatter_idx=s_scatter_idx,
            activation_type="swiglu",
        )
        expected = torch.empty(num_lanes)
        expected[s_scatter_idx.long()] = ds_scattered
        assert torch.equal(ds, expected)


class TestApplyPatchGate:
    """Source-signature check and the `_axolotl_patched` idempotency marker."""

    def _fake_kernel(self, tmp_path, bwd_source):
        bwd_path = tmp_path / "backward.py"
        bwd_path.write_text(bwd_source)
        bwd_mod = ModuleType("fake_sonic.functional.backward")
        bwd_mod.__file__ = str(bwd_path)
        exec(compile(bwd_source, str(bwd_path), "exec"), bwd_mod.__dict__)

        functional_mod = ModuleType("fake_sonic.functional")
        exec(
            "def moe_general_routing_inputs():\n    pass\n"
            "def _down_projection_backward_act():\n    pass\n",
            functional_mod.__dict__,
        )
        functional_mod.moe_general_routing_inputs.__module__ = "fake_sonic.functional"

        sys.modules["fake_sonic.functional"] = functional_mod
        sys.modules["fake_sonic.functional.backward"] = bwd_mod
        return SimpleNamespace(
            moe_general_routing_inputs=functional_mod.moe_general_routing_inputs
        ), functional_mod

    def _apply(self, kernel):
        apply_sonicmoe_ep_backward_patch.cache_clear()
        with mock.patch(
            "transformers.integrations.sonicmoe._load_sonicmoe_kernel",
            return_value=kernel,
        ):
            return apply_sonicmoe_ep_backward_patch()

    def test_patches_buggy_kernel_once(self, tmp_path):
        kernel, functional_mod = self._fake_kernel(
            tmp_path, f"# {_BUG_SIGNATURE}\ndef gemm_dgated():\n    pass\n"
        )
        assert self._apply(kernel) is True
        patched = functional_mod.__dict__["_down_projection_backward_act"]
        assert patched._axolotl_patched is True

        # Re-applying (even with the memo cleared) must not wrap again.
        assert self._apply(kernel) is True
        assert functional_mod.__dict__["_down_projection_backward_act"] is patched

    def test_skips_fixed_kernel(self, tmp_path):
        kernel, functional_mod = self._fake_kernel(
            tmp_path, "def gemm_dgated():\n    pass\n"
        )
        original = functional_mod.__dict__["_down_projection_backward_act"]
        assert self._apply(kernel) is False
        assert functional_mod.__dict__["_down_projection_backward_act"] is original
