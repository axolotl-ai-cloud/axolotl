"""Unit tests for shared Mamba2 SSM utilities (mamba_utils.py).

Tests cover get_seq_idx correctness under:
  - single-rank packing
  - context parallelism (mid-sample chunk starts)
  - batch dimension
  - dtype and device
  - no-negative regression (CP rank > 0 must never produce -1)
  - mamba2_cp_correction mathematical correctness
  - wrap_mamba_scan_for_cp wrapper behaviour
  - end-to-end CP split: full 2K scan == 2×1K split + correction
"""

import types
from unittest.mock import patch

import torch
import torch.nn.functional as F

from axolotl.monkeypatch.models.mamba_utils import (
    get_seq_idx,
    mamba2_cp_correction,
    wrap_mamba_scan_for_cp,
)


def _reference_ssm_scan(x, dt, A, B, C, dt_bias=None, dt_softplus=False, h0=None):
    """Pure-PyTorch step-by-step SSM scan (reference implementation).

    Implements the Mamba2 discrete SSM recurrence:
        Δ_t = softplus(dt_t + dt_bias)  or  dt_t
        Ā_t = exp(A · Δ_t)
        h_t = Ā_t · h_{t-1}  +  B_t ⊗ x_t
        y_t = (C_t · h_t).sum(dim=n)

    Args:
        x:  [B, T, H, d]
        dt: [B, T, H]
        A:  [H]  (log-space, negative)
        B:  [B, T, n_groups, n]
        C:  [B, T, n_groups, n]
        dt_bias:    [H] or None
        dt_softplus: bool
        h0: [B, H, d, n] initial state, or None → zeros

    Returns:
        out:     [B, T, H, d]
        h_final: [B, H, d, n]
    """
    B_batch, T, H, d = x.shape
    n_groups = B.shape[2]
    n = B.shape[3]
    heads_per_group = H // n_groups

    dt_eff = dt + dt_bias[None, None, :] if dt_bias is not None else dt
    if dt_softplus:
        dt_eff = F.softplus(dt_eff)

    h = torch.zeros(B_batch, H, d, n, dtype=x.dtype) if h0 is None else h0.clone()

    outputs = []
    for t in range(T):
        A_bar = torch.exp(A[None, :] * dt_eff[:, t, :])  # [B, H]
        B_t = B[:, t].repeat_interleave(heads_per_group, dim=1)  # [B, H, n]
        C_t = C[:, t].repeat_interleave(heads_per_group, dim=1)  # [B, H, n]

        h = A_bar[:, :, None, None] * h + B_t[:, :, None, :] * x[:, t, :, :, None]
        y_t = (C_t[:, :, None, :] * h).sum(dim=-1)  # [B, H, d]
        outputs.append(y_t)

    return torch.stack(outputs, dim=1), h


class TestGetSeqIdx:
    """Tests for get_seq_idx(position_ids) → seq_idx."""

    def test_single_sample_no_packing(self):
        """Single sample with no packing: all zeros."""
        pos = torch.tensor([[0, 1, 2, 3, 4]])
        out = get_seq_idx(pos)
        assert out.tolist() == [[0, 0, 0, 0, 0]]

    def test_two_packed_samples(self):
        """Two packed samples: index increments at the second sample boundary."""
        pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2]])
        out = get_seq_idx(pos)
        assert out.tolist() == [[0, 0, 0, 0, 1, 1, 1]]

    def test_three_packed_samples(self):
        """Three packed samples."""
        pos = torch.tensor([[0, 1, 0, 1, 2, 0]])
        out = get_seq_idx(pos)
        assert out.tolist() == [[0, 0, 1, 1, 1, 2]]

    def test_cp_rank_mid_sample_start(self):
        """CP rank > 0: chunk starts mid-sample (position_ids[0] != 0).

        Must produce non-negative seq_idx starting at 0, not -1.
        """
        pos = torch.tensor([[3, 4, 5, 0, 1, 2]])
        out = get_seq_idx(pos)
        assert out.tolist() == [[0, 0, 0, 1, 1, 1]]

    def test_cp_rank_entire_chunk_mid_sample(self):
        """CP rank whose entire chunk is mid-sample (no sample boundary)."""
        pos = torch.tensor([[5, 6, 7, 8, 9]])
        out = get_seq_idx(pos)
        assert out.tolist() == [[0, 0, 0, 0, 0]]

    def test_no_negative_values_regression(self):
        """seq_idx must never contain -1 for any valid position_ids input."""
        cases = [
            [[1, 2, 3]],
            [[10, 11, 12, 0, 1]],
            [[0, 0, 0]],
        ]
        for pos_list in cases:
            pos = torch.tensor(pos_list)
            out = get_seq_idx(pos)
            assert out.min().item() >= 0, f"Negative seq_idx for pos={pos_list}"

    def test_batch_dimension(self):
        """Batch of 3 sequences, each independently packed."""
        pos = torch.tensor(
            [
                [0, 1, 2, 0, 1],
                [0, 1, 0, 1, 2],
                [3, 4, 0, 1, 2],
            ]
        )
        out = get_seq_idx(pos)
        assert out.tolist() == [
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ]

    def test_output_dtype_is_int32(self):
        """Output dtype must be torch.int32 (mamba-ssm kernel requirement)."""
        pos = torch.tensor([[0, 1, 2, 0, 1]])
        out = get_seq_idx(pos)
        assert out.dtype == torch.int32

    def test_output_shape_matches_input(self):
        """Output shape matches input shape."""
        pos = torch.zeros(4, 128, dtype=torch.long)
        out = get_seq_idx(pos)
        assert out.shape == pos.shape

    def test_single_token(self):
        """Edge case: single token sequence."""
        pos = torch.tensor([[0]])
        out = get_seq_idx(pos)
        assert out.tolist() == [[0]]

    def test_cp_rank_starts_at_1(self):
        """CP rank that starts exactly at position 1 (not 0)."""
        pos = torch.tensor([[1, 2, 3, 0, 1]])
        out = get_seq_idx(pos)
        assert out.tolist() == [[0, 0, 0, 1, 1]]

    def test_many_packed_samples(self):
        """Many single-token samples packed together."""
        pos = torch.tensor([[0, 0, 0, 0, 0, 0]])
        out = get_seq_idx(pos)
        assert out.tolist() == [[0, 1, 2, 3, 4, 5]]


class TestMamba2CpCorrection:
    """Tests for mamba2_cp_correction mathematical correctness."""

    def test_zero_h_prev_is_noop(self):
        """When h_prev is all zeros, output should be unchanged."""
        B, T, H, d, n = 1, 8, 4, 16, 8
        n_groups = 2

        out = torch.randn(B, T, H * d)
        h_final = torch.randn(B, H, d, n)
        C = torch.randn(B, T, n_groups, n)
        cum_A = torch.randn(B, T, H)
        h_prev = torch.zeros(B, H, d, n)

        corrected_out, corrected_h = mamba2_cp_correction(
            out,
            h_final,
            C,
            cum_A,
            h_prev,
            num_heads=H,
            head_dim=d,
        )

        torch.testing.assert_close(corrected_out, out)
        torch.testing.assert_close(corrected_h, h_final)

    def test_correction_shapes(self):
        """Output shapes must match input shapes."""
        B, T, H, d, n = 2, 16, 8, 32, 16
        n_groups = 4

        out = torch.randn(B, T, H * d)
        h_final = torch.randn(B, H, d, n)
        C = torch.randn(B, T, n_groups, n)
        cum_A = torch.randn(B, T, H)
        h_prev = torch.randn(B, H, d, n)

        corrected_out, corrected_h = mamba2_cp_correction(
            out,
            h_final,
            C,
            cum_A,
            h_prev,
            num_heads=H,
            head_dim=d,
        )

        assert corrected_out.shape == out.shape
        assert corrected_h.shape == h_final.shape

    def test_correction_adds_to_output(self):
        """With nonzero h_prev, output should differ from input."""
        B, T, H, d, n = 1, 4, 2, 8, 4
        n_groups = 1

        out = torch.zeros(B, T, H * d)
        h_final = torch.zeros(B, H, d, n)
        C = torch.ones(B, T, n_groups, n)
        cum_A = torch.zeros(B, T, H)  # exp(0) = 1, so full propagation
        h_prev = torch.ones(B, H, d, n)

        corrected_out, corrected_h = mamba2_cp_correction(
            out,
            h_final,
            C,
            cum_A,
            h_prev,
            num_heads=H,
            head_dim=d,
        )

        # With exp(cum_A)=1, C=1, h_prev=1: delta_y should be nonzero
        assert corrected_out.abs().sum() > 0
        assert corrected_h.abs().sum() > 0

    def test_correction_h_final_formula(self):
        """Verify h_final correction: h_final + decay_T * h_prev."""
        B, T, H, d, n = 1, 4, 2, 8, 4
        n_groups = 1

        h_final = torch.zeros(B, H, d, n)
        C = torch.ones(B, T, n_groups, n)
        cum_A = torch.zeros(B, T, H)
        h_prev = torch.ones(B, H, d, n) * 2.0
        out = torch.zeros(B, T, H * d)

        _, corrected_h = mamba2_cp_correction(
            out,
            h_final,
            C,
            cum_A,
            h_prev,
            num_heads=H,
            head_dim=d,
        )

        # exp(0) * 2.0 = 2.0 for all elements
        expected = torch.ones(B, H, d, n) * 2.0
        torch.testing.assert_close(corrected_h, expected)


class TestCpSplitMatchesFullScan:
    """End-to-end: full sequence scan == split into chunks + CP correction.

    Runs a reference SSM scan on a full 2K-token sequence, then simulates
    2-rank CP by splitting into 2×1K, running each half with h₀=0, and
    applying mamba2_cp_correction to rank 1 using rank 0's final state.
    The concatenated result must match the single-rank reference.
    """

    def test_2k_vs_2x1k_output_matches(self):
        """Full 2048-token scan output == two 1024-token chunks + CP correction."""
        torch.manual_seed(42)
        B, T, H, d, n = 1, 2048, 4, 16, 8
        n_groups = 2
        dt_bias = torch.randn(H) * 0.1

        x = torch.randn(B, T, H, d)
        dt = torch.randn(B, T, H) * 0.1
        A = -torch.rand(H).abs() - 0.01
        B_ssm = torch.randn(B, T, n_groups, n) * 0.1
        C_ssm = torch.randn(B, T, n_groups, n) * 0.1

        ref_out, ref_h = _reference_ssm_scan(
            x, dt, A, B_ssm, C_ssm, dt_bias=dt_bias, dt_softplus=True
        )

        T2 = T // 2

        out_0, h_final_0 = _reference_ssm_scan(
            x[:, :T2],
            dt[:, :T2],
            A,
            B_ssm[:, :T2],
            C_ssm[:, :T2],
            dt_bias=dt_bias,
            dt_softplus=True,
        )

        out_1, h_final_1 = _reference_ssm_scan(
            x[:, T2:],
            dt[:, T2:],
            A,
            B_ssm[:, T2:],
            C_ssm[:, T2:],
            dt_bias=dt_bias,
            dt_softplus=True,
        )

        dt_eff_1 = F.softplus(dt[:, T2:] + dt_bias[None, None, :])
        cum_A_1 = torch.cumsum(A[None, None, :] * dt_eff_1, dim=1)

        corrected_out_1, corrected_h_1 = mamba2_cp_correction(
            out_1.view(B, T2, H * d),
            h_final_1,
            C_ssm[:, T2:],
            cum_A_1,
            h_final_0,
            num_heads=H,
            head_dim=d,
        )
        corrected_out_1 = corrected_out_1.view(B, T2, H, d)

        reconstructed = torch.cat([out_0, corrected_out_1], dim=1)

        torch.testing.assert_close(reconstructed, ref_out, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(corrected_h_1, ref_h, rtol=1e-4, atol=1e-4)

    def test_2k_vs_2x1k_with_batch(self):
        """Same split test with batch_size > 1."""
        torch.manual_seed(123)
        B, T, H, d, n = 3, 512, 2, 8, 4
        n_groups = 1
        dt_bias = torch.randn(H) * 0.05

        x = torch.randn(B, T, H, d)
        dt = torch.randn(B, T, H) * 0.1
        A = -torch.rand(H).abs() - 0.01
        B_ssm = torch.randn(B, T, n_groups, n) * 0.1
        C_ssm = torch.randn(B, T, n_groups, n) * 0.1

        ref_out, ref_h = _reference_ssm_scan(
            x, dt, A, B_ssm, C_ssm, dt_bias=dt_bias, dt_softplus=True
        )

        T2 = T // 2

        out_0, h_0 = _reference_ssm_scan(
            x[:, :T2],
            dt[:, :T2],
            A,
            B_ssm[:, :T2],
            C_ssm[:, :T2],
            dt_bias=dt_bias,
            dt_softplus=True,
        )
        out_1, h_1 = _reference_ssm_scan(
            x[:, T2:],
            dt[:, T2:],
            A,
            B_ssm[:, T2:],
            C_ssm[:, T2:],
            dt_bias=dt_bias,
            dt_softplus=True,
        )

        dt_eff_1 = F.softplus(dt[:, T2:] + dt_bias[None, None, :])
        cum_A_1 = torch.cumsum(A[None, None, :] * dt_eff_1, dim=1)

        corrected_out_1, corrected_h_1 = mamba2_cp_correction(
            out_1.view(B, T2, H * d),
            h_1,
            C_ssm[:, T2:],
            cum_A_1,
            h_0,
            num_heads=H,
            head_dim=d,
        )

        reconstructed = torch.cat([out_0, corrected_out_1.view(B, T2, H, d)], dim=1)

        torch.testing.assert_close(reconstructed, ref_out, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(corrected_h_1, ref_h, rtol=1e-4, atol=1e-4)

    def test_4_way_split(self):
        """4-rank CP: split 1024 tokens into 4×256 chunks with sequential correction."""
        torch.manual_seed(99)
        B, T, H, d, n = 1, 1024, 2, 8, 4
        n_groups = 1
        n_ranks = 4
        chunk = T // n_ranks
        dt_bias = torch.randn(H) * 0.05

        x = torch.randn(B, T, H, d)
        dt = torch.randn(B, T, H) * 0.1
        A = -torch.rand(H).abs() - 0.01
        B_ssm = torch.randn(B, T, n_groups, n) * 0.1
        C_ssm = torch.randn(B, T, n_groups, n) * 0.1

        ref_out, ref_h = _reference_ssm_scan(
            x, dt, A, B_ssm, C_ssm, dt_bias=dt_bias, dt_softplus=True
        )

        all_outs = []
        h_prev = torch.zeros(B, H, d, n)

        for rank in range(n_ranks):
            s, e = rank * chunk, (rank + 1) * chunk
            out_r, h_r = _reference_ssm_scan(
                x[:, s:e],
                dt[:, s:e],
                A,
                B_ssm[:, s:e],
                C_ssm[:, s:e],
                dt_bias=dt_bias,
                dt_softplus=True,
            )

            dt_eff_r = F.softplus(dt[:, s:e] + dt_bias[None, None, :])
            cum_A_r = torch.cumsum(A[None, None, :] * dt_eff_r, dim=1)

            corrected_out_r, corrected_h_r = mamba2_cp_correction(
                out_r.view(B, chunk, H * d),
                h_r,
                C_ssm[:, s:e],
                cum_A_r,
                h_prev,
                num_heads=H,
                head_dim=d,
            )

            all_outs.append(corrected_out_r.view(B, chunk, H, d))
            h_prev = corrected_h_r

        reconstructed = torch.cat(all_outs, dim=1)

        torch.testing.assert_close(reconstructed, ref_out, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(h_prev, ref_h, rtol=1e-3, atol=1e-3)


class TestWrapMambaScanForCp:
    """Tests for wrap_mamba_scan_for_cp wrapper."""

    @staticmethod
    def _make_module_with_scan(scan_fn):
        """Create a module namespace with a mamba_chunk_scan_combined attribute."""
        mod = types.ModuleType("fake_mamba_module")
        mod.mamba_chunk_scan_combined = scan_fn
        return mod

    def test_passthrough_when_cp_inactive(self):
        """When CP is not active, wrapper should return original result unchanged."""
        B, T, H, d, n = 1, 8, 4, 16, 8
        x = torch.randn(B, T, H, d)
        dt = torch.randn(B, T, H)
        A = -torch.rand(H)
        B_arg = torch.randn(B, T, 2, n)
        C_arg = torch.randn(B, T, 2, n)
        expected_out = torch.randn(B, T, H, d)
        expected_state = torch.randn(B, H, d, n)

        def fake_scan(*args, **kwargs):
            return expected_out, expected_state

        mod = self._make_module_with_scan(fake_scan)

        with patch(
            "axolotl.monkeypatch.models.mamba_utils.is_cp_active", return_value=False
        ):
            wrap_mamba_scan_for_cp(mod)
            out, state = mod.mamba_chunk_scan_combined(
                x,
                dt,
                A,
                B_arg,
                C_arg,
                chunk_size=64,
                return_final_states=True,
                dt_bias=None,
                dt_softplus=False,
            )

        torch.testing.assert_close(out, expected_out)
        torch.testing.assert_close(state, expected_state)

    def test_forces_return_final_states_when_cp_active(self):
        """When CP is active, wrapper must set return_final_states=True."""
        B, T, H, d, n = 1, 4, 2, 8, 4
        captured_kwargs = {}

        def fake_scan(*args, **kwargs):
            captured_kwargs.update(kwargs)
            scan_out = torch.zeros(B, T, H, d)
            ssm_state = torch.zeros(B, H, d, n)
            return scan_out, ssm_state

        mod = self._make_module_with_scan(fake_scan)

        with (
            patch(
                "axolotl.monkeypatch.models.mamba_utils.is_cp_active", return_value=True
            ),
            patch(
                "axolotl.monkeypatch.models.mamba_utils.ring_shift_ssm_state",
                side_effect=lambda h: torch.zeros_like(h),
            ),
        ):
            wrap_mamba_scan_for_cp(mod)
            mod.mamba_chunk_scan_combined(
                torch.zeros(B, T, H, d),
                torch.zeros(B, T, H),
                -torch.ones(H),
                torch.zeros(B, T, 1, n),
                torch.zeros(B, T, 1, n),
                chunk_size=64,
                return_final_states=False,
                dt_bias=None,
                dt_softplus=False,
            )

        assert captured_kwargs["return_final_states"] is True

    def test_idempotency_guard(self):
        """Calling wrap_mamba_scan_for_cp twice must not double-wrap."""
        call_count = 0

        def fake_scan(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            B, T, H, d, n = 1, 4, 2, 8, 4
            return torch.zeros(B, T, H, d), torch.zeros(B, H, d, n)

        mod = self._make_module_with_scan(fake_scan)

        with patch(
            "axolotl.monkeypatch.models.mamba_utils.is_cp_active", return_value=False
        ):
            wrap_mamba_scan_for_cp(mod)
            first_fn = mod.mamba_chunk_scan_combined
            wrap_mamba_scan_for_cp(mod)
            assert mod.mamba_chunk_scan_combined is first_fn
            assert getattr(mod, "_cp_scan_wrapped", False) is True
