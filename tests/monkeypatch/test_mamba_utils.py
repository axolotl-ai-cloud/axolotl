"""Unit tests for shared Mamba2 SSM utilities (mamba_utils.py).

Tests cover get_seq_idx correctness under:
  - single-rank packing
  - context parallelism (mid-sample chunk starts)
  - batch dimension
  - dtype and device
  - no-negative regression (CP rank > 0 must never produce -1)
  - mamba2_cp_correction mathematical correctness
  - wrap_mamba_scan_for_cp wrapper behaviour
"""

import types
from unittest.mock import patch

import torch

from axolotl.monkeypatch.models.mamba_utils import (
    get_seq_idx,
    mamba2_cp_correction,
    wrap_mamba_scan_for_cp,
)


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
