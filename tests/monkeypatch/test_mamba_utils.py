"""Unit tests for shared Mamba2 SSM utilities (mamba_utils.py).

Tests cover get_seq_idx correctness under:
  - single-rank packing
  - context parallelism (mid-sample chunk starts)
  - batch dimension
  - dtype and device
  - no-negative regression (CP rank > 0 must never produce -1)
"""

import torch

from axolotl.monkeypatch.models.mamba_utils import get_seq_idx


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
