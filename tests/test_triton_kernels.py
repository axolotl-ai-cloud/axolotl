# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for Triton kernels: entropy_from_logits and selective_log_softmax.

Adapted from harness/test_entropy.py and harness/test_selective_logsoftmax.py
into proper pytest tests, plus new OOB index safety tests.
"""

import math

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernels"
)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _ref_entropy(logits):
    """Reference entropy via log_softmax (numerically stable)."""
    logp = F.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


def _ref_selective_log_softmax(logits, index):
    """Reference selective log softmax via PyTorch gather."""
    squeeze = index.ndim == logits.ndim - 1
    if squeeze:
        index = index.unsqueeze(-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    result = torch.gather(log_probs, dim=-1, index=index)
    if squeeze:
        result = result.squeeze(-1)
    return result


# ---------------------------------------------------------------------------
# entropy_from_logits
# ---------------------------------------------------------------------------


class TestEntropyFromLogits:
    @pytest.mark.parametrize(
        "B,L",
        [
            (1, 128),
            (1, 2048),
            (4, 512),
            (8, 256),
            (1, 1),
        ],
    )
    def test_correctness_various_shapes(self, B, L):
        from axolotl.monkeypatch.trainer.utils import entropy_from_logits

        V = 1024
        torch.manual_seed(42)
        logits = torch.randn(B, L, V, device="cuda", dtype=torch.float32)
        result = entropy_from_logits(logits)
        expected = _ref_entropy(logits)
        assert result.shape == (B, L)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_2d_input(self):
        from axolotl.monkeypatch.trainer.utils import entropy_from_logits

        logits = torch.randn(16, 256, device="cuda", dtype=torch.float32)
        result = entropy_from_logits(logits)
        expected = _ref_entropy(logits)
        assert result.shape == (16,)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_large_vocab(self):
        from axolotl.monkeypatch.trainer.utils import entropy_from_logits

        V = 32000
        logits = torch.randn(2, V, device="cuda", dtype=torch.float32)
        result = entropy_from_logits(logits)
        expected = _ref_entropy(logits)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_uniform_distribution(self):
        """Uniform logits -> entropy = log(V)."""
        from axolotl.monkeypatch.trainer.utils import entropy_from_logits

        V = 1024
        logits = torch.zeros(2, V, device="cuda", dtype=torch.float32)
        result = entropy_from_logits(logits)
        expected_val = math.log(V)
        torch.testing.assert_close(
            result,
            torch.full((2,), expected_val, device="cuda", dtype=torch.float32),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_peaked_distribution(self):
        """One-hot-like logits -> entropy near 0."""
        from axolotl.monkeypatch.trainer.utils import entropy_from_logits

        logits = torch.full((2, 128), -100.0, device="cuda", dtype=torch.float32)
        logits[:, 0] = 100.0
        result = entropy_from_logits(logits)
        assert (result < 1e-3).all()

    def test_bfloat16(self):
        from axolotl.monkeypatch.trainer.utils import entropy_from_logits

        logits = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
        result = entropy_from_logits(logits)
        expected = _ref_entropy(logits.float())
        assert result.dtype == torch.bfloat16
        torch.testing.assert_close(result.float(), expected, atol=5e-2, rtol=5e-2)

    def test_float16(self):
        from axolotl.monkeypatch.trainer.utils import entropy_from_logits

        logits = torch.randn(4, 256, device="cuda", dtype=torch.float16)
        result = entropy_from_logits(logits)
        expected = _ref_entropy(logits.float())
        assert result.dtype == torch.float16
        torch.testing.assert_close(result.float(), expected, atol=5e-2, rtol=5e-2)

    def test_non_contiguous_3d_transpose(self):
        """Non-contiguous 3D tensor via transpose(0,1)."""
        from axolotl.monkeypatch.trainer.utils import entropy_from_logits

        V = 256
        raw = torch.randn(32, 4, V, device="cuda", dtype=torch.float32)
        logits = raw.transpose(0, 1)  # (4, 32, V) non-contiguous
        assert not logits.is_contiguous()
        result = entropy_from_logits(logits)
        expected = _ref_entropy(logits)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_non_contiguous_3d_slice(self):
        """Non-contiguous 3D tensor via batch slicing."""
        from axolotl.monkeypatch.trainer.utils import entropy_from_logits

        V = 256
        raw = torch.randn(8, 32, V, device="cuda", dtype=torch.float32)
        logits = raw[::2]  # (4, 32, V) non-contiguous
        assert not logits.is_contiguous()
        result = entropy_from_logits(logits)
        expected = _ref_entropy(logits)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_many_rows_beyond_max_grid(self):
        """More rows than MAX_GRID (8192) to test chunked dispatch."""
        from axolotl.monkeypatch.trainer.utils import entropy_from_logits

        logits = torch.randn(10000, 128, device="cuda", dtype=torch.float32)
        result = entropy_from_logits(logits)
        expected = _ref_entropy(logits)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_entropy_non_negative(self):
        from axolotl.monkeypatch.trainer.utils import entropy_from_logits

        logits = torch.randn(32, 512, device="cuda", dtype=torch.float32)
        result = entropy_from_logits(logits)
        assert (result >= -1e-5).all(), f"Negative entropy: {result.min()}"


# ---------------------------------------------------------------------------
# selective_log_softmax — forward correctness
# ---------------------------------------------------------------------------


class TestSelectiveLogSoftmax:
    @pytest.mark.parametrize(
        "B,L,K",
        [
            (1, 128, 1),
            (4, 512, 1),
            (8, 256, 1),
            (4, 256, 4),
            (4, 256, 7),
            (15, 129, 1),  # non-power-of-2
        ],
    )
    def test_correctness_various_shapes(self, B, L, K):
        from axolotl.monkeypatch.trainer.utils import selective_log_softmax

        V = 1024
        torch.manual_seed(42)
        logits = torch.randn(B, L, V, device="cuda", dtype=torch.float32)
        if K == 1:
            index = torch.randint(0, V, (B, L), device="cuda")
        else:
            index = torch.randint(0, V, (B, L, K), device="cuda")
        result = selective_log_softmax(logits, index)
        expected = _ref_selective_log_softmax(logits, index)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_squeezed_index(self):
        """Index with ndim == logits.ndim - 1 triggers squeeze path."""
        from axolotl.monkeypatch.trainer.utils import selective_log_softmax

        V = 256
        logits = torch.randn(8, V, device="cuda", dtype=torch.float32)
        index = torch.randint(0, V, (8,), device="cuda")
        result = selective_log_softmax(logits, index)
        expected = _ref_selective_log_softmax(logits, index)
        assert result.shape == (8,)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_large_vocab(self):
        from axolotl.monkeypatch.trainer.utils import selective_log_softmax

        V = 32000
        logits = torch.randn(2, V, device="cuda", dtype=torch.float32)
        index = torch.randint(0, V, (2, 1), device="cuda")
        result = selective_log_softmax(logits, index)
        expected = _ref_selective_log_softmax(logits, index)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_bfloat16(self):
        from axolotl.monkeypatch.trainer.utils import selective_log_softmax

        V = 1024
        torch.manual_seed(42)
        logits = torch.randn(4, 128, V, device="cuda", dtype=torch.bfloat16)
        index = torch.randint(0, V, (4, 128), device="cuda")
        result = selective_log_softmax(logits, index)
        expected = _ref_selective_log_softmax(logits.float(), index)
        assert result.dtype == torch.bfloat16
        torch.testing.assert_close(result.float(), expected, atol=0.1, rtol=0.1)

    def test_fp32_tight_tolerance(self):
        from axolotl.monkeypatch.trainer.utils import selective_log_softmax

        V = 1024
        torch.manual_seed(42)
        logits = torch.randn(2, 256, V, device="cuda", dtype=torch.float32)
        index = torch.randint(0, V, (2, 256), device="cuda")
        result = selective_log_softmax(logits, index)
        expected = _ref_selective_log_softmax(logits, index)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_all_same_index(self):
        from axolotl.monkeypatch.trainer.utils import selective_log_softmax

        V = 128
        logits = torch.randn(8, V, device="cuda", dtype=torch.float32)
        index = torch.zeros(8, 1, device="cuda", dtype=torch.long)
        result = selective_log_softmax(logits, index)
        expected = _ref_selective_log_softmax(logits, index)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_last_index(self):
        from axolotl.monkeypatch.trainer.utils import selective_log_softmax

        V = 128
        logits = torch.randn(8, V, device="cuda", dtype=torch.float32)
        index = torch.full((8, 1), V - 1, device="cuda", dtype=torch.long)
        result = selective_log_softmax(logits, index)
        expected = _ref_selective_log_softmax(logits, index)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_output_always_nonpositive(self):
        """Log softmax values should always be <= 0."""
        from axolotl.monkeypatch.trainer.utils import selective_log_softmax

        V = 256
        logits = torch.randn(32, V, device="cuda", dtype=torch.float32)
        index = torch.randint(0, V, (32, 1), device="cuda")
        result = selective_log_softmax(logits, index)
        assert (result <= 1e-5).all(), f"Positive log-prob: {result.max()}"

    def test_many_rows_beyond_max_grid(self):
        from axolotl.monkeypatch.trainer.utils import selective_log_softmax

        V = 128
        logits = torch.randn(10000, V, device="cuda", dtype=torch.float32)
        index = torch.randint(0, V, (10000, 1), device="cuda")
        result = selective_log_softmax(logits, index)
        expected = _ref_selective_log_softmax(logits, index)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# selective_log_softmax — backward / gradient correctness
# ---------------------------------------------------------------------------


class TestSelectiveLogSoftmaxBackward:
    @pytest.mark.parametrize(
        "B,L,V,K",
        [
            (2, 16, 64, 1),
            (2, 16, 64, 4),
            (1, 8, 128, 1),
            (2, 8, 128, 7),
        ],
    )
    def test_gradient_matches_reference(self, B, L, V, K):
        from axolotl.monkeypatch.trainer.utils import selective_log_softmax

        torch.manual_seed(42)
        logits_ref = torch.randn(
            B, L, V, device="cuda", dtype=torch.float32, requires_grad=True
        )
        logits_tri = logits_ref.detach().clone().requires_grad_(True)

        if K == 1:
            index = torch.randint(0, V, (B, L), device="cuda")
        else:
            index = torch.randint(0, V, (B, L, K), device="cuda")

        ref_out = _ref_selective_log_softmax(logits_ref, index)
        tri_out = selective_log_softmax(logits_tri, index)

        ref_out.sum().backward()
        tri_out.sum().backward()

        torch.testing.assert_close(
            logits_tri.grad, logits_ref.grad, atol=1e-5, rtol=1e-5
        )

    def test_gradient_bfloat16_full_vocab(self):
        from axolotl.monkeypatch.trainer.utils import selective_log_softmax

        V = 4096
        torch.manual_seed(42)
        logits_ref = torch.randn(
            2, 64, V, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        logits_tri = logits_ref.detach().clone().requires_grad_(True)
        index = torch.randint(0, V, (2, 64), device="cuda")

        _ref_selective_log_softmax(logits_ref, index).sum().backward()
        selective_log_softmax(logits_tri, index).sum().backward()

        torch.testing.assert_close(
            logits_tri.grad.float(), logits_ref.grad.float(), atol=0.1, rtol=0.1
        )

    def test_gradient_k1_squeezed(self):
        """Gradient with squeezed (1D) index."""
        from axolotl.monkeypatch.trainer.utils import selective_log_softmax

        V = 256
        logits = torch.randn(
            8, V, device="cuda", dtype=torch.float32, requires_grad=True
        )
        index = torch.randint(0, V, (8,), device="cuda")

        result = selective_log_softmax(logits, index)
        result.sum().backward()
        triton_grad = logits.grad.clone()

        logits.grad = None
        ref = torch.gather(
            F.log_softmax(logits, dim=-1), dim=-1, index=index.unsqueeze(-1)
        ).squeeze(-1)
        ref.sum().backward()

        torch.testing.assert_close(triton_grad, logits.grad, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# selective_log_softmax — out-of-bounds index safety
# ---------------------------------------------------------------------------


class TestSelectiveLogSoftmaxOOBSafety:
    """Verify that out-of-range indices don't crash or corrupt valid results."""

    def test_negative_indices_no_crash(self):
        from axolotl.monkeypatch.trainer.utils import selective_log_softmax

        V = 128
        logits = torch.randn(4, V, device="cuda", dtype=torch.float32)
        index = torch.tensor(
            [[-1], [0], [V - 1], [-5]], device="cuda", dtype=torch.long
        )
        result = selective_log_softmax(logits, index)
        assert result.shape == (4, 1)
        # Valid rows should be finite and match reference
        valid_idx = torch.tensor([[0], [V - 1]], device="cuda", dtype=torch.long)
        valid_logits = logits[1:3]
        expected = _ref_selective_log_softmax(valid_logits, valid_idx)
        torch.testing.assert_close(result[1:3], expected, atol=1e-4, rtol=1e-4)

    def test_index_exceeds_vocab_no_crash(self):
        from axolotl.monkeypatch.trainer.utils import selective_log_softmax

        V = 128
        logits = torch.randn(4, V, device="cuda", dtype=torch.float32)
        index = torch.tensor(
            [[0], [V], [V + 100], [V - 1]], device="cuda", dtype=torch.long
        )
        result = selective_log_softmax(logits, index)
        assert result.shape == (4, 1)
        # Valid rows (0 and 3) should match reference
        for row_idx, idx_val in [(0, 0), (3, V - 1)]:
            ref = _ref_selective_log_softmax(
                logits[row_idx : row_idx + 1],
                torch.tensor([[idx_val]], device="cuda", dtype=torch.long),
            )
            torch.testing.assert_close(
                result[row_idx : row_idx + 1], ref, atol=1e-4, rtol=1e-4
            )

    def test_mixed_valid_invalid_multi_index(self):
        from axolotl.monkeypatch.trainer.utils import selective_log_softmax

        V = 256
        K = 3
        logits = torch.randn(4, V, device="cuda", dtype=torch.float32)
        index = torch.tensor(
            [
                [0, 10, -1],  # last invalid
                [V, 5, 100],  # first invalid
                [50, 60, 70],  # all valid
                [-1, V + 1, -100],  # all invalid
            ],
            device="cuda",
            dtype=torch.long,
        )
        result = selective_log_softmax(logits, index)
        assert result.shape == (4, K)
        # Row 2 (all valid) must match reference exactly
        valid_index = torch.tensor([[50, 60, 70]], device="cuda", dtype=torch.long)
        expected = _ref_selective_log_softmax(logits[2:3], valid_index)
        torch.testing.assert_close(result[2:3], expected, atol=1e-4, rtol=1e-4)

    def test_oob_backward_no_crash(self):
        """Backward with OOB indices should not crash and grads should be finite."""
        from axolotl.monkeypatch.trainer.utils import selective_log_softmax

        V = 128
        logits = torch.randn(
            4, V, device="cuda", dtype=torch.float32, requires_grad=True
        )
        index = torch.tensor(
            [[-1], [0], [V + 10], [V - 1]], device="cuda", dtype=torch.long
        )
        result = selective_log_softmax(logits, index)
        result.sum().backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_oob_backward_valid_rows_correct(self):
        """Gradients for valid-index rows should match reference even when other rows have OOB."""
        from axolotl.monkeypatch.trainer.utils import selective_log_softmax

        V = 128
        logits = torch.randn(
            4, V, device="cuda", dtype=torch.float32, requires_grad=True
        )
        # Row 0: invalid, Row 1: valid, Row 2: invalid, Row 3: valid
        index = torch.tensor(
            [[-1], [42], [V + 5], [100]], device="cuda", dtype=torch.long
        )
        result = selective_log_softmax(logits, index)
        result.sum().backward()

        # Compute reference gradient for valid rows only
        logits_ref = logits.detach().clone().requires_grad_(True)
        valid_rows = [1, 3]
        valid_indices = [42, 100]
        for r, idx in zip(valid_rows, valid_indices, strict=True):
            ref_lp = F.log_softmax(logits_ref[r : r + 1], dim=-1)
            ref_val = ref_lp[0, idx]
            ref_val.backward(retain_graph=True)

        for r in valid_rows:
            torch.testing.assert_close(
                logits.grad[r], logits_ref.grad[r], atol=1e-4, rtol=1e-4
            )
