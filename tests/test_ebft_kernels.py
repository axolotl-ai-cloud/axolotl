"""Correctness tests for fused EBFT Triton kernels."""

import pytest
import torch
import torch.nn.functional as F

from axolotl.core.trainers.ebft.kernels import (
    fused_cosine_similarity,
    fused_diversity_penalty,
    fused_log_softmax_gather,
    fused_reinforce_loss,
)

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernels"
)

DEVICE = "cuda"


# ---------------------------------------------------------------------------
# 1. fused_log_softmax_gather
# ---------------------------------------------------------------------------
class TestFusedLogSoftmaxGather:
    def _reference(self, logits, labels):
        """PyTorch reference: log_softmax + gather."""
        lp = F.log_softmax(logits.float(), dim=-1)
        return lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    def test_basic_correctness(self):
        B, S, V = 2, 16, 1024
        logits = torch.randn(B, S, V, device=DEVICE, dtype=torch.bfloat16)
        labels = torch.randint(0, V, (B, S), device=DEVICE)

        ref = self._reference(logits, labels)
        out = fused_log_softmax_gather(logits, labels)

        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    def test_large_vocab(self):
        """Test with realistic vocab size (128K)."""
        B, S, V = 1, 8, 128256
        logits = torch.randn(B, S, V, device=DEVICE, dtype=torch.bfloat16)
        labels = torch.randint(0, V, (B, S), device=DEVICE)

        ref = self._reference(logits, labels)
        out = fused_log_softmax_gather(logits, labels)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_fp32_input(self):
        B, S, V = 2, 8, 512
        logits = torch.randn(B, S, V, device=DEVICE, dtype=torch.float32)
        labels = torch.randint(0, V, (B, S), device=DEVICE)

        ref = self._reference(logits, labels)
        out = fused_log_softmax_gather(logits, labels)

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_output_is_negative(self):
        """log_softmax values should always be <= 0."""
        B, S, V = 4, 32, 2048
        logits = torch.randn(B, S, V, device=DEVICE, dtype=torch.bfloat16)
        labels = torch.randint(0, V, (B, S), device=DEVICE)

        out = fused_log_softmax_gather(logits, labels)
        assert (out <= 1e-5).all(), "log_softmax values should be <= 0"

    def test_extreme_logits(self):
        """Test numerical stability with very large/small logits."""
        B, S, V = 1, 4, 256
        logits = torch.randn(B, S, V, device=DEVICE, dtype=torch.float32)
        logits[:, 0, :] = 1000.0  # very large
        logits[:, 1, :] = -1000.0  # very small
        logits[:, 2, 0] = 1000.0  # one hot-ish
        labels = torch.zeros(B, S, device=DEVICE, dtype=torch.long)

        ref = self._reference(logits, labels)
        out = fused_log_softmax_gather(logits, labels)

        assert torch.isfinite(out).all(), "Should handle extreme values"
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_2d_input(self):
        """Test with pre-flattened (N, V) input."""
        N, V = 64, 4096
        logits = torch.randn(N, V, device=DEVICE, dtype=torch.bfloat16)
        labels = torch.randint(0, V, (N,), device=DEVICE)

        ref = self._reference(logits.unsqueeze(0), labels.unsqueeze(0)).squeeze(0)
        out = fused_log_softmax_gather(logits, labels)

        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# 2. fused_reinforce_loss
# ---------------------------------------------------------------------------
class TestFusedReinforceLoss:
    def _reference(self, logps, advantages, mask):
        """PyTorch reference implementation."""
        loss_per_token = -logps * advantages
        return (loss_per_token * mask.float()).sum() / mask.float().sum().clamp(min=1)

    def test_basic_correctness(self):
        N = 1024
        logps = torch.randn(N, device=DEVICE, dtype=torch.float32)
        advs = torch.randn(N, device=DEVICE, dtype=torch.float32)
        mask = torch.randint(0, 2, (N,), device=DEVICE, dtype=torch.bool)

        ref = self._reference(logps, advs, mask)
        out = fused_reinforce_loss(logps, advs, mask)

        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_2d_input(self):
        """Test with (B, S) shaped inputs."""
        B, S = 4, 256
        logps = torch.randn(B, S, device=DEVICE, dtype=torch.float32)
        advs = torch.randn(B, S, device=DEVICE, dtype=torch.float32)
        mask = torch.randint(0, 2, (B, S), device=DEVICE, dtype=torch.bool)

        ref = self._reference(logps, advs, mask)
        out = fused_reinforce_loss(logps, advs, mask)

        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_all_masked(self):
        """All-zero mask should return 0."""
        N = 512
        logps = torch.randn(N, device=DEVICE, dtype=torch.float32)
        advs = torch.randn(N, device=DEVICE, dtype=torch.float32)
        mask = torch.zeros(N, device=DEVICE, dtype=torch.bool)

        out = fused_reinforce_loss(logps, advs, mask)
        assert out.item() == 0.0

    def test_all_unmasked(self):
        N = 512
        logps = torch.randn(N, device=DEVICE, dtype=torch.float32)
        advs = torch.randn(N, device=DEVICE, dtype=torch.float32)
        mask = torch.ones(N, device=DEVICE, dtype=torch.bool)

        ref = self._reference(logps, advs, mask)
        out = fused_reinforce_loss(logps, advs, mask)

        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_large(self):
        """Test with realistic size (4 * 3000 tokens)."""
        N = 12000
        logps = torch.randn(N, device=DEVICE, dtype=torch.float32)
        advs = torch.randn(N, device=DEVICE, dtype=torch.float32)
        mask = torch.randint(0, 2, (N,), device=DEVICE, dtype=torch.bool)

        ref = self._reference(logps, advs, mask)
        out = fused_reinforce_loss(logps, advs, mask)

        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# 3. fused_cosine_similarity
# ---------------------------------------------------------------------------
class TestFusedCosineSimilarity:
    def test_basic_correctness(self):
        N, D = 64, 256
        a = torch.randn(N, D, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(N, D, device=DEVICE, dtype=torch.bfloat16)

        ref = F.cosine_similarity(a.float(), b.float(), dim=-1)
        out = fused_cosine_similarity(a, b)

        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    def test_batched(self):
        """Test with (B, N, NB, D) shaped input."""
        B, N, NB, D = 2, 4, 16, 512
        a = torch.randn(B, N, NB, D, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(B, N, NB, D, device=DEVICE, dtype=torch.bfloat16)

        ref = F.cosine_similarity(a.float(), b.float(), dim=-1)
        out = fused_cosine_similarity(a, b)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_identical_vectors(self):
        """Identical vectors should give similarity = 1."""
        N, D = 16, 128
        a = torch.randn(N, D, device=DEVICE, dtype=torch.float32)

        out = fused_cosine_similarity(a, a)
        torch.testing.assert_close(
            out,
            torch.ones(N, device=DEVICE, dtype=torch.float32),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should give similarity = 0."""
        D = 128
        a = torch.zeros(1, D, device=DEVICE, dtype=torch.float32)
        b = torch.zeros(1, D, device=DEVICE, dtype=torch.float32)
        a[0, 0] = 1.0
        b[0, 1] = 1.0

        out = fused_cosine_similarity(a, b)
        assert abs(out.item()) < 1e-5

    def test_opposite_vectors(self):
        """Opposite vectors should give similarity = -1."""
        N, D = 8, 64
        a = torch.randn(N, D, device=DEVICE, dtype=torch.float32)
        out = fused_cosine_similarity(a, -a)
        torch.testing.assert_close(
            out,
            -torch.ones(N, device=DEVICE, dtype=torch.float32),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_large_dimension(self):
        """Test with large feature dimension (multi-layer concatenated features)."""
        N, D = 32, 4608  # 3 layers * 1536 hidden
        a = torch.randn(N, D, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(N, D, device=DEVICE, dtype=torch.bfloat16)

        ref = F.cosine_similarity(a.float(), b.float(), dim=-1)
        out = fused_cosine_similarity(a, b)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# 4. fused_diversity_penalty
# ---------------------------------------------------------------------------
class TestFusedDiversityPenalty:
    def _reference(self, embeddings):
        """PyTorch reference: bmm + mask diagonal + mean."""
        B, N, D = embeddings.shape
        sims = torch.bmm(embeddings.float(), embeddings.float().transpose(1, 2))
        eye = torch.eye(N, device=embeddings.device, dtype=torch.bool)
        sims = sims.masked_fill(eye.unsqueeze(0), 0.0)
        return sims.sum(dim=-1) / (N - 1)

    def test_basic_correctness(self):
        B, N, D = 4, 4, 256
        emb = torch.randn(B, N, D, device=DEVICE, dtype=torch.bfloat16)

        ref = self._reference(emb)
        out = fused_diversity_penalty(emb)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_two_samples(self):
        """With n=2, diversity = dot(a, b) for each."""
        B, D = 3, 128
        emb = torch.randn(B, 2, D, device=DEVICE, dtype=torch.float32)

        ref = self._reference(emb)
        out = fused_diversity_penalty(emb)

        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_identical_samples(self):
        """All identical samples should give max diversity."""
        B, N, D = 2, 4, 64
        vec = torch.randn(B, 1, D, device=DEVICE, dtype=torch.float32)
        emb = vec.expand(B, N, D).contiguous()

        out = fused_diversity_penalty(emb)
        # Should be ||vec||^2 for each (self-excluded mean of identical dot products)
        expected = (vec.squeeze(1) ** 2).sum(dim=-1, keepdim=True).expand(B, N)
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_large(self):
        """Test with realistic EBFT dimensions."""
        B, N, D = 1, 4, 4608  # multi-layer features
        emb = torch.randn(B, N, D, device=DEVICE, dtype=torch.bfloat16)

        ref = self._reference(emb)
        out = fused_diversity_penalty(emb)

        torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)

    def test_single_sample_returns_zeros(self):
        """N=1 should return zeros (no pairs), not garbage from uninitialized memory."""
        B, D = 3, 128
        emb = torch.randn(B, 1, D, device=DEVICE, dtype=torch.float32)
        out = fused_diversity_penalty(emb)
        assert (out == 0).all(), "N=1 diversity should be exactly zero"
