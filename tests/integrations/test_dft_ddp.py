"""
Tests for DFT compatibility with Distributed Data Parallelism (DDP).

This test suite verifies that DFT loss computation works correctly in a
multi-process DDP environment where each rank computes loss independently
and gradients are synced via All-Reduce.
"""

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.axolotl.integrations.dft.dft_utils import (
    apply_dft_weighting,
    compute_dft_loss,
    compute_per_token_cross_entropy,
    reduce_token_loss,
)


def setup_ddp(rank, world_size):
    """Initialize DDP process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize process group
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        rank=rank,
        world_size=world_size,
    )

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up DDP process group."""
    dist.destroy_process_group()


# Module-level test functions (must be picklable for multiprocessing)


def _test_loss_consistency(rank, world_size):
    """Compute DFT loss on same input across all ranks."""
    device = torch.device(f"cuda:{rank}")

    # Same input on all ranks (simulating identical batch after data distribution)
    torch.manual_seed(42)  # Same seed across ranks
    batch_size = 2
    seq_len = 10
    vocab_size = 100

    logits = torch.randn(
        batch_size, seq_len, vocab_size, device=device, requires_grad=True
    )
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Compute DFT loss
    loss = compute_dft_loss(logits, labels)

    return {
        "loss": loss.item(),
        "rank": rank,
    }


def _test_different_batches(rank, world_size):
    """Each rank processes different batch."""
    device = torch.device(f"cuda:{rank}")

    # Different input on each rank (realistic DDP scenario)
    torch.manual_seed(42 + rank)  # Different seed per rank
    batch_size = 2
    seq_len = 10
    vocab_size = 100

    logits = torch.randn(
        batch_size, seq_len, vocab_size, device=device, requires_grad=True
    )
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Compute DFT loss
    loss = compute_dft_loss(logits, labels)

    return {
        "loss": loss.item(),
        "rank": rank,
        "has_grad": logits.grad is None,  # Grad not computed yet
    }


def _test_chunked_ce(rank, world_size):
    """Test chunked CE with DDP."""
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42)
    batch_size = 2
    seq_len = 8
    vocab_size = 1000  # Large vocab to benefit from chunking
    chunk_size = 256

    logits = torch.randn(
        batch_size, seq_len, vocab_size, device=device, requires_grad=True
    )
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Compute with chunking
    per_token_loss, valid_mask = compute_per_token_cross_entropy(
        logits, labels, chunk_size=chunk_size
    )
    weighted_loss = apply_dft_weighting(per_token_loss)
    loss = reduce_token_loss(weighted_loss, valid_mask)

    return {
        "loss": loss.item(),
        "rank": rank,
    }


def _test_gradient_sync(rank, world_size):
    """Simulate gradient computation and verify they can be synced."""
    device = torch.device(f"cuda:{rank}")

    # Different data per rank
    torch.manual_seed(42 + rank)
    batch_size = 2
    seq_len = 10
    vocab_size = 100

    # Create a simple model
    model = torch.nn.Linear(vocab_size, vocab_size).to(device)

    # Generate logits through model
    input_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    logits = model(input_logits)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Compute DFT loss
    loss = compute_dft_loss(logits, labels)

    # Backward (this would trigger gradient sync in real DDP)
    loss.backward()

    # Check gradients exist
    has_grads = all(p.grad is not None for p in model.parameters())
    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )

    return {
        "loss": loss.item(),
        "has_grads": has_grads,
        "grad_norm": grad_norm,
        "rank": rank,
    }


def _test_with_padding(rank, world_size):
    """Test with padded sequences (ignore_index = -100)."""
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42 + rank)
    batch_size = 2
    seq_len = 10
    vocab_size = 100

    logits = torch.randn(
        batch_size, seq_len, vocab_size, device=device, requires_grad=True
    )
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Add padding to last 3 positions
    labels[:, -3:] = -100

    # Compute DFT loss
    loss = compute_dft_loss(logits, labels)

    return {
        "loss": loss.item(),
        "rank": rank,
    }


def ddp_test_worker(rank, world_size, test_fn_name, result_queue):
    """Worker function for DDP test.

    Args:
        rank: Process rank
        world_size: Total number of processes
        test_fn_name: Name of the test function to run (string)
        result_queue: Queue to store results
    """
    try:
        setup_ddp(rank, world_size)

        # Get test function by name
        test_fn = globals()[test_fn_name]

        result = test_fn(rank, world_size)
        result_queue.put((rank, result, None))
    except Exception:
        import traceback

        result_queue.put((rank, None, traceback.format_exc()))
    finally:
        cleanup_ddp()


class TestDFTDDPCompatibility(unittest.TestCase):
    """Test DFT compatibility with DDP."""

    def _run_ddp_test(self, test_fn_name, world_size=2):
        """Helper to run a test function in DDP mode.

        Args:
            test_fn_name: Name of module-level test function (string)
            world_size: Number of processes

        Returns:
            List of results from each rank
        """
        if not torch.cuda.is_available():
            self.skipTest("DDP test requires CUDA")

        if torch.cuda.device_count() < world_size:
            self.skipTest(
                f"DDP test requires {world_size} GPUs, found {torch.cuda.device_count()}"
            )

        # Use spawn to start processes
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        processes = []
        for rank in range(world_size):
            p = ctx.Process(
                target=ddp_test_worker,
                args=(rank, world_size, test_fn_name, result_queue),
            )
            p.start()
            processes.append(p)

        # Wait for all processes
        for p in processes:
            p.join()

        # Collect results
        results = []
        errors = []
        for _ in range(world_size):
            rank, result, error = result_queue.get()
            if error:
                errors.append(f"Rank {rank}:\n{error}")
            else:
                results.append((rank, result))

        if errors:
            self.fail("\n".join(errors))

        # Sort by rank
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def test_ddp_loss_consistency(self):
        """Test that DFT loss is identical across DDP ranks."""
        results = self._run_ddp_test("_test_loss_consistency", world_size=2)

        # Verify all ranks computed same loss
        loss_values = [r["loss"] for r in results]
        self.assertEqual(len(loss_values), 2)

        # All losses should be identical (same input, deterministic computation)
        self.assertAlmostEqual(loss_values[0], loss_values[1], places=5)

    def test_ddp_different_batches(self):
        """Test DFT with different data on each rank (real DDP scenario)."""
        results = self._run_ddp_test("_test_different_batches", world_size=2)

        # Each rank should compute different loss (different input)
        loss_values = [r["loss"] for r in results]
        self.assertEqual(len(loss_values), 2)
        self.assertNotEqual(loss_values[0], loss_values[1])

        # Both should be valid losses
        for loss in loss_values:
            self.assertGreater(loss, 0)
            self.assertTrue(torch.isfinite(torch.tensor(loss)))

    def test_ddp_with_chunked_ce(self):
        """Test DFT with chunked CE in DDP mode."""
        results = self._run_ddp_test("_test_chunked_ce", world_size=2)

        # Verify all ranks completed successfully
        self.assertEqual(len(results), 2)

        # Both should produce valid losses
        for result in results:
            self.assertGreater(result["loss"], 0)

    def test_ddp_gradient_sync_simulation(self):
        """Test that DFT loss allows gradient synchronization (simulation)."""
        results = self._run_ddp_test("_test_gradient_sync", world_size=2)

        # Verify both ranks computed gradients
        for result in results:
            self.assertTrue(
                result["has_grads"], f"Rank {result['rank']} missing gradients"
            )
            self.assertGreater(
                result["grad_norm"], 0, f"Rank {result['rank']} has zero grad norm"
            )

    def test_ddp_with_padding(self):
        """Test DFT with padded sequences in DDP mode."""
        results = self._run_ddp_test("_test_with_padding", world_size=2)

        # Both ranks should handle padding correctly
        for result in results:
            self.assertGreater(result["loss"], 0)
            self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))


if __name__ == "__main__":
    unittest.main()
