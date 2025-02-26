"""
benchmark utility helper for benchmarking the KL divergence triton kernel
"""
import gc
import time

import torch
from torch.utils.benchmark import Timer

from axolotl.integrations.kd.topk_logprob.forward_kl import loss as eager_loss
from axolotl.integrations.kd.topk_logprob.forward_kl_triton import loss as triton_loss


# pylint: disable=cell-var-from-loop
def benchmark_kl_div_loss_with_backward():
    # Test configurations
    batch_sizes = [1, 4]
    seq_lens = [64, 512, 2048, 4096, 8192]
    vocab_size = 32000
    top_k = 64

    # Store results
    results = []

    # Run benchmarks
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            # Generate random test data
            torch.manual_seed(42)

            # Create tensors with gradients
            student_logits = torch.randn(
                batch_size, seq_len, vocab_size, device="cuda", requires_grad=True
            )
            # pylint: disable=duplicate-code
            target_token_ids = torch.randint(
                0, vocab_size, (batch_size, seq_len, top_k), device="cuda"
            )
            target_logprobs_raw = torch.randn(batch_size, seq_len, top_k, device="cuda")
            target_logprobs = torch.log_softmax(target_logprobs_raw, dim=-1)
            target_mask = torch.randint(
                0, 2, (batch_size, seq_len, top_k), device="cuda"
            ).float()

            # Clone student_logits for the two implementations
            student_logits_ref = student_logits.clone().detach().requires_grad_(True)
            student_logits_triton = student_logits.clone().detach().requires_grad_(True)

            # Define functions for timing that include both forward and backward passes
            def run_reference():
                # Forward pass
                loss_ref = eager_loss(
                    student_logits_ref, target_token_ids, target_logprobs, target_mask
                )
                # Backward pass
                loss_ref.backward()

            def run_triton():
                # Forward pass
                # pylint: disable=duplicate-code
                loss_triton = triton_loss(
                    student_logits_triton,
                    target_token_ids,
                    target_logprobs,
                    target_mask,
                )
                # Backward pass
                loss_triton.backward()

            # Benchmark reference implementation (forward + backward)
            t0 = Timer(
                stmt="run_reference()",
                globals={
                    "run_reference": run_reference,
                },
            )
            # Reset gradients before timing
            student_logits_ref.grad = None
            ref_time = t0.timeit(10).median * 1000  # Convert to ms

            # Benchmark Triton implementation (forward + backward)
            t1 = Timer(
                stmt="run_triton()",
                globals={
                    "run_triton": run_triton,
                },
            )
            # Reset gradients before timing
            student_logits_triton.grad = None
            triton_time = t1.timeit(10).median * 1000  # Convert to ms

            # Compute speedup
            speedup = ref_time / triton_time if triton_time > 0 else float("inf")

            # Store results
            results.append(
                {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "reference_time_ms": ref_time,
                    "triton_time_ms": triton_time,
                    "speedup": speedup,
                }
            )

            print(f"Batch size: {batch_size}, Seq len: {seq_len}")
            print(f"  Reference time (fwd+bwd): {ref_time:.2f} ms")
            print(f"  Triton time (fwd+bwd): {triton_time:.2f} ms")
            print(f"  Speedup: {speedup:.2f}x")

    return results


def benchmark_forward_backward_separately():
    """
    Benchmark forward and backward passes separately to identify where the speedup comes from.
    """
    # Test configurations
    batch_sizes = [1, 4, 8]
    seq_lens = [64, 512, 2048]
    vocab_size = 32000
    top_k = 64

    # Store results
    detailed_results = []

    # Run benchmarks
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            # Generate random test data
            torch.manual_seed(42)

            # Create tensors with gradients
            student_logits = torch.randn(
                batch_size, seq_len, vocab_size, device="cuda", requires_grad=True
            )
            # pylint: disable=duplicate-code
            target_token_ids = torch.randint(
                0, vocab_size, (batch_size, seq_len, top_k), device="cuda"
            )
            target_logprobs_raw = torch.randn(batch_size, seq_len, top_k, device="cuda")
            target_logprobs = torch.log_softmax(target_logprobs_raw, dim=-1)
            target_mask = torch.randint(
                0, 2, (batch_size, seq_len, top_k), device="cuda"
            ).float()

            # Clone student_logits for the two implementations
            student_logits_ref = student_logits.clone().detach().requires_grad_(True)
            student_logits_triton = student_logits.clone().detach().requires_grad_(True)

            # Forward-only reference
            def run_reference_forward():
                with torch.no_grad():
                    return eager_loss(
                        student_logits_ref,
                        target_token_ids,
                        target_logprobs,
                        target_mask,
                    )

            # Forward-only triton
            def run_triton_forward():
                with torch.no_grad():
                    return triton_loss(
                        student_logits_triton,
                        target_token_ids,
                        target_logprobs,
                        target_mask,
                    )

            # Benchmark forward pass only

            t0_fwd = Timer(
                stmt="run_reference_forward()",
                globals={
                    "run_reference_forward": run_reference_forward,
                },
            )
            ref_fwd_time = t0_fwd.timeit(10).median * 1000  # Convert to ms

            t1_fwd = Timer(
                stmt="run_triton_forward()",
                globals={
                    "run_triton_forward": run_triton_forward,
                },
            )
            triton_fwd_time = t1_fwd.timeit(10).median * 1000  # Convert to ms

            # Pre-compute losses for backward pass benchmarking
            loss_ref = eager_loss(
                student_logits_ref, target_token_ids, target_logprobs, target_mask
            )
            loss_triton = triton_loss(
                student_logits_triton, target_token_ids, target_logprobs, target_mask
            )

            # Backward-only reference
            def run_reference_backward():
                student_logits_ref.grad = None
                loss_ref.backward()

            # Backward-only triton
            def run_triton_backward():
                student_logits_triton.grad = None
                loss_triton.backward()

            # Benchmark backward pass only
            t0_bwd = Timer(
                stmt="run_reference_backward()",
                globals={
                    "run_reference_backward": run_reference_backward,
                },
            )
            ref_bwd_time = t0_bwd.timeit(10).median * 1000  # Convert to ms

            t1_bwd = Timer(
                stmt="run_triton_backward()",
                globals={
                    "run_triton_backward": run_triton_backward,
                },
            )
            triton_bwd_time = t1_bwd.timeit(10).median * 1000  # Convert to ms

            # Compute speedups
            fwd_speedup = (
                ref_fwd_time / triton_fwd_time if triton_fwd_time > 0 else float("inf")
            )
            bwd_speedup = (
                ref_bwd_time / triton_bwd_time if triton_bwd_time > 0 else float("inf")
            )
            total_ref_time = ref_fwd_time + ref_bwd_time
            total_triton_time = triton_fwd_time + triton_bwd_time
            total_speedup = (
                total_ref_time / total_triton_time
                if total_triton_time > 0
                else float("inf")
            )

            # Store results
            detailed_results.append(
                {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "ref_forward_ms": ref_fwd_time,
                    "triton_forward_ms": triton_fwd_time,
                    "forward_speedup": fwd_speedup,
                    "ref_backward_ms": ref_bwd_time,
                    "triton_backward_ms": triton_bwd_time,
                    "backward_speedup": bwd_speedup,
                    "total_ref_ms": total_ref_time,
                    "total_triton_ms": total_triton_time,
                    "total_speedup": total_speedup,
                }
            )

            print(f"Batch size: {batch_size}, Seq len: {seq_len}")
            print(
                f"  Forward: Reference={ref_fwd_time:.2f}ms, Triton={triton_fwd_time:.2f}ms, Speedup={fwd_speedup:.2f}x"
            )
            print(
                f"  Backward: Reference={ref_bwd_time:.2f}ms, Triton={triton_bwd_time:.2f}ms, Speedup={bwd_speedup:.2f}x"
            )
            print(
                f"  Total: Reference={total_ref_time:.2f}ms, Triton={total_triton_time:.2f}ms, Speedup={total_speedup:.2f}x"
            )

    return detailed_results


def benchmark_memory_usage_with_backward():
    # Test configurations
    batch_sizes = [1, 2]
    seq_len = 8192
    vocab_size = 128000
    top_k = 64

    # Store results
    mem_results = []

    # Run benchmarks
    for batch_size in batch_sizes:
        # Generate random test data
        torch.manual_seed(42)
        student_logits = torch.randn(
            batch_size, seq_len, vocab_size, device="cuda", requires_grad=True
        )
        target_token_ids = torch.randint(
            0, vocab_size, (batch_size, seq_len, top_k), device="cuda"
        )
        target_logprobs_raw = torch.randn(batch_size, seq_len, top_k, device="cuda")
        target_logprobs = torch.log_softmax(target_logprobs_raw, dim=-1)
        target_mask = torch.randint(
            0, 2, (batch_size, seq_len, top_k), device="cuda"
        ).float()

        # Clone student_logits for the implementations
        student_logits_ref = student_logits.clone().detach().requires_grad_(True)
        student_logits_triton = student_logits.clone().detach().requires_grad_(True)

        # Measure PyTorch memory usage (forward + backward)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        loss_ref = eager_loss(
            student_logits_ref, target_token_ids, target_logprobs, target_mask
        )
        loss_ref.backward()
        torch.cuda.synchronize()
        pytorch_mem = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB

        # Measure Triton memory usage (forward + backward)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        loss_triton = triton_loss(
            student_logits_triton, target_token_ids, target_logprobs, target_mask
        )
        loss_triton.backward()
        torch.cuda.synchronize()
        triton_mem = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB

        # Measure Triton memory usage with different chunk sizes (forward + backward)
        for n_chunks in [1, 2, 4, 8]:
            student_logits_chunk = student_logits.clone().detach().requires_grad_(True)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            loss_chunk = triton_loss(
                student_logits_chunk,
                target_token_ids,
                target_logprobs,
                target_mask,
            )
            loss_chunk.backward()
            torch.cuda.synchronize()
            chunk_mem = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB

            mem_results.append(
                {
                    "batch_size": batch_size,
                    "implementation": f"Triton (chunks={n_chunks})",
                    "memory_mb": chunk_mem,
                }
            )

        # Store results
        mem_results.append(
            {
                "batch_size": batch_size,
                "implementation": "PyTorch",
                "memory_mb": pytorch_mem,
            }
        )

        mem_results.append(
            {
                "batch_size": batch_size,
                "implementation": "Triton (default)",
                "memory_mb": triton_mem,
            }
        )

        print(f"Batch size: {batch_size} (with backward pass)")
        print(f"  PyTorch memory: {pytorch_mem:.2f} MB")
        print(f"  Triton memory: {triton_mem:.2f} MB")
        print(f"  Memory reduction: {(1 - triton_mem/pytorch_mem)*100:.2f}%")

    return mem_results


def main():
    print("Running benchmarks with forward and backward passes...")
    benchmark_kl_div_loss_with_backward()
    clean()

    print("\nRunning detailed forward/backward benchmarks...")
    # benchmark_forward_backward_separately()
    # clean()

    print("\nRunning memory usage benchmarks with backward passes...")
    benchmark_memory_usage_with_backward()
    clean()


def clean():
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)


if __name__ == "__main__":
    main()
