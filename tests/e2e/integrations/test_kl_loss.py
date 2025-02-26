"""
sanity checks on kl loss and gradients
"""
import torch

# Import both implementations
from axolotl.integrations.kd.topk_logprob.forward_kl import loss as eager_loss
from axolotl.integrations.kd.topk_logprob.forward_kl_triton import loss as triton_loss


def test_kl_loss_gradient():
    """Test that the gradient of the Triton implementation matches the eager implementation."""

    # Set the random seed for reproducibility
    torch.manual_seed(42)

    # Create random inputs
    batch_size = 2
    seq_len = 3
    vocab_size = 100
    top_k = 5

    # Generate random student logits
    student_logits = torch.randn(
        batch_size, seq_len, vocab_size, requires_grad=True, device="cuda"
    )
    student_logits_triton = student_logits.detach().clone().requires_grad_(True)

    # Generate random target token IDs, ensuring they're valid indices
    target_token_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len, top_k), device="cuda"
    )

    # Generate random target logprobs (before normalization)
    target_logprobs_raw = torch.randn(batch_size, seq_len, top_k, device="cuda")

    # Normalize the target logprobs to ensure they form a valid distribution
    target_logprobs = torch.log_softmax(target_logprobs_raw, dim=-1)

    # Create a random mask with some tokens masked out
    target_mask = torch.randint(
        0, 2, (batch_size, seq_len, top_k), device="cuda"
    ).float()

    # Additional parameters
    num_items_in_batch = batch_size * seq_len
    kd_temperature = 1.0
    top_k_before_softmax = 0  # Test both modes

    # Compute the loss and gradients with eager implementation
    loss_eager = eager_loss(
        student_logits,
        target_token_ids,
        target_logprobs,
        target_mask,
        num_items_in_batch,
        kd_temperature,
        top_k_before_softmax,
    )
    loss_eager.backward()
    grad_eager = student_logits.grad.clone()

    # Reset gradients
    student_logits.grad.zero_()

    # Compute the loss and gradients with Triton implementation
    loss_triton = triton_loss(
        student_logits_triton,
        target_token_ids,
        target_logprobs,
        target_mask,
        num_items_in_batch,
        kd_temperature,
        top_k_before_softmax,
    )
    loss_triton.backward()
    grad_triton = student_logits_triton.grad.clone()

    # Compare loss values
    print(f"Eager loss: {loss_eager.item()}")
    print(f"Triton loss: {loss_triton.item()}")
    loss_diff = abs(loss_eager.item() - loss_triton.item())
    print(f"Loss difference: {loss_diff}")
    assert loss_diff < 1e-5, "Loss values differ significantly!"

    # Compare gradients
    grad_diff = (grad_eager - grad_triton).abs().max().item()
    print(f"Max gradient difference: {grad_diff}")

    # Print some sample gradients
    sample_idx = (0, 0, 0)  # (batch, seq, vocab)
    print(f"Sample eager gradient: {grad_eager[sample_idx].item()}")
    print(f"Sample triton gradient: {grad_triton[sample_idx].item()}")

    # Compute relative difference for non-zero gradients
    mask = grad_eager.abs() > 1e-10
    if mask.sum() > 0:
        rel_diff = (
            (
                (grad_eager[mask] - grad_triton[mask]).abs()
                / (grad_eager[mask].abs() + 1e-10)
            )
            .max()
            .item()
        )
        print(f"Max relative gradient difference: {rel_diff}")
        assert rel_diff < 1e-3, "Gradients differ significantly!"

    # Also test top_k_before_softmax = 1 mode
    top_k_before_softmax = 1

    # Reset the gradients
    student_logits = torch.randn(
        batch_size, seq_len, vocab_size, requires_grad=True, device="cuda"
    )
    student_logits_triton = student_logits.detach().clone().requires_grad_(True)

    # Compute the loss and gradients with eager implementation
    loss_eager = eager_loss(
        student_logits,
        target_token_ids,
        target_logprobs,
        target_mask,
        num_items_in_batch,
        kd_temperature,
        top_k_before_softmax,
    )
    loss_eager.backward()
    grad_eager = student_logits.grad.clone()

    # Compute the loss and gradients with Triton implementation
    loss_triton = triton_loss(
        student_logits_triton,
        target_token_ids,
        target_logprobs,
        target_mask,
        num_items_in_batch,
        kd_temperature,
        top_k_before_softmax,
    )
    loss_triton.backward()
    grad_triton = student_logits_triton.grad.clone()

    # Compare gradients for top_k_before_softmax = 1
    grad_diff = (grad_eager - grad_triton).abs().max().item()
    print("\nWith top_k_before_softmax=1:")
    print(f"Max gradient difference: {grad_diff}")

    # Compute relative difference for non-zero gradients
    mask = grad_eager.abs() > 1e-10
    if mask.sum() > 0:
        rel_diff = (
            (
                (grad_eager[mask] - grad_triton[mask]).abs()
                / (grad_eager[mask].abs() + 1e-10)
            )
            .max()
            .item()
        )
        assert (
            rel_diff < 1e-3
        ), f"Gradients differ significantly, Max relative gradient difference: {rel_diff}"
