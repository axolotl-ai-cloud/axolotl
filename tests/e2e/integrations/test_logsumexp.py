"""
sanity checks on logsumexp kernel validity
"""
import torch
import triton

from axolotl.integrations.kd.topk_logprob.logsumexp import logsumexp_kernel


# PyTorch implementation of logsumexp for reference
def torch_logsumexp(logits):
    """PyTorch implementation of logsumexp over last dimension"""
    return torch.logsumexp(logits, dim=-1)


# Wrapper function for Triton logsumexp kernel
def triton_logsumexp(logits):
    """Triton implementation of logsumexp over last dimension"""
    B, S, V = logits.shape  # pylint: disable=invalid-name
    output = torch.empty((B, S), dtype=torch.float32, device=logits.device)

    grid = (B * S,)
    logsumexp_kernel[grid](
        logits.contiguous(),
        output,
        B,
        S,
        V,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        output.stride(0),
        output.stride(1),
        min(1024, triton.next_power_of_2(V)),
    )

    return output


class TritonLogSumExp(torch.autograd.Function):
    """
    Wrap a custom autograd function to use the Triton logsumexp for gradient testing
    """

    @staticmethod
    def forward(ctx, logits):
        B, S, V = logits.shape  # pylint: disable=invalid-name
        output = torch.empty((B, S), dtype=torch.float32, device=logits.device)

        # Save inputs for backward pass
        ctx.save_for_backward(logits)
        ctx.shape = logits.shape

        grid = (B * S,)
        logsumexp_kernel[grid](
            logits.contiguous(),
            output,
            B,
            S,
            V,
            logits.stride(0),
            logits.stride(1),
            logits.stride(2),
            output.stride(0),
            output.stride(1),
            min(1024, triton.next_power_of_2(V)),
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (logits,) = ctx.saved_tensors

        # For logsumexp, the gradient is softmax(input) * grad_output
        # First compute the logsumexp
        lse = TritonLogSumExp.apply(logits)

        # Compute softmax by exponentiating differences
        softmax_output = torch.exp(logits - lse.unsqueeze(-1))

        # Compute gradient of logsumexp by multiplying the softmax output by the gradient
        grad_input = softmax_output * grad_output.unsqueeze(-1)

        return grad_input


def test_logsumexp_values():
    """Test that the Triton logsumexp implementation matches PyTorch's"""
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test with various input shapes
    test_shapes = [
        (2, 3, 10),  # small vocab
        (4, 5, 100),  # medium vocab
        (2, 2, 32000),  # large vocab (typical for LLMs)
    ]

    for shape in test_shapes:
        # Create random input tensors
        logits = torch.randn(shape, device="cuda", requires_grad=False)

        # Compute logsumexp using both implementations
        torch_result = torch_logsumexp(logits)
        triton_result = triton_logsumexp(logits)

        # Compare results
        max_diff = (torch_result - triton_result).abs().max().item()
        print(f"Shape {shape}, Max diff: {max_diff}")

        # Assert that the results are very close
        assert max_diff < 1e-5, f"Results differ for shape {shape}: max diff {max_diff}"


def test_logsumexp_edge_cases():
    """Test edge cases for numerical stability"""
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Case 1: Very large values that might cause overflow
    logits_large = torch.ones(2, 3, 100, device="cuda") * 1000

    # Case 2: Very small values that might cause underflow
    logits_small = torch.ones(2, 3, 100, device="cuda") * -1000

    # Case 3: Mix of large and small values
    logits_mixed = torch.zeros(2, 3, 100, device="cuda")
    logits_mixed[:, :, 0] = 1000  # One very large value

    # Case 4: All identical values
    logits_identical = torch.ones(2, 3, 100, device="cuda") * 5

    # Case 5: Extreme values with NaN check
    logits_extreme = torch.cat(
        [
            torch.full((1, 3, 50), 1e10, device="cuda"),
            torch.full((1, 3, 50), -1e10, device="cuda"),
        ],
        dim=0,
    )

    for i, logits in enumerate(
        [logits_large, logits_small, logits_mixed, logits_identical, logits_extreme]
    ):
        # Compute logsumexp using both implementations
        torch_result = torch_logsumexp(logits)
        triton_result = triton_logsumexp(logits)

        # Check for NaNs
        assert not torch.isnan(
            torch_result
        ).any(), f"PyTorch produced NaNs for case {i+1}"
        assert not torch.isnan(
            triton_result
        ).any(), f"Triton produced NaNs for case {i+1}"

        # Compare results
        max_diff = (torch_result - triton_result).abs().max().item()
        print(f"Edge case {i+1}, Max diff: {max_diff}")

        # For very extreme values, allow a bit more tolerance
        if i == 4:  # extreme case
            assert max_diff < 1e-2, f"Results differ too much for edge case {i+1}"
        else:
            assert max_diff < 1e-5, f"Results differ too much for edge case {i+1}"


def test_logsumexp_gradients():
    """Test that the gradients of Triton logsumexp match PyTorch's"""
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create input tensors with gradients enabled
    shapes = [(2, 3, 10), (4, 5, 100)]

    for shape in shapes:
        # Create two identical tensors for PyTorch and Triton
        logits_torch = torch.randn(shape, device="cuda", requires_grad=True)
        logits_triton = logits_torch.clone().detach().requires_grad_(True)

        # Forward pass
        torch_output = torch_logsumexp(logits_torch)
        triton_output = TritonLogSumExp.apply(logits_triton)

        # Compare forward pass values
        max_diff_forward = (torch_output - triton_output).abs().max().item()
        assert max_diff_forward < 1e-5, f"Forward pass values differ for shape {shape}"

        # Create random gradient
        grad_output = torch.randn_like(torch_output)

        # Backward pass
        torch_output.backward(grad_output)
        triton_output.backward(grad_output)

        # Compare gradients
        max_diff_grad = (logits_torch.grad - logits_triton.grad).abs().max().item()
        print(f"Shape {shape}, Max gradient diff: {max_diff_grad}")

        # Assert that gradients are very close
        assert (
            max_diff_grad < 1e-5
        ), f"Gradients differ for shape {shape}: max diff {max_diff_grad}"
