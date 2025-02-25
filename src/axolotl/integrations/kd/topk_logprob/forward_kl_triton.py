"""
Optimized Triton kernel for KL divergence loss between teacher and student models.
"""
import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.benchmark import Timer

# Helper function for computing logsumexp
@triton.jit
def logsumexp_kernel(
    logits_ptr, output_ptr,
    B, S, V,  # batch size, seq len, vocab size
    stride_b, stride_s, stride_v,
    out_stride_b, out_stride_s,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    batch_idx = pid // S
    seq_idx = pid % S

    # Bounds check
    if batch_idx >= B or seq_idx >= S:
        return

    # Pointers
    logits_base = logits_ptr + batch_idx * stride_b + seq_idx * stride_s

    # Find maximum for numerical stability
    max_val = -float('inf')
    for v_offset in range(0, V, BLOCK_SIZE):
        v_size = min(BLOCK_SIZE, V - v_offset)
        mask = tl.arange(0, BLOCK_SIZE) < v_size

        logits_block = tl.load(logits_base + (v_offset + tl.arange(0, BLOCK_SIZE)) * stride_v,
                               mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(logits_block, axis=0))

    # Compute sum of exp(logit - max_val)
    sum_exp = 0.0
    for v_offset in range(0, V, BLOCK_SIZE):
        v_size = min(BLOCK_SIZE, V - v_offset)
        mask = tl.arange(0, BLOCK_SIZE) < v_size

        logits_block = tl.load(logits_base + (v_offset + tl.arange(0, BLOCK_SIZE)) * stride_v,
                               mask=mask, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(logits_block - max_val), axis=0)

    # Compute logsumexp
    result = max_val + tl.log(sum_exp)

    # Store result
    tl.store(output_ptr + batch_idx * out_stride_b + seq_idx * out_stride_s, result)

# Triton-accelerated implementation of KL divergence loss for top-k tokens
class TopKKLDivergence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student_logits, target_token_ids, target_logprobs, target_mask,
                num_items_in_batch=-1, kd_temperature=1.0, top_k_before_softmax=0):
        """
        Forward pass for KL divergence loss between top-k logprobs.
        """
        # Convert inputs to appropriate types
        student_logits = student_logits.float()
        target_logprobs = target_logprobs.float()

        # Get dimensions
        batch_size, student_seq_len, vocab_size = student_logits.shape
        _, teacher_seq_len, top_k = target_token_ids.shape

        # Slice student logits to match teacher sequence length
        student_logits_for_kd = student_logits[:, :teacher_seq_len, :]

        if top_k_before_softmax:
            # 1. Apply temperature to student logits
            if kd_temperature != 1.0:
                student_logits_for_kd = student_logits_for_kd / kd_temperature

            # 2. Gather student logits for top-k tokens
            student_logits_topk = torch.gather(student_logits_for_kd, dim=-1, index=target_token_ids)

            # 3. Compute softmax over gathered logits
            student_logprobs_topk = torch.log_softmax(student_logits_topk, dim=-1)
        else:
            # 1. Apply temperature to student logits
            if kd_temperature != 1.0:
                student_logits_for_kd = student_logits_for_kd / kd_temperature

            # 2. Gather student logits for top-k tokens
            student_logits_topk = torch.gather(student_logits_for_kd, dim=-1, index=target_token_ids)

            # 3. Compute logsumexp over full vocabulary using Triton
            student_lse = torch.empty((batch_size, teacher_seq_len),
                                      dtype=torch.float32, device=student_logits.device)

            grid = (batch_size * teacher_seq_len,)
            logsumexp_kernel[grid](
                student_logits_for_kd.contiguous(), student_lse,
                batch_size, teacher_seq_len, vocab_size,
                student_logits_for_kd.stride(0), student_logits_for_kd.stride(1), student_logits_for_kd.stride(2),
                student_lse.stride(0), student_lse.stride(1),
                min(1024, triton.next_power_of_2(vocab_size))
            )

            # 4. Convert to logprobs
            student_logprobs_topk = student_logits_topk - student_lse.unsqueeze(-1)

        # Save tensors for backward pass
        ctx.save_for_backward(student_logits_for_kd, target_token_ids, target_logprobs, target_mask, student_logprobs_topk)
        ctx.kd_temperature = kd_temperature
        ctx.top_k_before_softmax = top_k_before_softmax
        ctx.num_items_in_batch = num_items_in_batch

        # Convert mask to boolean
        valid_mask = target_mask.bool()

        # Extract valid tokens only
        student_logprobs_valid = student_logprobs_topk[valid_mask]
        target_logprobs_valid = target_logprobs[valid_mask]

        # Convert teacher logprobs to probabilities
        teacher_probs_valid = torch.exp(target_logprobs_valid)

        # Compute KL divergence loss
        token_losses = teacher_probs_valid * (target_logprobs_valid - student_logprobs_valid)
        kd_loss = token_losses.sum()

        # Apply temperature scaling
        if kd_temperature != 1.0:
            kd_loss = kd_loss * (kd_temperature**2)

        # Normalize by number of items or valid tokens
        if num_items_in_batch > 0:
            kd_loss = kd_loss / float(num_items_in_batch)
        else:
            kd_loss = kd_loss / float(token_losses.size(0))

        return kd_loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for KL divergence loss.
        """
        student_logits, target_token_ids, target_logprobs, target_mask, student_logprobs = ctx.saved_tensors
        kd_temperature = ctx.kd_temperature
        top_k_before_softmax = ctx.top_k_before_softmax
        num_items_in_batch = ctx.num_items_in_batch

        valid_mask = target_mask.bool()

        batch_size, seq_len, vocab_size = student_logits.shape
        grad_student_logits = torch.zeros_like(student_logits)

        # Convert teacher logprobs to probs
        teacher_probs = torch.exp(target_logprobs)

        # Scale gradient by temperature if needed
        scale = (kd_temperature**2) if kd_temperature != 1.0 else 1.0

        # Normalize by number of items or valid tokens
        if num_items_in_batch > 0:
            scale = scale / float(num_items_in_batch)
        else:
            scale = scale / float(valid_mask.sum().item())

        # Apply gradient
        scale = scale * grad_output.item()

        # Let PyTorch compute the gradients for us
        with torch.enable_grad():
            student_logits_grad = student_logits.detach().requires_grad_(True)

            if top_k_before_softmax:
                student_logits_topk = torch.gather(
                    student_logits_grad / kd_temperature if kd_temperature != 1.0 else student_logits_grad,
                    dim=-1, index=target_token_ids
                )
                student_logprobs_topk = torch.log_softmax(student_logits_topk, dim=-1)
            else:
                temp_logits = student_logits_grad / kd_temperature if kd_temperature != 1.0 else student_logits_grad
                student_logits_topk = torch.gather(temp_logits, dim=-1, index=target_token_ids)
                student_lse = torch.logsumexp(temp_logits, dim=-1, keepdim=True)
                student_logprobs_topk = student_logits_topk - student_lse

            # Extract valid tokens only
            student_logprobs_valid = student_logprobs_topk[valid_mask]
            target_logprobs_valid = target_logprobs[valid_mask]
            teacher_probs_valid = torch.exp(target_logprobs_valid)

            # Compute KL divergence loss
            token_losses = teacher_probs_valid * (target_logprobs_valid - student_logprobs_valid)
            kd_loss = token_losses.sum() * scale

            # Backward pass
            kd_loss.backward()

            grad_student_logits = student_logits_grad.grad

        return grad_student_logits, None, None, None, None, None, None

# Wrapper function for chunked computation
def kl_div_loss_chunked(student_logits, target_token_ids, target_logprobs, target_mask,
                        num_items_in_batch=-1, kd_temperature=1.0, top_k_before_softmax=0,
                        n_chunks=1):
    """
    Memory-efficient KL divergence loss computation.

    Args:
        student_logits: Student logits [B, seq_len, vocab_size]
        target_token_ids: Teacher token IDs [B, seq_len, top_k]
        target_logprobs: Teacher logprobs [B, seq_len, top_k]
        target_mask: Token mask [B, seq_len, top_k]
        num_items_in_batch: Number of items for normalization (-1 for auto)
        kd_temperature: Temperature for KD
        top_k_before_softmax: Flag for softmax application order
        n_chunks: Number of chunks to process (for memory efficiency)
    """
    batch_size = student_logits.shape[0]

    # If n_chunks <= 0, use the entire batch size
    if n_chunks <= 0:
        n_chunks = batch_size

    # Determine the actual number of chunks to use (find largest factor <= n_chunks)
    factors = [i for i in range(1, batch_size + 1) if batch_size % i == 0]
    actual_chunks = factors[min(len(factors) - 1, max(0, next((i for i, f in enumerate(factors) if f >= n_chunks), len(factors) - 1)))]

    # Compute chunk size
    chunk_size = batch_size // actual_chunks
    total_loss = 0.0

    # Process in chunks
    for i in range(0, batch_size, chunk_size):
        chunk_end = min(i + chunk_size, batch_size)
        chunk_loss = TopKKLDivergence.apply(
            student_logits[i:chunk_end],
            target_token_ids[i:chunk_end],
            target_logprobs[i:chunk_end],
            target_mask[i:chunk_end],
            -1 if num_items_in_batch <= 0 else num_items_in_batch // actual_chunks,
            kd_temperature,
            top_k_before_softmax
        )
        total_loss += chunk_loss

    # Normalize by the number of chunks
    return total_loss / actual_chunks


def loss(
    student_logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    target_logprobs: torch.Tensor,
    target_mask: torch.Tensor,
    num_items_in_batch: int = -1,
    kd_temperature: float = 1.0,
    top_k_before_softmax: int = 0,
    n_chunks: int = 1
) -> torch.Tensor:
    """
    Triton-accelerated KL divergence loss for knowledge distillation.

    Args:
        student_logits: Student model logits [B, seq_len, vocab_size]
        target_token_ids: Teacher's top-k token IDs [B, seq_len, top_k]
        target_logprobs: Teacher's top-k logprobs [B, seq_len, top_k]
        target_mask: Mask for valid tokens [B, seq_len, top_k]
        num_items_in_batch: Number of items for normalization (-1 for auto)
        kd_temperature: Temperature for KD
        top_k_before_softmax: Flag for softmax application order
        n_chunks: Number of chunks for memory efficiency
    """
    return kl_div_loss_chunked(
        student_logits, target_token_ids, target_logprobs, target_mask,
        num_items_in_batch, kd_temperature, top_k_before_softmax,
        n_chunks
    )
