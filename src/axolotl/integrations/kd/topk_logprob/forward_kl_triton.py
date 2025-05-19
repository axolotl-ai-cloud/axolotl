"""
Optimized Triton kernel for KL divergence loss between teacher and student models.
"""
# pylint: disable=invalid-name,unused-argument
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def fused_logsumexp_logprobs_kernel(
    student_logits_ptr,  # Input logits in original dtype
    student_logprobs_ptr,  # Output logprobs (float32)
    token_ids_ptr,  # Token IDs for top-k
    B,
    S,
    V,
    K,  # batch size, seq len, vocab size, top-k
    temperature,
    stride_l_b,
    stride_l_s,
    stride_l_v,
    stride_lp_b,
    stride_lp_s,
    stride_lp_k,
    stride_t_b,
    stride_t_s,
    stride_t_k,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes logsumexp and logprobs for topk tokens.
    All computations are done in float32 for numerical stability.
    """
    # Program ID
    pid = tl.program_id(0)
    batch_idx = pid // S
    seq_idx = pid % S

    # Bounds check
    if batch_idx >= B or seq_idx >= S:
        return

    # Compute logsumexp over the vocabulary
    max_val = -float("inf")

    # Phase 1: Find max value across vocabulary
    for v_offset in range(0, V, BLOCK_SIZE):
        # Create block indices and mask
        block_size = min(BLOCK_SIZE, V - v_offset)
        block_idx = tl.arange(0, BLOCK_SIZE)
        mask = block_idx < block_size

        # Load logits block and convert to float32 in-place
        ptrs = (
            student_logits_ptr
            + batch_idx * stride_l_b
            + seq_idx * stride_l_s
            + (v_offset + block_idx) * stride_l_v
        )
        block_logits = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)

        # Apply temperature scaling if needed
        if temperature != 1.0:
            block_logits = block_logits / temperature

        # Update max value
        block_max = tl.max(block_logits, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Phase 2: Compute sum of exp(logits - max_val)
    sum_exp = 0.0

    for v_offset in range(0, V, BLOCK_SIZE):
        # Create block indices and mask
        block_size = min(BLOCK_SIZE, V - v_offset)
        block_idx = tl.arange(0, BLOCK_SIZE)
        mask = block_idx < block_size

        # Load logits block and convert to float32 in-place
        ptrs = (
            student_logits_ptr
            + batch_idx * stride_l_b
            + seq_idx * stride_l_s
            + (v_offset + block_idx) * stride_l_v
        )
        block_logits = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)

        # Apply temperature scaling if needed
        if temperature != 1.0:
            block_logits = block_logits / temperature

        # Compute exp(logits - max_val) and add to sum
        block_exp = tl.exp(block_logits - max_val)
        sum_exp += tl.sum(block_exp * mask, axis=0)

    # Compute final logsumexp
    logsumexp = max_val + tl.log(sum_exp)

    # Phase 3: Compute and store logprobs for the top-k tokens
    token_ids_base = token_ids_ptr + batch_idx * stride_t_b + seq_idx * stride_t_s
    logprobs_base = (
        student_logprobs_ptr + batch_idx * stride_lp_b + seq_idx * stride_lp_s
    )

    for k in range(K):
        # Load token ID for position k
        token_id = tl.load(token_ids_base + k * stride_t_k)

        # Load the corresponding logit and convert to float32
        token_logit_ptr = (
            student_logits_ptr
            + batch_idx * stride_l_b
            + seq_idx * stride_l_s
            + token_id * stride_l_v
        )
        token_logit = tl.load(token_logit_ptr).to(tl.float32)

        # Apply temperature scaling if needed
        if temperature != 1.0:
            token_logit = token_logit / temperature

        # Compute logprob directly: logit - logsumexp
        token_logprob = token_logit - logsumexp

        # Store the result
        tl.store(logprobs_base + k * stride_lp_k, token_logprob)


@triton.jit
def grad_softmax_kernel(
    grad_student_logits_ptr,
    target_token_ids_ptr,
    teacher_probs_ptr,
    student_probs_ptr,
    mask_ptr,
    B,
    S,
    V,
    K,  # batch size, seq len, vocab size, top-k
    scale,
    stride_gl_b,
    stride_gl_s,
    stride_gl_v,
    stride_t_b,
    stride_t_s,
    stride_t_k,
    stride_p_b,
    stride_p_s,
    stride_p_k,
    stride_sp_b,
    stride_sp_s,
    stride_sp_k,
    stride_m_b,
    stride_m_s,
    stride_m_k,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    batch_idx = pid // S
    seq_idx = pid % S

    # Bounds check
    if batch_idx >= B or seq_idx >= S:
        return

    # Base pointers for this (batch, seq) pair
    grad_logits_base = (
        grad_student_logits_ptr + batch_idx * stride_gl_b + seq_idx * stride_gl_s
    )
    token_ids_base = (
        target_token_ids_ptr + batch_idx * stride_t_b + seq_idx * stride_t_s
    )
    teacher_probs_base = (
        teacher_probs_ptr + batch_idx * stride_p_b + seq_idx * stride_p_s
    )
    student_probs_base = (
        student_probs_ptr + batch_idx * stride_sp_b + seq_idx * stride_sp_s
    )
    mask_base = mask_ptr + batch_idx * stride_m_b + seq_idx * stride_m_s

    # Process each teacher probability one at a time, computing all gradients for it
    for k in range(0, K):
        # Load data for current position k
        teacher_prob = tl.load(teacher_probs_base + k * stride_p_k)
        student_prob_k = tl.load(student_probs_base + k * stride_sp_k)
        mask_val = tl.load(mask_base + k * stride_m_k)

        # Precompute the self-influence term (multiplied by scale)
        self_term = teacher_prob * (1.0 - student_prob_k) * scale

        # Calculate gradient contributions for all positions j
        for j in range(0, K):
            token_id_j = tl.load(token_ids_base + j * stride_t_k)
            student_prob_j = tl.load(student_probs_base + j * stride_sp_k)
            mask_j = tl.load(mask_base + j * stride_m_k)

            # Calculate the masking factor
            combined_mask = mask_val * mask_j

            # Determine if this is a diagonal or off-diagonal term
            is_k_equals_j = tl.where(k == j, 1.0, 0.0)

            # Compute the gradient contribution
            # For diagonal (k==j): -teacher_prob * (1-student_prob_k) * scale * mask
            # For off-diagonal: -(-teacher_prob * student_prob_j) * scale * mask
            grad_contribution = (
                -(
                    self_term * is_k_equals_j
                    - teacher_prob * student_prob_j * scale * (1.0 - is_k_equals_j)
                )
                * combined_mask
            )

            # Atomically update the gradient for this token
            tl.atomic_add(
                grad_logits_base + token_id_j * stride_gl_v, grad_contribution
            )


@triton.jit
def grad_topk_softmax_kernel(
    grad_student_logits_ptr,
    student_logits_ptr,
    target_token_ids_ptr,
    teacher_probs_ptr,
    student_probs_ptr,
    mask_ptr,
    B,
    S,
    V,
    K,  # batch size, seq len, vocab size, top-k
    scale,
    stride_gl_b,
    stride_gl_s,
    stride_gl_v,
    stride_l_b,
    stride_l_s,
    stride_l_v,
    stride_t_b,
    stride_t_s,
    stride_t_k,
    stride_p_b,
    stride_p_s,
    stride_p_k,
    stride_sp_b,
    stride_sp_s,
    stride_sp_k,
    stride_m_b,
    stride_m_s,
    stride_m_k,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    batch_idx = pid // S
    seq_idx = pid % S

    # Bounds check
    if batch_idx >= B or seq_idx >= S:
        return

    # Base pointers for this (batch, seq) pair
    grad_logits_base = (
        grad_student_logits_ptr + batch_idx * stride_gl_b + seq_idx * stride_gl_s
    )
    # logits_base = student_logits_ptr + batch_idx * stride_l_b + seq_idx * stride_l_s
    token_ids_base = (
        target_token_ids_ptr + batch_idx * stride_t_b + seq_idx * stride_t_s
    )
    teacher_probs_base = (
        teacher_probs_ptr + batch_idx * stride_p_b + seq_idx * stride_p_s
    )
    student_probs_base = (
        student_probs_ptr + batch_idx * stride_sp_b + seq_idx * stride_sp_s
    )
    mask_base = mask_ptr + batch_idx * stride_m_b + seq_idx * stride_m_s

    # Load all token IDs, probs and masks for this position
    token_ids = tl.zeros([K], dtype=tl.int32)
    teacher_probs = tl.zeros([K], dtype=tl.float32)
    student_probs = tl.zeros([K], dtype=tl.float32)
    masks = tl.zeros([K], dtype=tl.float32)

    for k in range(K):
        token_ids[k] = tl.load(token_ids_base + k * stride_t_k)
        teacher_probs[k] = tl.load(teacher_probs_base + k * stride_p_k)
        student_probs[k] = tl.load(student_probs_base + k * stride_sp_k)
        masks[k] = tl.load(mask_base + k * stride_m_k)

    # Process gradients for all tokens in this position
    for k in range(K):
        # token_id = token_ids[k]
        mask_k = masks[k]

        # Skip computation if mask is zero by multiplying gradient by mask
        for j in range(K):
            other_token_id = token_ids[j]
            mask_j = masks[j]
            combined_mask = mask_k * mask_j

            # Compute gradient differently for diagonal vs off-diagonal entries
            # Using * 1.0 to convert boolean to float
            is_diagonal = tl.where(j == k, 1.0, 0.0)

            # Self influence: gradient = teacher_prob * (1 - student_prob)
            self_grad = teacher_probs[k] * (1.0 - student_probs[k]) * is_diagonal

            # Cross influence: gradient = -teacher_prob[k] * student_prob[j]
            cross_grad = -teacher_probs[k] * student_probs[j] * (1.0 - is_diagonal)

            # Combined gradient scaled by mask
            grad_val = (self_grad + cross_grad) * scale * combined_mask

            tl.atomic_add(grad_logits_base + other_token_id * stride_gl_v, grad_val)


# Triton-accelerated implementation of KL divergence loss for top-k tokens
# Chunking helper functions for handling long sequences
def chunk_tensor(
    tensor: torch.Tensor, max_seq_len: int
) -> Tuple[torch.Tensor, Optional[int]]:
    """Split a tensor along sequence dimension if needed."""
    _, seq_len, *__ = tensor.shape

    if seq_len <= max_seq_len:
        return tensor, None

    num_chunks = (seq_len + max_seq_len - 1) // max_seq_len
    chunks = []

    for i in range(num_chunks):
        start_idx = i * max_seq_len
        end_idx = min((i + 1) * max_seq_len, seq_len)
        chunks.append(tensor[:, start_idx:end_idx, ...])

    return chunks, num_chunks


def merge_chunks(chunks: list, original_shape: torch.Size):
    """Merge chunks back into a single tensor with original shape."""
    return torch.cat(chunks, dim=1)


# Triton-accelerated implementation of KL divergence loss for top-k tokens
class TopKKLDivergence(torch.autograd.Function):
    """
    Autograd function for KL divergence loss between top-k logprobs
    with support for chunking to handle very long sequences.
    """

    # Max sequence length to process in a single kernel launch
    # This is a tunable parameter that might need adjustment based on GPU memory
    MAX_SEQ_LEN = 8192

    @staticmethod
    def forward(
        ctx,
        student_logits,
        target_token_ids,
        target_logprobs,
        target_mask,
        num_items_in_batch=-1,
        kd_temperature=1.0,
        top_k_before_softmax=0,
    ):
        """
        Forward pass for KL divergence loss between top-k logprobs with chunking.
        """
        # Only convert target_logprobs to float, leave student_logits as is
        target_logprobs = target_logprobs.float()

        # Get dimensions
        batch_size, _, vocab_size = student_logits.shape
        _, teacher_seq_len, top_k = target_token_ids.shape

        # Slice student logits to match teacher sequence length
        student_logits_for_kd = student_logits[:, :teacher_seq_len, :]

        # Store original values for backward pass
        ctx.original_seq_len = teacher_seq_len
        ctx.original_dtype = student_logits.dtype

        # Apply chunking for long sequences
        if teacher_seq_len > TopKKLDivergence.MAX_SEQ_LEN:
            # Chunk the inputs
            student_logits_chunks, num_chunks = chunk_tensor(
                student_logits_for_kd, TopKKLDivergence.MAX_SEQ_LEN
            )
            target_token_ids_chunks, _ = chunk_tensor(
                target_token_ids, TopKKLDivergence.MAX_SEQ_LEN
            )
            # target_logprobs_chunks, _ = chunk_tensor(
            #     target_logprobs, TopKKLDivergence.MAX_SEQ_LEN
            # )
            # target_mask_chunks, _ = chunk_tensor(
            #     target_mask, TopKKLDivergence.MAX_SEQ_LEN
            # )

            # Process each chunk
            student_logprobs_chunks = []
            student_probs_chunks = []

            for i in range(num_chunks):
                chunk_logits = student_logits_chunks[i]
                chunk_token_ids = target_token_ids_chunks[i]
                chunk_seq_len = chunk_logits.shape[1]

                if top_k_before_softmax:
                    # Apply temperature to student logits
                    if kd_temperature != 1.0:
                        chunk_logits = chunk_logits / kd_temperature

                    # Gather student logits for top-k tokens
                    chunk_logits_topk = torch.gather(
                        chunk_logits, dim=-1, index=chunk_token_ids
                    )

                    # Compute softmax over gathered logits
                    chunk_logprobs_topk = torch.log_softmax(chunk_logits_topk, dim=-1)
                    chunk_probs_topk = torch.exp(chunk_logprobs_topk)
                else:
                    # Allocate output tensor for logprobs directly (always in float32)
                    chunk_logprobs_topk = torch.empty(
                        (batch_size, chunk_seq_len, top_k),
                        dtype=torch.float32,
                        device=chunk_logits.device,
                    )

                    # Launch fused kernel directly
                    grid = (batch_size * chunk_seq_len,)
                    fused_logsumexp_logprobs_kernel[grid](
                        chunk_logits.contiguous(),
                        chunk_logprobs_topk,
                        chunk_token_ids.contiguous(),
                        batch_size,
                        chunk_seq_len,
                        vocab_size,
                        top_k,
                        kd_temperature,
                        chunk_logits.stride(0),
                        chunk_logits.stride(1),
                        chunk_logits.stride(2),
                        chunk_logprobs_topk.stride(0),
                        chunk_logprobs_topk.stride(1),
                        chunk_logprobs_topk.stride(2),
                        chunk_token_ids.stride(0),
                        chunk_token_ids.stride(1),
                        chunk_token_ids.stride(2),
                        min(1024, triton.next_power_of_2(vocab_size)),
                    )

                    # Calculate probs from logprobs
                    chunk_probs_topk = torch.exp(chunk_logprobs_topk)

                # Store results
                student_logprobs_chunks.append(chunk_logprobs_topk)
                student_probs_chunks.append(chunk_probs_topk)

            # Merge results
            student_logprobs_topk = torch.cat(student_logprobs_chunks, dim=1)
            student_probs_topk = torch.cat(student_probs_chunks, dim=1)

            # Save chunking info for backward pass
            ctx.used_chunking = True
            ctx.num_chunks = num_chunks

        else:
            # Original code path for shorter sequences
            if top_k_before_softmax:
                # Apply temperature to student logits
                if kd_temperature != 1.0:
                    student_logits_for_kd = student_logits_for_kd / kd_temperature

                # Gather student logits for top-k tokens
                student_logits_topk = torch.gather(
                    student_logits_for_kd, dim=-1, index=target_token_ids
                )

                # Compute softmax over gathered logits
                student_logprobs_topk = torch.log_softmax(student_logits_topk, dim=-1)
                student_probs_topk = torch.exp(student_logprobs_topk)
            else:
                # Allocate output tensor for logprobs directly (always in float32)
                student_logprobs_topk = torch.empty(
                    (batch_size, teacher_seq_len, top_k),
                    dtype=torch.float32,
                    device=student_logits.device,
                )

                # Launch fused kernel directly
                grid = (batch_size * teacher_seq_len,)
                fused_logsumexp_logprobs_kernel[grid](
                    student_logits_for_kd.contiguous(),
                    student_logprobs_topk,
                    target_token_ids.contiguous(),
                    batch_size,
                    teacher_seq_len,
                    vocab_size,
                    top_k,
                    kd_temperature,
                    student_logits_for_kd.stride(0),
                    student_logits_for_kd.stride(1),
                    student_logits_for_kd.stride(2),
                    student_logprobs_topk.stride(0),
                    student_logprobs_topk.stride(1),
                    student_logprobs_topk.stride(2),
                    target_token_ids.stride(0),
                    target_token_ids.stride(1),
                    target_token_ids.stride(2),
                    min(1024, triton.next_power_of_2(vocab_size)),
                )

                # Calculate probs from logprobs
                student_probs_topk = torch.exp(student_logprobs_topk)

            # No chunking used
            ctx.used_chunking = False

        # Save tensors for backward pass
        ctx.save_for_backward(
            student_logits_for_kd,
            target_token_ids,
            target_logprobs,
            target_mask,
            student_probs_topk,
        )
        ctx.kd_temperature = kd_temperature
        ctx.top_k_before_softmax = top_k_before_softmax
        ctx.num_items_in_batch = num_items_in_batch

        # Convert mask to boolean
        valid_mask = target_mask.bool()

        # Extract valid tokens only - this is where the error was happening
        # Use cloned contiguous tensors and explicit indexing for safety
        student_logprobs_flat = student_logprobs_topk.view(-1, top_k)
        target_logprobs_flat = target_logprobs.view(-1, top_k)
        valid_mask_flat = valid_mask.view(-1, top_k)

        # Gather valid indices explicitly to avoid illegal memory access
        valid_indices = torch.nonzero(valid_mask_flat.view(-1)).squeeze(-1)
        student_logprobs_valid = torch.index_select(
            student_logprobs_flat.view(-1), 0, valid_indices
        )
        target_logprobs_valid = torch.index_select(
            target_logprobs_flat.view(-1), 0, valid_indices
        )

        # Convert teacher logprobs to probabilities
        teacher_probs_valid = torch.exp(target_logprobs_valid)

        # Compute KL divergence loss
        token_losses = teacher_probs_valid * (
            target_logprobs_valid - student_logprobs_valid
        )
        kd_loss = token_losses.sum()

        # Apply temperature scaling
        # pylint: disable=duplicate-code
        if kd_temperature != 1.0:
            kd_loss = kd_loss * (kd_temperature**2)

        # Normalize by number of items or valid tokens
        if num_items_in_batch > 0:
            kd_loss = kd_loss / float(num_items_in_batch)
        else:
            num_valid_tokens = valid_indices.numel()
            kd_loss = kd_loss / float(num_valid_tokens if num_valid_tokens > 0 else 1)

        return kd_loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Optimized backward pass for KL divergence loss with proper dtype handling and chunking.
        """
        (
            student_logits,
            target_token_ids,
            target_logprobs,
            target_mask,
            student_probs,
        ) = ctx.saved_tensors
        kd_temperature = ctx.kd_temperature
        num_items_in_batch = ctx.num_items_in_batch
        original_dtype = ctx.original_dtype

        # Get dimensions
        batch_size, _, vocab_size = student_logits.shape
        _, teacher_seq_len, top_k = target_token_ids.shape

        # Initialize gradient tensor in float32 to support atomic operations
        grad_student_logits = torch.zeros_like(student_logits, dtype=torch.float32)

        # Compute scaling factor
        scale = grad_output.item()

        # Apply temperature scaling from forward pass
        if kd_temperature != 1.0:
            scale = scale * (kd_temperature**2)

        # Normalize by number of items or valid tokens
        if num_items_in_batch > 0:
            scale = scale / float(num_items_in_batch)
        else:
            scale = scale / float(target_mask.sum().item())

        # Apply chain rule for temperature scaling (1/temperature)
        if kd_temperature != 1.0:
            scale = scale / kd_temperature

        # Convert teacher logprobs to probabilities
        teacher_probs = torch.exp(target_logprobs)

        # Use chunking for the backward pass if used in forward
        if getattr(ctx, "used_chunking", False):
            num_chunks = ctx.num_chunks
            max_seq = TopKKLDivergence.MAX_SEQ_LEN

            # Process each chunk
            for i in range(num_chunks):
                start_idx = i * max_seq
                end_idx = min((i + 1) * max_seq, teacher_seq_len)
                chunk_len = end_idx - start_idx

                # Get chunk slices
                # student_logits_chunk = student_logits[:, start_idx:end_idx, :]
                target_token_ids_chunk = target_token_ids[:, start_idx:end_idx, :]
                teacher_probs_chunk = teacher_probs[:, start_idx:end_idx, :]
                student_probs_chunk = student_probs[:, start_idx:end_idx, :]
                target_mask_chunk = target_mask[:, start_idx:end_idx, :]
                grad_student_logits_chunk = grad_student_logits[:, start_idx:end_idx, :]

                # Launch gradient computation kernel for this chunk
                grid = (batch_size * chunk_len,)
                grad_softmax_kernel[grid](
                    grad_student_logits_chunk.contiguous(),
                    target_token_ids_chunk.contiguous(),
                    teacher_probs_chunk.contiguous(),
                    student_probs_chunk.contiguous(),
                    target_mask_chunk.contiguous(),
                    batch_size,
                    chunk_len,
                    vocab_size,
                    top_k,
                    scale,
                    grad_student_logits_chunk.stride(0),
                    grad_student_logits_chunk.stride(1),
                    grad_student_logits_chunk.stride(2),
                    target_token_ids_chunk.stride(0),
                    target_token_ids_chunk.stride(1),
                    target_token_ids_chunk.stride(2),
                    teacher_probs_chunk.stride(0),
                    teacher_probs_chunk.stride(1),
                    teacher_probs_chunk.stride(2),
                    student_probs_chunk.stride(0),
                    student_probs_chunk.stride(1),
                    student_probs_chunk.stride(2),
                    target_mask_chunk.stride(0),
                    target_mask_chunk.stride(1),
                    target_mask_chunk.stride(2),
                    min(1024, triton.next_power_of_2(top_k)),
                )

                # Update the gradient tensor (already in-place)
        else:
            # Original code path for shorter sequences
            # Launch gradient computation kernel
            grid = (batch_size * teacher_seq_len,)
            grad_softmax_kernel[grid](
                grad_student_logits.contiguous(),
                target_token_ids.contiguous(),
                teacher_probs.contiguous(),
                student_probs.contiguous(),
                target_mask.contiguous(),
                batch_size,
                teacher_seq_len,
                vocab_size,
                top_k,
                scale,
                grad_student_logits.stride(0),
                grad_student_logits.stride(1),
                grad_student_logits.stride(2),
                target_token_ids.stride(0),
                target_token_ids.stride(1),
                target_token_ids.stride(2),
                teacher_probs.stride(0),
                teacher_probs.stride(1),
                teacher_probs.stride(2),
                student_probs.stride(0),
                student_probs.stride(1),
                student_probs.stride(2),
                target_mask.stride(0),
                target_mask.stride(1),
                target_mask.stride(2),
                min(1024, triton.next_power_of_2(top_k)),
            )

        # Convert gradient back to original dtype if needed
        if original_dtype != torch.float32:
            grad_student_logits = grad_student_logits.to(original_dtype)

        # Return gradients for student_logits and None for other inputs
        return grad_student_logits, None, None, None, None, None, None


# Wrapper function for chunked computation
def loss(
    student_logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    target_logprobs: torch.Tensor,
    target_mask: torch.Tensor,
    num_items_in_batch: int = -1,
    kd_temperature: float = 1.0,
    top_k_before_softmax: int = 0,
    max_seq_len: Optional[int] = None,
):
    """
    Triton-accelerated Memory-efficient KL divergence loss computation for knowledge distillation
    with support for very long sequences.

    Args:
        student_logits: Student logits [B, seq_len, vocab_size]
        target_token_ids: Teacher token IDs [B, seq_len, top_k]
        target_logprobs: Teacher logprobs [B, seq_len, top_k]
        target_mask: Token mask [B, seq_len, top_k]
        num_items_in_batch: Number of items for normalization (-1 for auto)
        kd_temperature: Temperature for KD
        top_k_before_softmax: Flag for softmax application order
        max_seq_len: Override default MAX_SEQ_LEN value for chunking
    """
    # Allow overriding the max sequence length
    if max_seq_len is not None and max_seq_len > 0:
        TopKKLDivergence.MAX_SEQ_LEN = max_seq_len

    total_loss = TopKKLDivergence.apply(
        student_logits,
        target_token_ids,
        target_logprobs,
        target_mask,
        -1 if num_items_in_batch <= 0 else num_items_in_batch,
        kd_temperature,
        top_k_before_softmax,
    )

    return total_loss
