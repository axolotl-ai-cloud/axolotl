"""
Triton kernel for optimized kl divergence loss
"""

import torch
import triton
import triton.language as tl

# --------------------------------------------------------
# Triton Kernel for forward pass
# --------------------------------------------------------
# We'll assume:
#   - B * seq_len threads in 1D dimension
#   - Each thread handles K tokens (the top-K from teacher).
#   - For large K, you might want a more 2D approach to keep good occupancy.
#
# Pseudocode steps inside kernel:
#   1) compute index for [batch, seq_position]
#   2) read top-K token IDs from teacher_token_ids
#   3) gather student_logits_topk
#   4) compute logsumexp for those K logits
#   5) compute student_logprobs_topk
#   6) read teacher_logprobs
#   7) compute teacher_probs = exp(teacher_logprobs)
#   8) compute partial KL = sum(teacher_probs * (teacher_logprobs - student_logprobs_topk))
#   9) store partial KL in a buffer
#
# Later, we'll do a reduction on partial KL across all threads.
#
# NOTE: This is a reference skeleton. You must adapt indexing carefully.
#


@triton.jit
def kd_forward_kernel(
    # student_logits after gather: [B, seq_len, K] flattened to 1D in row-major
    student_logits_ptr: tl.tensor,
    # teacher_logprobs: [B, seq_len, K] flattened
    teacher_logprobs_ptr: tl.tensor,
    # mask: [B, seq_len, K] flattened (bool or 0/1)
    mask_ptr: tl.tensor,
    # partial_kd: [B*seq_len] flattened buffer to store partial sums
    partial_kd_ptr: tl.tensor,
    B: tl.int32,  # pylint: disable=invalid-name
    seq_len: tl.int32,
    K: tl.int32,  # pylint: disable=invalid-name
    BLOCK_SIZE: tl.constexpr,  # pylint: disable=invalid-name
):
    """
    For each position in [0..B*seq_len), we:
      - gather the K student logits
      - compute logsumexp
      - compute the KL sum = sum_{k} t_prob_k * ( t_log_k - s_logprob_k )
      - store that partial sum into partial_kd_ptr[offset].
    """
    # 1) Identify which [B*seq_len] index this block handles
    pid = tl.program_id(0)

    # 2) Vector of [0..BLOCK_SIZE) local offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    # 3) Global indices = pid * BLOCK_SIZE + offsets
    idx = pid * BLOCK_SIZE + offsets

    # 4) Mask to ensure we don’t read out-of-bounds
    total_positions = B * seq_len
    mask_pos = idx < total_positions

    # 5) Convert a 1D `idx` => (b_idx, s_idx)
    #    b_idx is the batch number, s_idx is the sequence position
    b_idx = idx // seq_len
    s_idx = idx % seq_len

    # We'll accumulate the KL for each index in a register array
    kl_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # -------------------------------------------------------------------------
    # First pass: find max logits over K to implement logsumexp
    # -------------------------------------------------------------------------
    max_val = tl.full([BLOCK_SIZE], -1e30, dtype=tl.float32)

    # Python-level loops are allowed in Triton as long as the
    # operations inside are Triton ops, not torch or Python math.
    for k in range(K):
        # pointer offset in the flattened [B, seq_len, K] = b_idx*(seq_len*K) + s_idx*K + k
        offset_k = b_idx * (seq_len * K) + s_idx * K + k

        # load student logits, masked out-of-bounds with a large negative
        # so they don't affect the max
        student_val = tl.where(mask_pos, tl.load(student_logits_ptr + offset_k), -1e30)
        # update running max
        max_val = tl.where(student_val > max_val, student_val, max_val)

    # -------------------------------------------------------------------------
    # Second pass: sum of exp(...) to complete logsumexp
    # -------------------------------------------------------------------------
    exp_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in range(K):
        offset_k = b_idx * (seq_len * K) + s_idx * K + k
        student_val = tl.where(mask_pos, tl.load(student_logits_ptr + offset_k), -1e30)
        # exponent
        exponent = tl.exp(student_val - max_val)
        exp_sum += exponent

    # final logsumexp
    logsumexp_val = max_val + tl.log(exp_sum)

    # -------------------------------------------------------------------------
    # Third pass: compute partial KL per position
    #   KL = sum_{k in valid} p^T_k * (teacher_logprobs_k - student_logprobs_k)
    #
    #   - teacher_logprobs_k => t_log
    #   - teacher_prob_k = exp(t_log)
    #   - student_logprobs_k = s_val - logsumexp_val
    # -------------------------------------------------------------------------
    for k in range(K):
        offset_k = b_idx * (seq_len * K) + s_idx * K + k
        # teacher logprobs
        t_log = tl.where(mask_pos, tl.load(teacher_logprobs_ptr + offset_k), -1e30)
        # teacher prob
        t_prob = tl.exp(t_log)

        # student logit
        s_val = tl.where(mask_pos, tl.load(student_logits_ptr + offset_k), -1e30)
        # student logprob
        s_logprob = s_val - logsumexp_val

        # local KL
        kl_val = t_prob * (t_log - s_logprob)

        # also read mask to disable invalid tokens if mask is not purely sequence-based
        valid_k = tl.load(mask_ptr + offset_k)
        # if mask is bool => use 'valid_k != 0', if it's 0/1 => same
        is_valid = valid_k > 0

        # zero out if either this index is out-of-bounds or mask is invalid
        kl_val = tl.where(mask_pos & is_valid, kl_val, 0.0)

        # accumulate
        kl_acc += kl_val

    # -------------------------------------------------------------------------
    # Store the partial KL in partial_kd_ptr for each element in idx.
    # Later in Python, you can do partial_kd.sum() to get the total KL.
    # -------------------------------------------------------------------------
    tl.store(partial_kd_ptr + idx, kl_acc, mask=mask_pos)


def kd_forward_pass_triton(
    student_logits,  # [B, seq_len, K]  (already gathered)
    teacher_logprobs,  # [B, seq_len, K]
    mask,  # [B, seq_len, K] bool or 0/1
    BLOCK_SIZE=1024,  # pylint: disable=invalid-name
):
    """
    Returns total KL (float). We do the sum on the Python side.
    NOTE: No normalization is done here.
          You might divide by `num_items_in_batch` or # valid tokens afterward.
    """
    B, seq_len, K = student_logits.shape  # pylint: disable=invalid-name
    # Flatten
    student_logits_flat = student_logits.reshape(-1)
    teacher_logprobs_flat = teacher_logprobs.reshape(-1)
    mask_flat = mask.reshape(-1)

    total_positions = B * seq_len
    # We'll store partial KL sums for each of the B*seq_len positions
    partial_kd = torch.empty(
        total_positions, dtype=student_logits.dtype, device=student_logits.device
    )

    # Grid config
    grid = ((total_positions + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    kd_forward_kernel[grid](
        student_logits_flat,
        teacher_logprobs_flat,
        mask_flat,
        partial_kd,
        B,
        seq_len,
        K,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Sum on CPU or GPU
    kd_sum = partial_kd.sum()
    return kd_sum


class _KLDivergenceTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student_logits, teacher_logprobs, mask):
        """
        student_logits: (B, seq_len, K)
        teacher_logprobs: (B, seq_len, K)
        mask: (B, seq_len, K)
        """
        kd_sum = kd_forward_pass_triton(student_logits, teacher_logprobs, mask)
        kd_loss = kd_sum  # Not normalized here. You can do that externally.

        # Save for backward
        ctx.save_for_backward(student_logits, teacher_logprobs, mask)
        return kd_loss

    @staticmethod
    def backward(ctx, grad_output):
        # We'll do naive PyTorch re-computation for gradient wrt student_logits
        student_logits, teacher_logprobs, mask = ctx.saved_tensors
        # grad_output is dLoss/dOut => a scalar
        # Let’s compute dLoss/dStudentLogits with the same formula as your original code

        with torch.enable_grad():
            stl = student_logits.clone().detach().requires_grad_(True)
            t_log = teacher_logprobs
            # mask might be bool or 0/1
            # compute logsumexp
            lse = torch.logsumexp(stl, dim=-1, keepdim=True)
            s_logprob = stl - lse
            t_prob = t_log.exp()

            # forward KL = sum_{k} p^T_k ( t_log_k - s_logprob_k )
            kl_val = t_prob * (t_log - s_logprob)
            # mask out
            kl_val = kl_val * mask  # zero out invalid

            kd_loss = kl_val.sum()
            # now compute dLoss/d stl
            grad_stl = torch.autograd.grad(kd_loss, stl, grad_outputs=grad_output)[0]

        return grad_stl, None, None


def kd_loss_triton(
    student_logits,  # [B, teacher_seq_len, vocab_size], but typically we gather for top-K
    teacher_logprobs,
    mask,
    num_items_in_batch=None,  # pylint: disable=unused-argument
):
    """
    Wrapper that calls our Triton-based forward+backward for KD.
    For production, you likely want to do the gather (teacher top-K) outside
    or inside a separate kernel. This function expects that you've *already*
    called gather on student_logits -> shape [B, seq_len, K].
    """
    return _KLDivergenceTritonFn.apply(
        student_logits,
        teacher_logprobs,
        mask,  # num_items_in_batch
    )
