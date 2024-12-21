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
    student_logits_ptr,       # float32[B, seq_len, K] after gather
    teacher_logprobs_ptr,     # float32[B, seq_len, K]
    mask_ptr,                 # bool[B, seq_len, K] or int8
    partial_kd_ptr,           # float32[B, seq_len] (accumulator)
    B,                        # total batch size
    seq_len,                  # total sequence length from teacher
    K,                        # top-K from teacher
    BLOCK_SIZE: tl.constexpr  # how many tokens per block in dimension0
):
    # program_id is the global index for each block
    pid = tl.program_id(0)

    # Each block handles a range of seq positions in [0..B*seq_len)
    block_start = pid * BLOCK_SIZE
    block_end = tl.min((pid+1)*BLOCK_SIZE, B * seq_len)
    length = block_end - block_start

    # Offsets for indexing
    # We want to interpret a linear index in [0..B*seq_len) as (batch_idx, seq_idx)
    # E.g.:
    #   batch_idx = block_start // seq_len
    #   seq_idx   = block_start %  seq_len
    # but we must do this for each element in the block. We'll do that inside a loop.

    # We'll store a running partial KL sum in registers
    # We do a for-loop for each position in the block, then do a thread-level reduction
    kd_reg = 0.0

    # We'll iterate over each item in [block_start, block_end).
    # A more advanced approach can use vectorization / warp-based parallelism inside the block.
    for offset in range(length):
        # Convert offset -> actual index in [0..B*seq_len)
        linear_idx = block_start + offset
        # batch index and sequence index
        b_idx = linear_idx // seq_len
        s_idx = linear_idx % seq_len

        # For K top tokens, read the relevant student logits and teacher logprobs
        # We'll load them in a small loop:
        logsumexp_val = float('-inf')
        # We'll store them in a local array for a second pass
        student_logits_k = [0.0 for _ in range(K)]
        teacher_logprobs_k = [0.0 for _ in range(K)]
        valid_k = [0 for _ in range(K)]

        # gather the top-K logits & teacher logprobs
        for k in range(K):
            # load student logit
            student_val = tl.load(
                student_logits_ptr
                + b_idx*seq_len*K
                + s_idx*K
                + k,
                mask=(b_idx < B) and (s_idx < seq_len)
            )
            teacher_val = tl.load(
                teacher_logprobs_ptr
                + b_idx*seq_len*K
                + s_idx*K
                + k,
                mask=(b_idx < B) and (s_idx < seq_len)
            )
            # get mask
            mask_val = tl.load(
                mask_ptr
                + b_idx*seq_len*K
                + s_idx*K
                + k,
                mask=(b_idx < B) and (s_idx < seq_len)
            )

            student_logits_k[k] = student_val
            teacher_logprobs_k[k] = teacher_val
            valid_k[k] = mask_val

            # track max for logsumexp (naive approach)
            if student_val > logsumexp_val:
                logsumexp_val = student_val

        # now compute logsumexp for the K student logits
        # logsumexp = max_val + log(sum( exp(student_val - max_val) ))
        exp_sum = 0.0
        for k in range(K):
            if valid_k[k] != 0:  # if valid
                exp_sum += float(torch.exp(student_logits_k[k] - logsumexp_val))
        # safe check
        if exp_sum == 0.0:
            exp_sum = 1e-8
        logsumexp_val = logsumexp_val + float(torch.log(torch.tensor(exp_sum)))

        # compute partial kl
        # sum_{k in valid} p^T_k (log p^T_k - log p^S_k)
        # teacher_probs_k = exp(teacher_logprobs_k)
        for k in range(K):
            if valid_k[k] != 0:  # only valid tokens
                teacher_prob = float(torch.exp(teacher_logprobs_k[k]))
                student_logprob = student_logits_k[k] - logsumexp_val
                kd_val = teacher_prob * (teacher_logprobs_k[k] - student_logprob)
                kd_reg += kd_val

    # Write out partial kd for this block. We store a single partial sum in partial_kd_ptr
    # We'll store it at partial_kd_ptr[pid]
    # In real code, you might do an atomic add into partial_kd_ptr or a parallel reduction pass
    # for now, let's just store it at index=pid
    tl.store(partial_kd_ptr + pid, kd_reg)


class _KLDivergenceTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student_logits, teacher_logprobs, mask, num_items_in_batch):
        """
        Inputs shape assumptions (after gather!):
          - student_logits: [B, seq_len, K]
          - teacher_logprobs: [B, seq_len, K]
          - mask: [B, seq_len, K] (bool or 0/1) for valid tokens
        """
        B, seq_len, K = student_logits.shape

        # Prepare output buffer for partial sums
        # We'll have BLOCK_SIZE define how many (batch*seq_len) items each block processes
        # For simplicity, let's aim for one block per 1024 positions
        BLOCK_SIZE = 1024
        # compute how many blocks we need
        total_positions = B * seq_len
        grid = ( (total_positions + BLOCK_SIZE - 1) // BLOCK_SIZE , )

        partial_kd = torch.empty(
            grid[0], dtype=student_logits.dtype, device=student_logits.device
        )

        # Launch kernel
        kd_forward_kernel[grid](
            student_logits,
            teacher_logprobs,
            mask,
            partial_kd,
            B, seq_len, K,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # Sum partials on CPU or GPU
        kd_sum = partial_kd.sum()

        # normalize
        if num_items_in_batch is not None:
            kd_loss = kd_sum / num_items_in_batch
        else:
            # Just average over all valid tokens; in practice you'd need the count of valid tokens
            # For a quick approximation, let's do kd_sum / total_positions (or do a separate reduction on mask)
            # This is a simplification. For correctness, you should count valid tokens in the kernel.
            kd_loss = kd_sum / (total_positions * K)

        # Save context for backward
        # Typically, you'd need to save the raw student_logits, teacher_logprobs, etc. for grad
        # But be mindful of memory usage. We’ll demonstrate the minimal approach here:
        ctx.save_for_backward(student_logits, teacher_logprobs, mask, torch.tensor(num_items_in_batch or 0))
        ctx.B = B
        ctx.seq_len = seq_len
        ctx.K = K
        ctx.total_positions = total_positions
        ctx.BLOCK_SIZE = BLOCK_SIZE

        return kd_loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output is dLoss/dOut (a scalar).
        We want dLoss/dStudentLogits.
        Recall that:

          Loss = sum_{valid k} p^T_k ( log p^T_k - (student_logits_k - logsumexp(student_logits_all_k)) )
                = sum_{valid k} p^T_k log p^T_k - sum_{valid k} p^T_k student_logits_k + sum_{valid k} p^T_k logsumexp(...)

        Let’s break down the derivative wrt student_logits_k. More precisely, from:
          d/d student_logits_k [ - p^T_k student_logprobs_k ]
        you get:
          - p^T_k * ( d/d student_logits_k [ student_logits_k - logsumexp(...) ] )
          = - p^T_k * (1 - p^S_k)
          = p^T_k * p^S_k - p^T_k
          = p^S_k * p^T_k - p^T_k
          = p^T_k( p^S_k - 1 )

        In practice, we also must handle the mask.
        A real implementation typically re-runs the gather & logsumexp calculations or caches them in forward().
        For brevity, we do a naive approach in PyTorch (not Triton) for the backward.
        For maximum speed, you'd do a second Triton kernel.

        We'll do a minimal approach here: recompute everything on the host side or a pure PyTorch pass.
        """
        student_logits, teacher_logprobs, mask, num_items_in_batch_t = ctx.saved_tensors
        num_items_in_batch = int(num_items_in_batch_t.item())
        B, seq_len, K = ctx.B, ctx.seq_len, ctx.K

        # We can either replicate the entire forward logic in PyTorch for gradient
        # or do a second Triton pass. Here, let's do it in PyTorch for clarity.

        # 1) compute logsumexp of student_logits_k for each [b, s]
        # 2) compute p^S_k
        # 3) compute p^T_k from teacher_logprobs
        # 4) dLoss/dStudentLogits = grad_output * p^T_k ( p^S_k - 1 ), masked
        # 5) sum or gather the final gradient

        with torch.enable_grad():
            # treat student_logits as if it requires grad
            stl = student_logits.clone().detach().requires_grad_(True)
            # compute logsumexp along K
            logsumexp_val = torch.logsumexp(stl, dim=-1, keepdim=True)  # [B, seq_len, 1]
            student_logprobs_topk = stl - logsumexp_val
            teacher_probs = teacher_logprobs.exp()
            # p^S_k
            p_s = student_logprobs_topk.exp()

            # forward kl = sum p^T_k ( teacher_logprobs_k - student_logprobs_topk )
            # derivative wrt stl = p^T_k( p^S_k - 1 )
            grad_stl = teacher_probs * (p_s - 1.0)
            # respect the mask
            grad_stl = grad_stl * mask  # zero out invalid

            # sum or average
            if num_items_in_batch != 0:
                grad_stl = grad_stl / num_items_in_batch
            else:
                grad_stl = grad_stl / (B * seq_len * K)  # fallback

            # multiply by upstream grad_output
            grad_stl = grad_stl * grad_output

        return grad_stl, None, None, None


def kd_loss_triton(
    student_logits,  # [B, teacher_seq_len, vocab_size], but typically we gather for top-K
    teacher_logprobs,
    mask,
    num_items_in_batch=None
):
    """
    Wrapper that calls our Triton-based forward+backward for KD.
    For production, you likely want to do the gather (teacher top-K) outside
    or inside a separate kernel. This function expects that you've *already*
    called gather on student_logits -> shape [B, seq_len, K].
    """
    return _KLDivergenceTritonFn.apply(student_logits, teacher_logprobs, mask, num_items_in_batch)
