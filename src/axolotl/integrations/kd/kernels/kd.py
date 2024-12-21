import torch
import triton
import triton.language as tl

configs = [
    triton.Config({"BLOCK_SIZE": 32}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE": 64}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=2),
    # Add more if needed
]


@triton.autotune(configs=configs, key=["N", "K"])
@triton.jit
def fwd_kl_topk_kernel(
    teacher_lp_ptr,  # float32 [N, K]
    student_lp_ptr,  # float32 [N, K]
    mask_ptr,  # bool    [N, K]
    loss_out_ptr,  # float32 [N]
    stride_tn,
    stride_tk,
    stride_sn,
    stride_sk,
    stride_mn,
    stride_mk,
    stride_loss_n,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each kernel instance: row_id = tl.program_id(0). We'll tile the K dimension in chunks of BLOCK_SIZE.
    Summation => store into loss_out[row_id].
    """
    row_id = tl.program_id(0)
    if row_id >= N:
        return

    # Base pointers for teacher, student, mask rows
    t_row_ptr = teacher_lp_ptr + row_id * stride_tn
    s_row_ptr = student_lp_ptr + row_id * stride_sn
    m_row_ptr = mask_ptr + row_id * stride_mn

    # We'll accumulate KL in local variable
    kl_sum = 0.0

    # tile the K dimension
    num_tiles = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in range(num_tiles):
        k_offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = k_offset < K

        # load teacher logprobs
        t_lp = tl.load(t_row_ptr + k_offset * stride_tk, mask=mask, other=-float("inf"))
        # load student logprobs
        s_lp = tl.load(s_row_ptr + k_offset * stride_sk, mask=mask, other=-float("inf"))

        # load mask => bool
        valid = tl.load(
            m_row_ptr + k_offset * stride_mk, mask=mask, other=0
        )  # 0 or 1 => bool
        valid_bool = valid.to(tl.int1)

        # teacher probs
        t_p = tl.exp(t_lp)

        # local_kl = p^T * (lp^T - lp^S) * valid
        local_kl = t_p * (t_lp - s_lp)
        # sum only over valid positions
        kl_sum += tl.sum(local_kl, where=valid_bool)

    # store rowwise result
    tl.store(loss_out_ptr + row_id * stride_loss_n, kl_sum)


@triton.autotune(configs=configs, key=["N", "K"])
@triton.jit
def bwd_kl_topk_kernel(
    teacher_lp_ptr,  # float32 [N, K]
    mask_ptr,  # bool    [N, K]
    grad_stud_ptr,  # float32 [N, K], output
    stride_tn,
    stride_tk,
    stride_mn,
    stride_mk,
    stride_gn,
    stride_gk,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    For forward KL, d/d(student_lp) = - exp(teacher_lp), if mask=1, else 0.
    Each kernel instance processes one row [K].
    """
    row_id = tl.program_id(0)
    if row_id >= N:
        return

    t_row_ptr = teacher_lp_ptr + row_id * stride_tn
    m_row_ptr = mask_ptr + row_id * stride_mn
    g_row_ptr = grad_stud_ptr + row_id * stride_gn

    num_tiles = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in range(num_tiles):
        k_offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = k_offset < K

        t_lp = tl.load(t_row_ptr + k_offset * stride_tk, mask=mask, other=-float("inf"))
        valid = tl.load(m_row_ptr + k_offset * stride_mk, mask=mask, other=0).to(
            tl.int1
        )

        grad_val = -tl.exp(t_lp)  # derivative
        grad_val = tl.where(valid, grad_val, 0.0)

        tl.store(g_row_ptr + k_offset * stride_gk, grad_val, mask=mask)


class FwdKLTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        teacher_lp_topk: torch.Tensor,
        student_lp_topk: torch.Tensor,
        mask_topk: torch.Tensor,
        reduction: str = "batchmean",
    ) -> torch.Tensor:
        """
        teacher_lp_topk: [N, K]
        student_lp_topk: [N, K]
        mask_topk:       [N, K] bool
        returns either scalar (if batchmean) or [N] if 'none'
        """
        assert teacher_lp_topk.shape == student_lp_topk.shape
        assert teacher_lp_topk.shape == mask_topk.shape

        N, K = teacher_lp_topk.shape
        dev = teacher_lp_topk.device
        dtype = teacher_lp_topk.dtype

        # Contiguous
        t_lp_c = teacher_lp_topk.contiguous()
        s_lp_c = student_lp_topk.contiguous()
        m_c = mask_topk.contiguous()

        # [N] rowwise sums
        loss_out = torch.empty(N, dtype=torch.float32, device=dev)

        grid = (N,)

        fwd_kl_topk_kernel[grid](
            t_lp_c,
            s_lp_c,
            m_c,
            loss_out,
            # strides
            t_lp_c.stride(0),
            t_lp_c.stride(1),
            s_lp_c.stride(0),
            s_lp_c.stride(1),
            m_c.stride(0),
            m_c.stride(1),
            loss_out.stride(0),
            N=N,
            K=K
            # BLOCK_SIZE, warps, stages => autotune
        )

        if reduction == "batchmean":
            loss_val = loss_out.mean()
        elif reduction == "none":
            loss_val = loss_out
        else:
            raise ValueError("reduction must be 'batchmean' or 'none'")

        # Save for backward
        ctx.save_for_backward(t_lp_c, m_c)
        ctx.reduction = reduction
        ctx.shape = (N, K)

        return loss_val

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is either scalar ([1]) if batchmean, or shape=[N] if 'none'
        t_lp_c, m_c = ctx.saved_tensors
        (N, K) = ctx.shape

        # We'll create a grad for the student's top-K logprobs
        grad_stud = torch.empty_like(t_lp_c)  # [N, K]

        grid = (N,)
        bwd_kl_topk_kernel[grid](
            t_lp_c,
            m_c,
            grad_stud,
            t_lp_c.stride(0),
            t_lp_c.stride(1),
            m_c.stride(0),
            m_c.stride(1),
            grad_stud.stride(0),
            grad_stud.stride(1),
            N=N,
            K=K,
        )

        # Multiply by grad_output
        # If batchmean => scalar
        # If none => shape=[N]
        if grad_output.numel() == 1:
            grad_stud *= grad_output
        else:
            # shape=[N], broadcast over K
            grad_stud *= grad_output.unsqueeze(1)

        return grad_stud, None, None, None


def forward_kl_topk(
    teacher_lp_topk: torch.Tensor,
    student_lp_topk: torch.Tensor,
    mask_topk: torch.Tensor,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """
    Calls the autograd function that launches Triton kernels for forward + backward.
    """
    return FwdKLTopKFunction.apply(
        teacher_lp_topk, student_lp_topk, mask_topk, reduction
    )


def prepare_topk_student_teacher(
    student_logits: torch.Tensor,  # [B, teacher_seq_len, vocab_size]
    target_token_ids: torch.Tensor,  # [B, teacher_seq_len, K]
    target_logprobs: torch.Tensor,  # [B, teacher_seq_len, K], teacher logprobs
    target_mask: torch.Tensor,  # [B, teacher_seq_len, K], bool or 0/1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gathers student logits for the teacher's top-K tokens and flattens the first 2 dims => N = B * teacher_seq_len.

    Returns:
      (student_lp_topk, teacher_lp_topk, valid_mask) each shape = [N, K].
    """
    B, S, K = target_token_ids.shape
    # Gather the student logits => [B, S, K]
    # 1) slice or use the entire student_logits if it matches teacher_seq_len
    student_logits_for_kd = student_logits[:, :S, :]  # ensure alignment if needed

    # 2) gather top-k => [B, S, K]
    student_logits_topk = torch.gather(
        student_logits_for_kd, dim=-1, index=target_token_ids
    )

    # 3) convert student logits to logprobs => [B, S, K]
    student_logprobs_topk = student_logits_topk - torch.logsumexp(
        student_logits_topk, dim=-1, keepdim=True
    )

    # Flatten batch dimension
    N = B * S
    student_logprobs_topk_f = student_logprobs_topk.view(N, K)  # [N, K]
    teacher_logprobs_topk_f = target_logprobs.view(N, K)  # [N, K]
    mask_f = target_mask.view(N, K).bool()  # [N, K]

    return student_logprobs_topk_f, teacher_logprobs_topk_f, mask_f
