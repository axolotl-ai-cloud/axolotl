# Adapted from Unsloth
# https://github.com/unslothai/unsloth/blob/4b97a810b509c93f44be4c037c7aa18fb8922884/unsloth/kernels/cross_entropy_loss.py

import triton
import triton.language as tl
import torch

MAX_FUSED_SIZE = 65536

def calculate_settings(n):
    BLOCK_SIZE = triton.next_power_of_2(n)
    # CUDA only supports 65536 - 2^16 threads per block
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps
pass

@triton.jit
def _cross_entropy_forward(logits_ptr, logits_row_stride,
                           loss_ptr,
                           lse_ptr,
                           labels_ptr,
                           n_cols,
                           BLOCK_SIZE: tl.constexpr,):
    """
        Cross Entropy Loss = 1/n sum [ -yi log(Pi) ]
        Pi = exp(xi) / sum(exp(xi))
        CE_i = -y log(p) = -y log[ exp(x) / sum(exp(x)) ]
             = -y [ x - log[sum(exp(x))] ]
             = y * (log[sum(exp(x))] - x)
        If y == 0: CE_i = 0
        If y == 1: CE_i = logsumexp - x
    """
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_row_stride
    loss_ptr   += row_idx
    lse_ptr    += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # TODO: Fixup int32 locations to int64
    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask = mask, other = -float("inf")).to(tl.float32)
    max_logits = tl.max(logits, 0)
    # Maximum stops overflow
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr, lse)

    if label_idx != -100:
        logits_label = tl.load(logits_ptr + label_idx).to(tl.float32)
        loss = lse - logits_label
    else:
        loss = 0.0
    tl.store(loss_ptr, loss)
pass


@triton.jit
def _cross_entropy_backward(logits_ptr, logits_row_stride,
                            dloss_ptr,   dloss_row_stride,
                            lse_ptr,
                            labels_ptr,
                            n_cols,
                            BLOCK_SIZE: tl.constexpr,):
    """
        CE_i = -y log(P) = y * (log[sum(exp(x))] - x)
        dC/dx = d/dx (y * log[sum(exp(x))] - x * y)

        From https://en.wikipedia.org/wiki/LogSumExp
        d/dx logsumexp = exp(x) / sum(exp(x)) = softmax(x)

        dC/dx = y * exp(x) / sum(exp(x)) - d/dx (x * y)
        dC/dx = y * exp[ log[exp(x) / sum(exp(x))] ] using x = exp(log(x)) trick
        dC/dx = y * exp[x - logsumexp] - d/dx (x * y)

        If y == 0: dC/dx = 0
        If y == 1 and x == label: dC/dlabel = exp[x - logsumexp] - 1
        If y == 1 and x != label: dC/dx     = exp[x - logsumexp]
    """
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_row_stride
    dloss_ptr  += row_idx *  dloss_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    # TODO: Fixup int32 locations to int64
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)

    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask = mask, other = 0).to(tl.float32)
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)

    probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(logits_ptr + col_offsets, dloss * probs, mask = mask)



class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        n_rows, n_cols = logits.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        losses    = torch.empty(n_rows, dtype = torch.float32, device = "cuda")
        logsumexp = torch.empty(n_rows, dtype = torch.float32, device = "cuda")

        _cross_entropy_forward[(n_rows,)](
            logits, logits.stride(0),
            losses,
            logsumexp,
            labels,
            n_cols,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps,
        )

        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(logits, logsumexp, labels)
        return losses
    pass

    @staticmethod
    def backward(ctx, dlosses):
        logits, logsumexp, labels = ctx.saved_tensors
        n_rows, n_cols = logits.shape

        _cross_entropy_backward[(n_rows,)](
            logits,   logits.stride(0),
            dlosses, dlosses.stride(0),
            logsumexp,
            labels,
            n_cols,
            BLOCK_SIZE = ctx.BLOCK_SIZE,
            num_warps  = ctx.num_warps,
        )
        return logits, None, None,
    pass
pass


def fast_cross_entropy_loss(logits, labels):
    """
    Arguments:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len,)
    Returns:
        losses: float
    """
    batch, seq_len, d = logits.shape
    assert(labels.shape == (batch, seq_len))

    loss = CrossEntropyLoss.apply(
        logits.view(batch*seq_len, d),
        labels.view(-1),
    )
    n_items = torch.count_nonzero(labels != -100)
    return loss.sum() / n_items
pass