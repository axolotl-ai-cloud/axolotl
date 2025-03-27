"""
Optimized Triton kernels for logsumexp
"""
# pylint: disable=invalid-name,unused-argument
import triton
import triton.language as tl


# Helper function for computing logsumexp
@triton.jit
def logsumexp_kernel(
    logits_ptr,
    output_ptr,
    B,
    S,
    V,  # batch size, seq len, vocab size
    stride_b,
    stride_s,
    stride_v,
    out_stride_b,
    out_stride_s,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    # pylint: disable=duplicate-code
    pid = tl.program_id(0)
    batch_idx = pid // S
    seq_idx = pid % S

    # Bounds check
    if batch_idx >= B or seq_idx >= S:
        return

    # Pointers
    logits_base = logits_ptr + batch_idx * stride_b + seq_idx * stride_s

    # Find maximum for numerical stability
    max_val = -float("inf")
    for v_offset in range(0, V, BLOCK_SIZE):
        v_size = min(BLOCK_SIZE, V - v_offset)
        mask = tl.arange(0, BLOCK_SIZE) < v_size

        logits_block = tl.load(
            logits_base + (v_offset + tl.arange(0, BLOCK_SIZE)) * stride_v,
            mask=mask,
            other=-float("inf"),
        )
        max_val = tl.maximum(max_val, tl.max(logits_block, axis=0))

    # Compute sum of exp(logit - max_val)
    sum_exp = 0.0
    for v_offset in range(0, V, BLOCK_SIZE):
        v_size = min(BLOCK_SIZE, V - v_offset)
        mask = tl.arange(0, BLOCK_SIZE) < v_size

        logits_block = tl.load(
            logits_base + (v_offset + tl.arange(0, BLOCK_SIZE)) * stride_v,
            mask=mask,
            other=-float("inf"),
        )
        sum_exp += tl.sum(tl.exp(logits_block - max_val), axis=0)

    # Compute logsumexp
    result = max_val + tl.log(sum_exp)

    # Store result
    tl.store(output_ptr + batch_idx * out_stride_b + seq_idx * out_stride_s, result)
