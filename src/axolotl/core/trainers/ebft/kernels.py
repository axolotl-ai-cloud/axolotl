"""
Fused Triton kernels for strided EBFT.

These kernels eliminate intermediate tensor materializations that dominate
the elementwise/fill category (~40% of CUDA time in profiling).

Kernels:
  1. fused_log_softmax_gather: log_softmax + gather in one pass (no full vocab materialization)
  2. fused_masked_reinforce_loss: -logp * advantage * mask, reduced to scalar
  3. fused_cosine_similarity: batched cosine similarity without intermediate tensors
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# 1. Fused log_softmax + gather (selective log softmax)
# ---------------------------------------------------------------------------
# Instead of: log_softmax(logits, dim=-1)  → (B, S, V)  → gather(index=labels)
# We compute: for each (b, s), the log_softmax value at logits[b, s, labels[b, s]]
# This avoids materializing the full (B, S, V) log_softmax output.


@triton.jit
def _fused_log_softmax_gather_kernel(
    logits_ptr,  # (B*S, V) row-major
    labels_ptr,  # (B*S,) int64
    output_ptr,  # (B*S,) float32
    V: tl.constexpr,  # vocab size
    BLOCK_V: tl.constexpr,  # tile width over vocab
):
    """Compute log_softmax(logits)[label] for each row without materializing full output."""
    row = tl.program_id(0)

    logits_row_ptr = logits_ptr + row * V
    label = tl.load(labels_ptr + row)

    # Pass 1: find max for numerical stability
    max_val = -float("inf")
    for v_start in range(0, V, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < V
        vals = tl.load(logits_row_ptr + v_offsets, mask=mask, other=-float("inf"))
        max_val = tl.maximum(max_val, tl.max(vals, axis=0))

    # Pass 2: compute sum(exp(x - max))
    sum_exp = 0.0
    for v_start in range(0, V, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < V
        vals = tl.load(logits_row_ptr + v_offsets, mask=mask, other=-float("inf"))
        sum_exp += tl.sum(tl.exp(vals - max_val), axis=0)

    log_sum_exp = tl.log(sum_exp)

    # Gather: log_softmax[label] = logits[label] - max - log_sum_exp
    target_logit = tl.load(logits_row_ptr + label)
    result = target_logit - max_val - log_sum_exp

    tl.store(output_ptr + row, result)


def fused_log_softmax_gather(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Compute log_softmax(logits, dim=-1).gather(-1, labels) without materializing full output.

    Args:
        logits: (B, S, V) or (B*S, V) float tensor (bf16 or fp32)
        labels: (B, S) or (B*S,) int64 tensor of target indices

    Returns:
        (B, S) or (B*S,) float32 tensor of selected log probabilities
    """
    orig_shape = logits.shape[:-1]
    V = logits.shape[-1]
    logits_2d = logits.reshape(-1, V).contiguous()
    labels_1d = labels.reshape(-1).contiguous()
    n_rows = logits_2d.shape[0]

    output = torch.empty(n_rows, device=logits.device, dtype=torch.float32)

    # Choose BLOCK_V: must be power of 2, large enough for good occupancy
    BLOCK_V = min(triton.next_power_of_2(V), 65536)

    _fused_log_softmax_gather_kernel[(n_rows,)](
        logits_2d,
        labels_1d,
        output,
        V=V,
        BLOCK_V=BLOCK_V,
    )

    return output.view(orig_shape)


# ---------------------------------------------------------------------------
# 2. Fused masked REINFORCE loss reduction
# ---------------------------------------------------------------------------
# Instead of: (-logp * adv * mask).sum() / mask.sum()
# We do the masked multiply-accumulate in one kernel, returning (sum, count).


@triton.jit
def _fused_reinforce_loss_kernel(
    logps_ptr,  # (N,) float32 per-token log probs
    advs_ptr,  # (N,) float32 advantages
    mask_ptr,  # (N,) bool action mask
    partial_sum_ptr,  # (n_blocks,) partial sums
    partial_cnt_ptr,  # (n_blocks,) partial counts
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offsets < N

    logps = tl.load(logps_ptr + offsets, mask=valid, other=0.0)
    advs = tl.load(advs_ptr + offsets, mask=valid, other=0.0)
    m = tl.load(mask_ptr + offsets, mask=valid, other=0).to(tl.float32)

    # -logp * advantage * mask
    loss = -logps * advs * m
    block_sum = tl.sum(loss, axis=0)
    block_cnt = tl.sum(m, axis=0)

    tl.store(partial_sum_ptr + block_id, block_sum)
    tl.store(partial_cnt_ptr + block_id, block_cnt)


def fused_reinforce_loss(
    per_token_logps: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked REINFORCE loss: (-logp * adv * mask).sum() / mask.sum().

    All inputs should be flat or will be flattened. Returns scalar loss tensor.
    """
    logps_flat = per_token_logps.reshape(-1).contiguous()
    advs_flat = advantages.reshape(-1).contiguous()
    mask_flat = action_mask.reshape(-1).contiguous()
    N = logps_flat.shape[0]

    BLOCK_N = 1024
    n_blocks = triton.cdiv(N, BLOCK_N)

    partial_sum = torch.empty(n_blocks, device=logps_flat.device, dtype=torch.float32)
    partial_cnt = torch.empty(n_blocks, device=logps_flat.device, dtype=torch.float32)

    _fused_reinforce_loss_kernel[(n_blocks,)](
        logps_flat,
        advs_flat,
        mask_flat,
        partial_sum,
        partial_cnt,
        N=N,
        BLOCK_N=BLOCK_N,
    )

    total_sum = partial_sum.sum()
    total_cnt = partial_cnt.sum().clamp(min=1)
    return total_sum / total_cnt


# ---------------------------------------------------------------------------
# 3. Fused cosine similarity (batched, for EBFT rewards)
# ---------------------------------------------------------------------------
# Instead of: F.cosine_similarity(gen, gt, dim=-1) which normalizes then dots,
# we fuse the dot product, norm computation, and division into one kernel.


@triton.jit
def _fused_cosine_sim_kernel(
    a_ptr,  # (N, D) first set of vectors
    b_ptr,  # (N, D) second set of vectors
    out_ptr,  # (N,) cosine similarities
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    a_row_ptr = a_ptr + row * D
    b_row_ptr = b_ptr + row * D

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        mask = d_offsets < D
        a_vals = tl.load(a_row_ptr + d_offsets, mask=mask, other=0.0).to(tl.float32)
        b_vals = tl.load(b_row_ptr + d_offsets, mask=mask, other=0.0).to(tl.float32)

        dot += tl.sum(a_vals * b_vals, axis=0)
        norm_a += tl.sum(a_vals * a_vals, axis=0)
        norm_b += tl.sum(b_vals * b_vals, axis=0)

    denom = tl.sqrt(norm_a) * tl.sqrt(norm_b)
    denom = tl.where(denom > 1e-8, denom, 1e-8)
    result = dot / denom

    tl.store(out_ptr + row, result)


def fused_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity along the last dimension.

    Args:
        a, b: (..., D) tensors of the same shape

    Returns:
        (...,) tensor of cosine similarities
    """
    orig_shape = a.shape[:-1]
    D = a.shape[-1]
    a_2d = a.reshape(-1, D).contiguous()
    b_2d = b.reshape(-1, D).contiguous()
    N = a_2d.shape[0]

    output = torch.empty(N, device=a.device, dtype=torch.float32)

    BLOCK_D = min(triton.next_power_of_2(D), 4096)

    _fused_cosine_sim_kernel[(N,)](
        a_2d,
        b_2d,
        output,
        D=D,
        BLOCK_D=BLOCK_D,
    )

    return output.view(orig_shape)


# ---------------------------------------------------------------------------
# 4. Fused pairwise diversity penalty
# ---------------------------------------------------------------------------
# Instead of: bmm(gen, gen.T) → mask diagonal → sum / (n-1)
# We compute the pairwise dot products and exclusion in one kernel.


@triton.jit
def _fused_diversity_kernel(
    emb_ptr,  # (B, N, D) embeddings, row-major
    out_ptr,  # (B, N) diversity penalties
    N: tl.constexpr,  # n_samples
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """For each (b, i), compute mean dot product to all j != i."""
    b = tl.program_id(0)
    i = tl.program_id(1)

    # Pointer to emb[b, i, :]
    emb_bi_ptr = emb_ptr + (b * N + i) * D

    total_sim = 0.0
    for j in range(N):
        emb_bj_ptr = emb_ptr + (b * N + j) * D

        dot = 0.0
        for d_start in range(0, D, BLOCK_D):
            d_offsets = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < D
            a_vals = tl.load(emb_bi_ptr + d_offsets, mask=d_mask, other=0.0).to(
                tl.float32
            )
            b_vals = tl.load(emb_bj_ptr + d_offsets, mask=d_mask, other=0.0).to(
                tl.float32
            )
            dot += tl.sum(a_vals * b_vals, axis=0)

        # Exclude self-similarity (j == i)
        is_other = j != i
        total_sim += dot * is_other

    result = total_sim / (N - 1)
    tl.store(out_ptr + b * N + i, result)


def fused_diversity_penalty(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute mean pairwise dot product (excluding self) per sample.

    Args:
        embeddings: (B, N, D) tensor where N is n_samples

    Returns:
        (B, N) tensor of diversity penalties
    """
    B, N, D = embeddings.shape
    embeddings = embeddings.contiguous()
    output = torch.zeros(B, N, device=embeddings.device, dtype=torch.float32)
    if N <= 1:
        return output  # diversity is 0 when there's only one sample

    BLOCK_D = min(triton.next_power_of_2(D), 4096)

    _fused_diversity_kernel[(B, N)](
        embeddings,
        output,
        N=N,
        D=D,
        BLOCK_D=BLOCK_D,
    )

    return output
