"""
Fused linear cross-entropy for MemFT-OT (only-threshold).

Computes the MemFT token-weighted loss

    L = sum_i w_i * CE_i / (sum_i w_i + eps),   w_i = 1[CE_i > L_crit] * 1[label_i != ignore]

directly from hidden states and the LM-head weight, without ever materializing
the full [N, V] logits tensor. Logits are formed one token-chunk at a time; a
Triton kernel computes the per-token loss and overwrites the chunk in-place with
the (unnormalized) logit gradient, which is immediately projected down to
hidden-state and weight gradients. Peak extra memory is one chunk of logits
instead of all of them.

Because the OT weights are detached, the normalizer sum_i w_i is a scalar w.r.t.
autograd, so a single forward pass over the chunks is exact: gradients are
accumulated unnormalized and divided by (sum_w + eps) once at the end.

Modelled on Liger's fused_linear_cross_entropy.
"""

import math

import torch
import triton
import triton.language as tl

LN2 = math.log(2.0)

# target number of logit elements per token-chunk (~512MB at bf16); caps the
# peak [chunk, vocab] allocation while keeping chunks large enough for the
# matmuls to stay efficient.
_MEMFT_CHUNK_ELEMS = 2**28


@triton.jit
def _memft_ce_kernel(
    logits_ptr,  # [C, V]  overwritten in-place with grad_logits (unnormalized)
    logits_row_stride,
    labels_ptr,  # [C]
    loss_ptr,  # [C]  out: w_i * CE_i
    weight_ptr,  # [C]  out: w_i
    n_cols,
    critical_loss,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    logits_ptr += row * logits_row_stride
    label = tl.load(labels_ptr + row)

    # treat the ignore sentinel and any out-of-range id as masked, so the
    # unmasked label load below can never go out of bounds
    if (label == ignore_index) or (label < 0) or (label >= n_cols):
        for off in range(0, n_cols, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            tl.store(logits_ptr + cols, 0.0, mask=cols < n_cols)
        tl.store(loss_ptr + row, 0.0)
        tl.store(weight_ptr + row, 0.0)
        return

    # online max + sum(exp) in fp32
    m = -float("inf")
    d = 0.0
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(logits_ptr + cols, mask=cols < n_cols, other=-float("inf")).to(
            tl.float32
        )
        block_max = tl.max(x)
        new_m = tl.maximum(m, block_max)
        d = d * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m))
        m = new_m

    label_logit = tl.load(logits_ptr + label).to(tl.float32)
    logd = tl.log(d)
    ce = logd + m - label_logit  # -log_softmax[label]

    w = tl.where(ce > critical_loss, 1.0, 0.0)

    # grad_logits = w * (softmax - onehot); written in-place, unnormalized
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(logits_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        softmax = tl.exp(x - m) / d
        grad = w * softmax
        grad = tl.where(cols == label, grad - w, grad)
        tl.store(logits_ptr + cols, grad, mask=mask)

    tl.store(loss_ptr + row, w * ce)
    tl.store(weight_ptr + row, w)


def _memft_ce_chunk(logits, labels, critical_loss, ignore_index):
    """Run the Triton CE kernel on a [C, V] chunk; overwrites ``logits`` with grad."""
    n_rows, n_cols = logits.shape
    block_size = min(8192, triton.next_power_of_2(n_cols))
    loss = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    weight = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    _memft_ce_kernel[(n_rows,)](
        logits,
        logits.stride(0),
        labels,
        loss,
        weight,
        n_cols,
        float(critical_loss),
        int(ignore_index),
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return loss, weight


class MemFTLinearCrossEntropy(torch.autograd.Function):
    """Fused linear + MemFT-OT cross-entropy.

    forward(hidden [N, H], weight [V, H], labels [N]) -> scalar loss.
    """

    @staticmethod
    def forward(
        ctx, hidden, weight, labels, critical_loss, epsilon, ignore_index, chunk_tokens
    ):
        n_tokens, _hidden_size = hidden.shape
        vocab_size = weight.shape[0]
        device = hidden.device

        # bigger chunks mean fewer, larger matmuls (speed) and fewer grad_weight
        # accumulation steps (precision), traded against the peak [chunk, vocab]
        # logit allocation (memory). When chunk_tokens is not given, pick the
        # largest chunk whose logit slice stays within a fixed element budget.
        if chunk_tokens is not None and chunk_tokens > 0:
            chunk_size = chunk_tokens
        else:
            chunk_size = triton.next_power_of_2(
                max(1, _MEMFT_CHUNK_ELEMS // vocab_size)
            )
        chunk_size = max(1, min(chunk_size, n_tokens))
        num_chunks = math.ceil(n_tokens / chunk_size)

        grad_hidden = torch.zeros_like(hidden)
        # accumulate the weight gradient in fp32: it sums one contribution per
        # chunk, so a low-precision running sum would drift with num_chunks.
        grad_weight = torch.zeros(weight.shape, dtype=torch.float32, device=device)

        loss_unnorm = torch.zeros((), dtype=torch.float32, device=device)
        sum_w = torch.zeros((), dtype=torch.float32, device=device)

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, n_tokens)
            hidden_c = hidden[start:end]
            labels_c = labels[start:end]

            logits_c = hidden_c @ weight.t()  # [C, V]
            loss_c, weight_c = _memft_ce_chunk(
                logits_c, labels_c, critical_loss, ignore_index
            )
            # logits_c now holds unnormalized grad_logits
            grad_logits_c = logits_c.to(weight.dtype)
            grad_hidden[start:end] = grad_logits_c @ weight
            grad_weight += (grad_logits_c.t() @ hidden_c).float()

            loss_unnorm += loss_c.sum()
            sum_w += weight_c.sum()

        denom = sum_w + epsilon
        loss = loss_unnorm / denom
        grad_hidden /= denom.to(grad_hidden.dtype)
        grad_weight = (grad_weight / denom).to(weight.dtype)

        ctx.save_for_backward(grad_hidden, grad_weight)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_hidden, grad_weight = ctx.saved_tensors
        grad_hidden = grad_output * grad_hidden
        grad_weight = grad_output * grad_weight
        return grad_hidden, grad_weight, None, None, None, None, None


def memft_linear_cross_entropy(
    hidden,
    lm_head_weight,
    labels,
    critical_loss=LN2,
    epsilon=1e-8,
    ignore_index=-100,
    chunk_tokens=None,
):
    """MemFT-OT loss fused with the LM-head projection.

    Args:
        hidden: ``[..., H]`` final hidden states (already shifted so position i
            predicts ``labels[i]``).
        lm_head_weight: ``[V, H]`` LM-head weight.
        labels: ``[...]`` target ids, ``ignore_index`` where masked.
        chunk_tokens: explicit token-chunk size; ``None`` uses the element-budget
            heuristic (~2048 tokens at the Llama-3 vocab).
    """
    hidden = hidden.reshape(-1, hidden.shape[-1])
    labels = labels.reshape(-1).to(hidden.device)
    return MemFTLinearCrossEntropy.apply(
        hidden,
        lm_head_weight,
        labels,
        critical_loss,
        epsilon,
        ignore_index,
        chunk_tokens,
    )
