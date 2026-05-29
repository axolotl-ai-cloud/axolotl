# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _entropy_online_kernel(
    logits_ptr,
    output_ptr,
    stride_row,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Online entropy: single pass with running max correction."""
    row = tl.program_id(0)
    row_ptr = logits_ptr + tl.cast(row, tl.int64) * stride_row

    running_max = tl.full([], float("-inf"), dtype=tl.float32)
    running_sum_exp = tl.full([], 0.0, dtype=tl.float32)
    running_weighted = tl.full([], 0.0, dtype=tl.float32)

    for v_start in range(0, V, BLOCK_V):
        offs = v_start + tl.arange(0, BLOCK_V)
        mask = offs < V
        x = tl.load(row_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)

        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(running_max, block_max)

        correction = tl.exp(running_max - new_max)
        running_sum_exp = running_sum_exp * correction
        running_weighted = running_weighted * correction

        exp_x = tl.exp(x - new_max)
        exp_x = tl.where(mask, exp_x, 0.0)
        x = tl.where(mask, x, 0.0)
        running_sum_exp += tl.sum(exp_x, axis=0)
        running_weighted += tl.sum(exp_x * x, axis=0)

        running_max = new_max

    entropy = tl.log(running_sum_exp) + running_max - running_weighted / running_sum_exp
    tl.store(output_ptr + row, entropy)


@triton.jit
def _entropy_online_kernel_strided(
    logits_ptr,
    output_ptr,
    stride_outer,
    stride_inner,
    n_inner,
    row_offset,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Online entropy for non-contiguous 3D (B, L, V) tensors."""
    local_row = tl.program_id(0)
    row = local_row + row_offset
    outer_idx = row // n_inner
    inner_idx = row % n_inner
    off = outer_idx.to(tl.int64) * stride_outer + inner_idx.to(tl.int64) * stride_inner
    row_ptr = logits_ptr + off

    running_max = tl.full([], float("-inf"), dtype=tl.float32)
    running_sum_exp = tl.full([], 0.0, dtype=tl.float32)
    running_weighted = tl.full([], 0.0, dtype=tl.float32)

    for v_start in range(0, V, BLOCK_V):
        offs = v_start + tl.arange(0, BLOCK_V)
        mask = offs < V
        x = tl.load(row_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)

        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(running_max, block_max)

        correction = tl.exp(running_max - new_max)
        running_sum_exp = running_sum_exp * correction
        running_weighted = running_weighted * correction

        exp_x = tl.exp(x - new_max)
        exp_x = tl.where(mask, exp_x, 0.0)
        x = tl.where(mask, x, 0.0)
        running_sum_exp += tl.sum(exp_x, axis=0)
        running_weighted += tl.sum(exp_x * x, axis=0)

        running_max = new_max

    entropy = tl.log(running_sum_exp) + running_max - running_weighted / running_sum_exp
    tl.store(output_ptr + local_row, entropy)


def entropy_from_logits(logits: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    """Triton-fused entropy (online single-pass). Handles non-contiguous tensors without copying."""
    original_shape = logits.shape[:-1]
    V = logits.shape[-1]
    N = 1
    for s in original_shape:
        N *= s

    if not logits.is_cuda:
        # CPU fallback: stable entropy via log_softmax
        logp = F.log_softmax(logits.float(), dim=-1)
        ent = -(logp.exp() * logp).sum(dim=-1)
        return ent.to(logits.dtype).reshape(original_shape)

    output = torch.empty(N, device=logits.device, dtype=torch.float32)

    BLOCK_V = 4096
    MAX_GRID_CONTIG = 8192
    MAX_GRID_STRIDED = 2048

    # Vocab (last) dim must be contiguous for coalesced loads
    if logits.stride(-1) != 1:
        logits = logits.contiguous()

    if logits.is_contiguous():
        flat_logits = logits.reshape(-1, V)
        stride = flat_logits.stride(0)
        for start in range(0, N, MAX_GRID_CONTIG):
            n_rows = min(MAX_GRID_CONTIG, N - start)
            _entropy_online_kernel[(n_rows,)](
                flat_logits[start], output[start], stride, V=V, BLOCK_V=BLOCK_V
            )
    elif logits.ndim == 3:
        stride_outer = logits.stride(0)
        stride_inner = logits.stride(1)
        n_inner = logits.shape[1]
        for start in range(0, N, MAX_GRID_STRIDED):
            n_rows = min(MAX_GRID_STRIDED, N - start)
            _entropy_online_kernel_strided[(n_rows,)](
                logits,
                output[start],
                stride_outer,
                stride_inner,
                n_inner,
                start,
                V=V,
                BLOCK_V=BLOCK_V,
            )
    else:
        logits = logits.contiguous()
        flat_logits = logits.reshape(-1, V)
        stride = flat_logits.stride(0)
        for start in range(0, N, MAX_GRID_CONTIG):
            n_rows = min(MAX_GRID_CONTIG, N - start)
            _entropy_online_kernel[(n_rows,)](
                flat_logits[start], output[start], stride, V=V, BLOCK_V=BLOCK_V
            )

    return output.to(logits.dtype).reshape(original_shape)


# ---------------------------------------------------------------------------
# selective_log_softmax — fused forward + backward Triton kernels
# ---------------------------------------------------------------------------


def selective_log_softmax_original(logits, index) -> torch.Tensor:
    """Original selective_log_softmax (reference/fallback)."""
    squeeze = index.ndim == logits.ndim - 1
    if squeeze:
        index = index.unsqueeze(-1)

    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index)
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values.unsqueeze(-1)
    else:
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index, strict=True):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)

    if squeeze:
        per_token_logps = per_token_logps.squeeze(-1)

    return per_token_logps


@triton.jit
def _selective_logsoftmax_fwd_kernel(
    logits_ptr,
    index_ptr,
    output_ptr,
    logsumexp_ptr,
    stride_logits_row,
    stride_index_row,
    stride_output_row,
    actual_K,
    K_BLOCK: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Forward: online logsumexp + gather. Saves logsumexp for backward."""
    row = tl.program_id(0)
    logits_row_ptr = logits_ptr + tl.cast(row, tl.int64) * stride_logits_row

    # Online logsumexp
    running_max = tl.full([], float("-inf"), dtype=tl.float32)
    running_sum_exp = tl.full([], 0.0, dtype=tl.float32)

    for v_start in range(0, V, BLOCK_V):
        offs = v_start + tl.arange(0, BLOCK_V)
        mask = offs < V
        x = tl.load(logits_row_ptr + offs, mask=mask, other=float("-inf")).to(
            tl.float32
        )

        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(running_max, block_max)
        running_sum_exp = running_sum_exp * tl.exp(running_max - new_max)

        exp_x = tl.exp(x - new_max)
        exp_x = tl.where(mask, exp_x, 0.0)
        running_sum_exp += tl.sum(exp_x, axis=0)
        running_max = new_max

    lse = tl.log(running_sum_exp) + running_max
    tl.store(logsumexp_ptr + row, lse)

    # Gather and subtract
    index_row_ptr = index_ptr + tl.cast(row, tl.int64) * stride_index_row
    output_row_ptr = output_ptr + tl.cast(row, tl.int64) * stride_output_row

    k_offs = tl.arange(0, K_BLOCK)
    k_mask = k_offs < actual_K
    indices = tl.load(index_row_ptr + k_offs, mask=k_mask, other=0).to(tl.int64)
    valid_mask = k_mask & (indices >= 0) & (indices < V)
    safe_indices = tl.where(valid_mask, indices, 0)
    selected = tl.load(logits_row_ptr + safe_indices, mask=valid_mask, other=0.0).to(
        tl.float32
    )
    tl.store(output_row_ptr + k_offs, selected - lse, mask=valid_mask)


@triton.jit
def _selective_logsoftmax_bwd_kernel(
    grad_output_ptr,
    logits_ptr,
    index_ptr,
    logsumexp_ptr,
    grad_logits_ptr,
    stride_grad_out_row,
    stride_logits_row,
    stride_index_row,
    stride_grad_logits_row,
    actual_K,
    K_BLOCK: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Backward: d_logits[j] = -softmax(x)[j] * sum(grad_out) + (grad_out[k] if j == index[k]).

    Single fused pass over V. For each tile, computes the base gradient and adds
    scatter contributions inline by checking which indices fall in the current tile.
    No separate scatter pass — no read-after-write issues.
    """
    row = tl.program_id(0)
    logits_row_ptr = logits_ptr + tl.cast(row, tl.int64) * stride_logits_row
    grad_logits_row_ptr = (
        grad_logits_ptr + tl.cast(row, tl.int64) * stride_grad_logits_row
    )
    grad_out_row_ptr = grad_output_ptr + tl.cast(row, tl.int64) * stride_grad_out_row
    index_row_ptr = index_ptr + tl.cast(row, tl.int64) * stride_index_row

    lse = tl.load(logsumexp_ptr + row).to(tl.float32)

    # Load grad_output and indices (K_BLOCK elements, masked)
    k_offs = tl.arange(0, K_BLOCK)
    k_mask = k_offs < actual_K
    grad_out = tl.load(grad_out_row_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)
    indices = tl.load(
        index_row_ptr + k_offs, mask=k_mask, other=-1
    )  # -1 = never matches
    valid_mask = k_mask & (indices >= 0) & (indices < V)
    grad_out = tl.where(valid_mask, grad_out, 0.0)
    indices = tl.where(valid_mask, indices, -1)
    grad_sum = tl.sum(grad_out, axis=0)

    # Fused pass: for each tile, compute -softmax * grad_sum + scatter
    for v_start in range(0, V, BLOCK_V):
        offs = v_start + tl.arange(0, BLOCK_V)  # [BLOCK_V]
        mask = offs < V
        x = tl.load(logits_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        softmax_j = tl.exp(x - lse)
        softmax_j = tl.where(mask, softmax_j, 0.0)
        grad_j = -softmax_j * grad_sum

        # Scatter: check which selected indices fall in this tile
        # offs: [BLOCK_V], indices: [K_BLOCK]
        # Broadcast: offs[:, None] == indices[None, :] → [BLOCK_V, K_BLOCK]
        match = offs[:, None] == indices[None, :]  # [BLOCK_V, K_BLOCK]
        # Sum grad_out contributions: for each position j, sum grad_out[k] where index[k]==j
        scatter_contrib = tl.sum(
            tl.where(match, grad_out[None, :], 0.0), axis=1
        )  # [BLOCK_V]
        grad_j += scatter_contrib

        tl.store(grad_logits_row_ptr + offs, grad_j, mask=mask)


class _SelectiveLogSoftmaxTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, flat_logits, flat_index, K, K_BLOCK, V, BLOCK_V, MAX_GRID):
        N = flat_logits.shape[0]
        output = torch.empty(N, K_BLOCK, device=flat_logits.device, dtype=torch.float32)
        logsumexp = torch.empty(N, device=flat_logits.device, dtype=torch.float32)

        for start in range(0, N, MAX_GRID):
            n_rows = min(MAX_GRID, N - start)
            _selective_logsoftmax_fwd_kernel[(n_rows,)](
                flat_logits[start],
                flat_index[start],
                output[start],
                logsumexp[start],
                flat_logits.stride(0),
                flat_index.stride(0),
                output.stride(0),
                K,
                K_BLOCK=K_BLOCK,
                V=V,
                BLOCK_V=BLOCK_V,
            )

        ctx.save_for_backward(flat_logits, flat_index, logsumexp)
        ctx.K = K
        ctx.K_BLOCK = K_BLOCK
        ctx.V = V
        ctx.BLOCK_V = BLOCK_V
        ctx.MAX_GRID = MAX_GRID
        return output

    @staticmethod
    def backward(ctx, grad_output):
        flat_logits, flat_index, logsumexp = ctx.saved_tensors
        K, K_BLOCK, V, BLOCK_V, MAX_GRID = (
            ctx.K,
            ctx.K_BLOCK,
            ctx.V,
            ctx.BLOCK_V,
            ctx.MAX_GRID,
        )
        N = flat_logits.shape[0]

        grad_logits = torch.empty_like(flat_logits)

        # grad_output may have K_BLOCK cols; backward kernel reads actual_K
        grad_output_contig = grad_output.contiguous()

        for start in range(0, N, MAX_GRID):
            n_rows = min(MAX_GRID, N - start)
            _selective_logsoftmax_bwd_kernel[(n_rows,)](
                grad_output_contig[start],
                flat_logits[start],
                flat_index[start],
                logsumexp[start],
                grad_logits[start],
                grad_output_contig.stride(0),
                flat_logits.stride(0),
                flat_index.stride(0),
                grad_logits.stride(0),
                K,
                K_BLOCK=K_BLOCK,
                V=V,
                BLOCK_V=BLOCK_V,
            )

        # Return grads for: flat_logits, flat_index, K, K_BLOCK, V, BLOCK_V, MAX_GRID
        return grad_logits, None, None, None, None, None, None


def selective_log_softmax(logits, index) -> torch.Tensor:
    """
    Fused selective_log_softmax with Triton forward+backward kernels.

    Equivalent to: torch.gather(logits.log_softmax(-1), dim=-1, index=index)
    """
    squeeze = index.ndim == logits.ndim - 1
    if squeeze:
        index = index.unsqueeze(-1)

    if not logits.is_cuda or logits.dtype == torch.float64:
        # Triton kernel computes in float32; fall back for float64 and CPU
        return selective_log_softmax_original(
            logits, index.squeeze(-1) if squeeze else index
        )

    V = logits.shape[-1]
    K = index.shape[-1]
    original_index_shape = index.shape

    try:
        flat_logits = logits.view(-1, V)
    except RuntimeError:
        flat_logits = logits.reshape(-1, V).contiguous()
    flat_index = index.reshape(-1, K).contiguous()

    BLOCK_V = 4096
    MAX_GRID = 8192
    K_BLOCK = max(1, triton.next_power_of_2(K))

    output = _SelectiveLogSoftmaxTriton.apply(
        flat_logits, flat_index, K, K_BLOCK, V, BLOCK_V, MAX_GRID
    )

    if K_BLOCK != K:
        output = output[:, :K]

    per_token_logps = output.to(logits.dtype).reshape(original_index_shape)

    if squeeze:
        per_token_logps = per_token_logps.squeeze(-1)

    return per_token_logps
