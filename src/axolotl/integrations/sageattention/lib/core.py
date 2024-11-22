"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any, Optional

import torch
from torch.autograd import Function

from .triton.attn_qk_int8_per_block_causal_varlen import (
    backward as sageattn_varlen_backward,
)
from .triton.attn_qk_int8_per_block_causal_varlen import forward as attn_true_varlen
from .triton.quant_per_block_varlen import (
    per_block_int8 as per_block_int8_varlen_triton,
)


def get_cuda_arch_versions():
    cuda_archs = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_archs.append(f"sm{major}{minor}")
    return cuda_archs


def sageattn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[cu_seqlens_q[-1], num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor, shape: ``[cu_seqlens_k[-1], num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor, shape: ``[cu_seqlens_k[-1], num_kv_heads, head_dim]``.

    cu_seqlens_q : torch.Tensor
        The cumulative sequence lengths for the query sequences in the batch, used to index into `q`.
        Shape: ``[batch_size + 1]``, where each entry represents the cumulative length of sequences up to that batch index.

    cu_seqlens_k : torch.Tensor
        The cumulative sequence lengths for the key and value sequences in the batch, used to index into `k` and `v`.
        Shape: ``[batch_size + 1]``, where each entry represents the cumulative length of sequences up to that batch index.

    max_seqlen_q : int
        The maximum sequence length for the query tensor in the batch.

    max_seqlen_k : int
        The maximum sequence length for the key and value tensors in the batch.

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len for each sequence.
        Default: False.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    Returns
    -------
    torch.Tensor
        The output tensor, shape: ``[cu_seqlens_q[-1], num_qo_heads, head_dim]``.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16``, ``torch.bfloat16`` or ``torch.float32``.
    - The tensors `cu_seqlens_q` and `cu_seqlens_k` must have the dtype ``torch.int32`` or ``torch.int64``.
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    head_dim = q.size(-1)
    assert head_dim in [64, 128], "varlen only support head_dim [64, 128]."

    assert (
        q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    ), "Last dim of qkv must be contiguous."
    assert (
        cu_seqlens_q.is_contiguous() and cu_seqlens_k.is_contiguous()
    ), "cu_seqlens_q and cu_seqlens_k must be contiguous."

    if dtype == torch.bfloat16 or dtype == torch.float32:
        v = v.to(torch.float16)

    if smooth_k:
        km = k.mean(
            dim=0, keepdim=True
        )  # ! km is calculated on the all the batches. Calculate over each individual sequence requires dedicated kernel.
        k -= km

    (
        q_int8,
        q_scale,
        k_int8,
        k_scale,
        cu_seqlens_q_scale,
        cu_seqlens_k_scale,
    ) = per_block_int8_varlen_triton(
        q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sm_scale=sm_scale
    )

    o = attn_true_varlen(
        q_int8,
        k_int8,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        q_scale,
        k_scale,
        cu_seqlens_q_scale,
        cu_seqlens_k_scale,
        output_dtype=dtype,
    )

    return o


class SageAttentionFunction(Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
    ):
        """
        query: Tensor of shape [batch_size, num_heads, seq_len_q, head_dim]
        key: Tensor of shape [batch_size, num_heads, seq_len_k, head_dim]
        value: Tensor of shape [batch_size, num_heads, seq_len_k, head_dim]
        attn_mask: Optional[Tensor], mask tensor
        dropout_p: float, dropout probability
        is_causal: bool, whether to apply causal masking
        scale: Optional[float], scaling factor for attention scores
        """
        # Ensure inputs are contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # Handle default scale
        if scale is None:
            scale = 1.0 / (query.size(-1) ** 0.5)

        # Save parameters needed for backward
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.dropout_p = dropout_p
        ctx.attn_mask = attn_mask

        # Prepare cumulative sequence lengths and max sequence lengths
        # Assuming batch sizes are consistent across query, key, and value
        batch_size, num_heads, seq_len_q, head_dim = query.shape
        seq_len_k = key.shape[2]

        # Flatten batch and head dimensions
        q = query.view(
            -1, seq_len_q, head_dim
        )  # [batch_size * num_heads, seq_len_q, head_dim]
        k = key.view(-1, seq_len_k, head_dim)
        v = value.view(-1, seq_len_k, head_dim)

        # Create cumulative sequence lengths
        cu_seqlens_q = torch.arange(
            0,
            (batch_size * num_heads + 1) * seq_len_q,
            seq_len_q,
            dtype=torch.int32,
            device=query.device,
        )
        cu_seqlens_k = torch.arange(
            0,
            (batch_size * num_heads + 1) * seq_len_k,
            seq_len_k,
            dtype=torch.int32,
            device=key.device,
        )
        max_seqlen_q = seq_len_q
        max_seqlen_k = seq_len_k

        # Call your custom per-block int8 quantization function
        (
            q_int8,
            q_scale,
            k_int8,
            k_scale,
            cu_seqlens_q_scale,
            cu_seqlens_k_scale,
        ) = per_block_int8_varlen_triton(
            q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sm_scale=scale
        )

        # Call your custom attention function
        if is_causal:
            output = attn_true_varlen(
                q_int8,
                k_int8,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                q_scale,
                k_scale,
                cu_seqlens_q_scale,
                cu_seqlens_k_scale,
                output_dtype=query.dtype,
            )
        else:
            raise NotImplementedError("Non-causal attention is not implemented yet.")

        # Reshape output to match the expected shape
        output = output.view(batch_size, num_heads, seq_len_q, head_dim)

        # Save tensors for backward
        ctx.save_for_backward(
            query,
            key,
            value,
            q_int8,
            k_int8,
            q_scale,
            k_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_seqlens_q_scale,
            cu_seqlens_k_scale,
            output,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            query,
            key,
            value,
            q_int8,
            k_int8,
            q_scale,
            k_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_seqlens_q_scale,
            cu_seqlens_k_scale,
            output,
        ) = ctx.saved_tensors

        scale = ctx.scale
        is_causal = ctx.is_causal
        dropout_p = ctx.dropout_p
        attn_mask = ctx.attn_mask

        # Flatten batch and head dimensions
        batch_size, num_heads, seq_len_q, head_dim = query.shape
        seq_len_k = key.shape[2]
        grad_output = grad_output.contiguous()
        do = grad_output.view(-1, seq_len_q, head_dim)

        # Compute gradients w.r.t. q, k, v
        dq, dk, dv = sageattn_varlen_backward(
            do,
            query.view(-1, seq_len_q, head_dim),
            key.view(-1, seq_len_k, head_dim),
            value.view(-1, seq_len_k, head_dim),
            cu_seqlens_q,
            cu_seqlens_k,
            seq_len_q,
            seq_len_k,
            q_int8,
            k_int8,
            q_scale,
            k_scale,
            cu_seqlens_q_scale,
            cu_seqlens_k_scale,
            scale,
            is_causal,
        )

        # Reshape gradients to match the input shapes
        dq = dq.view(batch_size, num_heads, seq_len_q, head_dim)
        dk = dk.view(batch_size, num_heads, seq_len_k, head_dim)
        dv = dv.view(batch_size, num_heads, seq_len_k, head_dim)

        # Handle optional arguments
        d_attn_mask = None  # Assuming attn_mask does not require gradients
        d_dropout_p = (
            None  # Dropout probability is a hyperparameter, typically not optimized
        )
        d_is_causal = None  # Not differentiable
        d_scale = None  # If scale is a tensor and requires grad, compute its gradient

        return dq, dk, dv, d_attn_mask, d_dropout_p, d_is_causal, d_scale


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    """
    Custom scaled dot product attention using SageAttentionFunction.
    """
    return SageAttentionFunction.apply(
        query, key, value, attn_mask, dropout_p, is_causal, scale
    )


def monkeypatch_sdp_w_sage_attention():
    """
    Replace torch.nn.functional.scaled_dot_product_attention with custom scaled dot product attention using SageAttentionFunction.
    """
    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention
