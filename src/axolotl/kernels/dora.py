"""
Triton kernels for DoRA (Weight-Decomposed Low-Rank Adaptation).

Fuses the weight norm computation and magnitude scaling to avoid
materializing the full [out_features, in_features] combined weight matrix.
The B@A product is computed row-by-row inside the kernel.
"""

import torch
import triton
import triton.language as tl

from .quantize import dequantize


@triton.jit
def _dora_fused_norm_kernel(
    # Pointers
    W_ptr,  # base weight [out, in] (dequantized, row-major)
    B_ptr,  # LoRA B [out, rank] (row-major)
    A_ptr,  # LoRA A [rank, in] (row-major)
    mag_ptr,  # magnitude vector [out]
    out_ptr,  # output mag_norm_scale [out]
    # Shapes
    out_features,
    in_features,
    rank,
    # Scaling
    lora_scale,  # float scaling factor
    # Block sizes
    BLOCK_IN: tl.constexpr,
    BLOCK_R: tl.constexpr,  # >= rank, power of 2
):
    """Compute mag_norm_scale[i] = magnitude[i] / ||W[i,:] + s * (B[i,:] @ A)[:] ||_2

    Each program handles one output row. B[row,:] is loaded once (small),
    then we tile over in_features computing the dot product with A[:,tile]
    and accumulating the squared norm.

    This avoids materializing the full [out, in] B@A matrix.
    """
    row = tl.program_id(0)
    if row >= out_features:
        return

    # Accumulate squared norm across tiles of in_features
    norm_sq_acc = tl.zeros([BLOCK_IN], dtype=tl.float32)

    for start in range(0, in_features, BLOCK_IN):
        cols = start + tl.arange(0, BLOCK_IN)
        col_mask = cols < in_features

        # Load W[row, cols]
        w_vals = tl.load(
            W_ptr + row * in_features + cols,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)

        # Compute (B[row,:] @ A[:, cols]) for this tile
        # Load B[row, r] as scalar and A[r, cols] as vector for each r
        ba_vals = tl.zeros([BLOCK_IN], dtype=tl.float32)
        for r in tl.static_range(BLOCK_R):
            # Load scalar B[row, r]
            b_val = tl.load(
                B_ptr + row * rank + r,
                mask=(r < rank),
                other=0.0,
            ).to(tl.float32)
            # Load vector A[r, cols]
            a_vals = tl.load(
                A_ptr + r * in_features + cols,
                mask=(col_mask & (r < rank)),
                other=0.0,
            ).to(tl.float32)
            ba_vals += b_val * a_vals

        # Combined: W + s * (B @ A)
        combined = w_vals + lora_scale * ba_vals

        # Accumulate squared values
        norm_sq_acc += tl.where(col_mask, combined * combined, 0.0)

    # Reduce to scalar norm
    norm_sq = tl.sum(norm_sq_acc, axis=0)
    norm = tl.sqrt(norm_sq + 1e-12)  # epsilon for numerical stability

    # Load magnitude and compute scale
    mag = tl.load(mag_ptr + row).to(tl.float32)
    scale = mag / norm

    tl.store(out_ptr + row, scale)


def triton_dora_scale(
    W: torch.Tensor,
    W_quant,
    A: torch.Tensor,
    B: torch.Tensor,
    s: float,
    magnitude: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute DoRA mag_norm_scale using fused Triton kernel.

    Computes B@A row-by-row inside the kernel, avoiding the full
    [out_features, in_features] materialization.

    Args:
        W: base weight [out, in] (possibly quantized)
        W_quant: quantization state
        A: LoRA A [rank, in]
        B: LoRA B [out, rank]
        s: LoRA scaling factor
        magnitude: learned magnitude [out]
        dtype: compute dtype

    Returns:
        mag_norm_scale: [out] tensor = magnitude / ||W + s * B @ A||_2
    """
    # Dequantize W to [out, in]
    W_full = dequantize(W.t(), W_quant).t().contiguous().to(dtype)

    out_features, in_features = W_full.shape
    rank = A.shape[0]

    out = torch.empty(out_features, dtype=dtype, device=W.device)

    # Block sizes
    BLOCK_IN = triton.next_power_of_2(min(in_features, 2048))
    BLOCK_R = triton.next_power_of_2(rank)

    _dora_fused_norm_kernel[(out_features,)](
        W_full,
        B.contiguous().to(dtype),
        A.contiguous().to(dtype),
        magnitude.contiguous(),
        out,
        out_features=out_features,
        in_features=in_features,
        rank=rank,
        lora_scale=s,
        BLOCK_IN=BLOCK_IN,
        BLOCK_R=BLOCK_R,
    )

    return out.detach()
