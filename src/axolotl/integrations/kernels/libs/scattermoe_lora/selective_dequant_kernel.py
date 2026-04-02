"""
Triton kernel for fused selective expert gather + NF4 dequantization.

Instead of:
  1. Gather packed uint8 data for active experts (memory copy)
  2. Gather absmax for active experts (memory copy)
  3. Call BnB dequantize_4bit CUDA kernel

This kernel does all three in one pass:
  - Reads packed NF4 bytes from expert-strided positions
  - Looks up the NF4 codebook
  - Multiplies by the per-block absmax
  - Writes bf16 output directly

This eliminates the intermediate gather buffer entirely.
"""

import torch
import triton
import triton.language as tl

# NF4 codebook (16 values, precomputed by BnB)
# These are the normalized float4 reconstruction values
NF4_CODEBOOK = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]


@triton.jit
def _selective_dequant_nf4_kernel(
    # Input: packed NF4 data (flattened, expert-major order)
    packed_ptr,
    # Input: absmax values (flattened, expert-major order)
    absmax_ptr,
    # Input: active expert indices
    active_experts_ptr,
    # Input: NF4 codebook (16 float values)
    codebook_ptr,
    # Output: dequantized bf16 weights [num_active, expert_numel]
    out_ptr,
    stride_out_e,  # stride for expert dim in output
    # Dimensions
    num_active,
    packed_per_expert,  # expert_numel // 2
    blocks_per_expert,  # expert_numel // blocksize
    blocksize: tl.constexpr,
    # Tile size
    BLOCK_SIZE: tl.constexpr,  # elements per thread block (must be multiple of 2)
):
    """
    Each program processes BLOCK_SIZE elements from one expert.

    Grid: (num_active, cdiv(expert_numel, BLOCK_SIZE))

    For each output element:
      1. Compute which byte in packed data contains this element
      2. Extract the 4-bit nibble (high or low)
      3. Look up in NF4 codebook
      4. Scale by absmax for this block
    """
    expert_local_idx = tl.program_id(0)  # which active expert (0..num_active-1)
    block_id = tl.program_id(1)  # which element block

    # Load the global expert index
    expert_global = tl.load(active_experts_ptr + expert_local_idx).to(tl.int64)

    expert_numel = packed_per_expert * 2  # 2 elements per packed byte
    elem_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = elem_offset < expert_numel

    # Each element is packed as: byte[i//2], low nibble for even i, high for odd i
    byte_idx = elem_offset // 2
    is_high = (elem_offset % 2) == 1

    # Read packed bytes from the global expert's region
    packed_global_offset = expert_global * packed_per_expert + byte_idx
    packed_bytes = tl.load(packed_ptr + packed_global_offset, mask=mask, other=0).to(
        tl.int32
    )

    # Extract 4-bit nibble
    # BnB packing: high nibble = even element, low nibble = odd element
    nibble = tl.where(is_high, packed_bytes & 0xF, (packed_bytes >> 4) & 0xF)

    # NF4 codebook lookup
    # Load all 16 codebook values (small, fits in registers)
    # Use gather from codebook pointer
    code_val = tl.load(codebook_ptr + nibble, mask=mask, other=0.0)

    # Load absmax for this element's quantization block
    block_idx = elem_offset // blocksize
    absmax_global_offset = expert_global * blocks_per_expert + block_idx
    absmax_val = tl.load(absmax_ptr + absmax_global_offset, mask=mask, other=1.0)

    # Dequantize: value = codebook[nibble] * absmax
    result = code_val * absmax_val

    # Store to output
    out_offset = expert_local_idx * stride_out_e + elem_offset
    tl.store(out_ptr + out_offset, result.to(out_ptr.dtype.element_ty), mask=mask)


def selective_dequant_nf4_triton(
    packed_data: torch.Tensor,
    absmax: torch.Tensor,
    active_experts: torch.Tensor,
    expert_shape: tuple[int, int],
    blocksize: int,
    dtype: torch.dtype = torch.bfloat16,
    codebook: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused selective gather + NF4 dequantization via Triton kernel.

    Args:
        packed_data: Flattened packed NF4 data [total_packed] or [total_packed, 1]
        absmax: Per-block scaling factors [total_blocks]
        active_experts: Sorted indices of experts to dequantize [num_active]
        expert_shape: (dim1, dim2) per expert
        blocksize: Quantization block size
        dtype: Output dtype (default bf16)
        codebook: NF4 lookup table [16] (uses default NF4 codebook if None)

    Returns:
        Dequantized weights [num_active, dim1, dim2]
    """
    num_active = active_experts.shape[0]
    expert_numel = expert_shape[0] * expert_shape[1]
    packed_per_expert = expert_numel // 2
    blocks_per_expert = expert_numel // blocksize

    # Prepare codebook on device
    if codebook is None:
        codebook = torch.tensor(
            NF4_CODEBOOK, dtype=torch.float32, device=packed_data.device
        )
    else:
        codebook = codebook.to(device=packed_data.device, dtype=torch.float32)

    # Flatten inputs
    packed_flat = packed_data.reshape(-1)
    absmax_flat = absmax.reshape(-1).float()  # absmax is usually fp32

    # Output buffer
    out = torch.empty(num_active, expert_numel, dtype=dtype, device=packed_data.device)

    BLOCK_SIZE = 1024  # Process 1024 elements per thread block

    grid = (num_active, triton.cdiv(expert_numel, BLOCK_SIZE))

    _selective_dequant_nf4_kernel[grid](
        packed_flat,
        absmax_flat,
        active_experts,
        codebook,
        out,
        out.stride(0),
        num_active=num_active,
        packed_per_expert=packed_per_expert,
        blocks_per_expert=blocks_per_expert,
        blocksize=blocksize,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.reshape(num_active, *expert_shape)
