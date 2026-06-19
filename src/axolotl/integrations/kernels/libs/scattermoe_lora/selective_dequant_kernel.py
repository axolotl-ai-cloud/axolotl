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


# --- NVFP4 (E2M1 + E4M3 block-16) fast dequant -----------------------------------------
# torchao's eager ``NVFP4Tensor.dequantize()`` materializes large unfused fp32 intermediates
# (~145 GB / ~164 ms at E=256) and torch.compile miscompiles its subclass dispatch, so the
# NVFP4+LoRA fast path needs a dedicated kernel. One memory-bound pass: unpack the E2M1
# nibble, gather the codebook, multiply the linear E4M3 block-16 scale. Validated bit-exact
# vs torchao at E up to 256. (per_tensor_scale is folded outside the kernel.)
#
# OCP E2M1 codebook (low nibble first); ``scale`` is the un-swizzled linear E4M3 block scale.
_NVFP4_E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


@triton.jit
def _nvfp4_dequant_kernel(
    QDATA,  # [R, C/2] uint8 packed E2M1 (2 nibbles/byte, low=col 2j, high=col 2j+1)
    SCALE,  # [R, C/16] E4M3 linear block scale
    LUT,    # [16] fp32 E2M1 codebook
    OUT,    # [R*C] output, dtype = OUT.dtype.element_ty
    PTS,    # fp32 per_tensor_scale: [1] (scalar) or [E] (per-expert); dummy ptr if unused
    total,
    C: tl.constexpr,    # last (contracted) dim; power-of-2 so //C,%C,>>1,>>4 are shifts
    C2: tl.constexpr,   # C // 2
    C16: tl.constexpr,  # C // 16
    ROWS_PER_E: tl.constexpr,  # rows per expert (R // E); for per-expert pts indexing
    PTS_MODE: tl.constexpr,    # 0 none, 1 scalar, 2 per-expert
    BLOCK: tl.constexpr,
):
    # flat 1D over output elements; int64 offsets (r*C2 reaches ~4e9, int32 overflows).
    g = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = g < total
    r = g // C
    c = (g % C).to(tl.int32)
    q = tl.load(QDATA + r * C2 + (c >> 1), mask=m, other=0)
    nib = tl.where((c & 1) == 1, (q >> 4) & 0xF, q & 0xF).to(tl.int32)
    val = tl.load(LUT + nib)
    sc = tl.load(SCALE + r * C16 + (c >> 4), mask=m, other=0.0).to(tl.float32)
    # Fold per_tensor_scale onto the block scale in fp32, then a single bf16 round at the
    # store. Folding onto `sc` (not the product) reproduces the validated path's exact
    # `block_scale * per_tensor` order (mx_weights.py) -> bit-identical, single-rounding.
    # (A post-store bf16 multiply double-rounds, ~2x the error.)
    if PTS_MODE == 1:
        sc = sc * tl.load(PTS).to(tl.float32)
    elif PTS_MODE == 2:
        e = r // ROWS_PER_E
        sc = sc * tl.load(PTS + e, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + g, (val * sc).to(OUT.dtype.element_ty), mask=m)


def dequant_nvfp4_full_triton(nv_param, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequantize a full NVFP4Tensor expert weight ``[E, N, K]`` to ``dtype`` (one pass).

    Drop-in fast replacement for ``nv_param.dequantize(dtype)`` for the un-swizzled linear
    E4M3-block-16 layout the axolotl loader produces (see ``selective_nvfp4_weights_fwd``),
    folding the optional ``per_tensor_scale`` in-kernel in fp32 (single rounding, torchao
    parity). Accepts a scalar pts (the loader's shape) or a per-expert ``[E]`` / ``[E,1,1]``
    (the FSDP carry shape).
    """
    qd, sc = nv_param.qdata, nv_param.scale
    lead = qd.shape[:-1]
    C2 = qd.shape[-1]
    C = C2 * 2
    assert C & (C - 1) == 0, f"NVFP4 fast dequant needs power-of-2 last dim, got {C}"
    rows = 1
    for d in lead:
        rows *= d
    total = rows * C
    lut = torch.tensor(_NVFP4_E2M1_LUT, device=qd.device, dtype=torch.float32)

    # Resolve the per_tensor_scale into a flat fp32 buffer + a mode the kernel folds in fp32.
    pts = getattr(nv_param, "per_tensor_scale", None)
    rows_per_e = 1
    if pts is None:
        pts_buf = lut  # unused dummy pointer (PTS_MODE=0 never loads it)
        pts_mode = 0
    elif pts.numel() == 1:
        pts_buf = pts.reshape(1).to(torch.float32).contiguous()
        pts_mode = 1
    else:
        # per-expert: leading dim of pts is the expert axis ([E] or [E,1,1]); index by r // (R/E)
        n_experts = pts.shape[0]
        assert rows % n_experts == 0, (
            f"per-expert per_tensor_scale [{n_experts}] does not divide rows {rows}"
        )
        rows_per_e = rows // n_experts
        pts_buf = pts.reshape(n_experts).to(torch.float32).contiguous()
        pts_mode = 2

    out = torch.empty(total, device=qd.device, dtype=dtype)
    BLOCK = 8192
    grid = (triton.cdiv(total, BLOCK),)
    _nvfp4_dequant_kernel[grid](
        qd.reshape(-1).contiguous(),
        sc.reshape(-1).contiguous(),
        lut,
        out,
        pts_buf,
        total,
        C=C,
        C2=C2,
        C16=C // 16,
        ROWS_PER_E=rows_per_e,
        PTS_MODE=pts_mode,
        BLOCK=BLOCK,
    )
    return out.reshape(*lead, C)
