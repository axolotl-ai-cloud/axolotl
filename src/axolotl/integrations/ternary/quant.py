"""Eager reference quantizer — the semantic oracle every kernel must match.

Pinned numerics (see design.md):

- weight scale `s = mean(|W|)` accumulated in fp32 and clamped to `SCALE_EPS`,
- codes `= round_half_even(W / s)` clipped to `{-1, 0, +1}`,
- dequantization uses `s16 = f16_round(s)` so a baked master re-quantizes to the
  exact same codes in every downstream packer,
- activations use a per-token scale `s_x = max(|x|) / 127` (clamped to `SCALE_EPS`)
  with codes clipped to `[-ACT_QMAX, ACT_QMAX]`,
- λ-interpolation on both paths: `y = v + λ · (quant(v) - v)`.

A scale of shape `(out_features, 1)` is a per-row scale — one group spanning every
input feature — so the group machinery below covers it unchanged.

The five-value dual-scale grid (`weight_scale: dual`) is the one departure from the
single-plane rule: each row carries an `(s_lo, s_hi)` pair and every weight takes the
nearest of `{0, ±s_lo, ±s_hi}`, coded as `{-2, -1, 0, +1, +2}`. Everything else — the
fp32 accumulation, the `SCALE_EPS` floor, the `f16`-rounded dequant scale and the
λ-interpolation — is identical, so a baked five-value master re-quantizes to itself.
"""

from __future__ import annotations

import torch

SCALE_EPS: float = 1e-5
ACT_QMAX: int = 127
CODES_PER_BYTE: int = 4

# a five-value code is snapshot as two ternary planes, sign and level
DUAL_PLANES: int = 2

# elements the baked-weight probe inspects before committing to a full-tensor check
BAKED_PROBE_ELEMENTS: int = 1024

_LANE_SHIFTS: tuple[int, ...] = (0, 2, 4, 6)


def _group_view(tensor: torch.Tensor, group_size: int) -> torch.Tensor:
    if tensor.shape[-1] % group_size:
        raise ValueError(
            f"group_size {group_size} does not divide last dim {tensor.shape[-1]}"
        )
    return tensor.reshape(
        *tensor.shape[:-1], tensor.shape[-1] // group_size, group_size
    )


def _scaled(tensor: torch.Tensor, scale: torch.Tensor, divide: bool) -> torch.Tensor:
    """Apply a per-tensor (0-dim) or per-group (`(..., n_groups)`) fp32 scale."""
    if scale.ndim == 0:
        return tensor / scale if divide else tensor * scale
    grouped = _group_view(tensor, tensor.shape[-1] // scale.shape[-1])
    scale = scale.unsqueeze(-1)
    out = grouped / scale if divide else grouped * scale
    return out.reshape(tensor.shape)


def _interp(
    value: torch.Tensor, quantized: torch.Tensor, lambda_: float
) -> torch.Tensor:
    if lambda_ >= 1.0:
        return quantized
    if lambda_ <= 0.0:
        return value
    out = value.to(torch.float32) + lambda_ * (
        quantized.to(torch.float32) - value.to(torch.float32)
    )
    return out.to(value.dtype)


def absmean_scale(weight: torch.Tensor, group_size: int | None = None) -> torch.Tensor:
    """Return the fp32 absmean weight scale, clamped to `SCALE_EPS`.

    Args:
        weight: Latent weight, any floating dtype, shape `(out_features, in_features)`.
        group_size: When set, scale per contiguous group of `group_size` input
            elements instead of per tensor.

    Returns:
        fp32 scale: a 0-dim tensor per tensor, or `(out_features, in_features //
        group_size)` in group mode.
    """
    absw = weight.to(torch.float32).abs()
    scale = (
        absw.mean() if group_size is None else _group_view(absw, group_size).mean(-1)
    )
    return scale.clamp_min(SCALE_EPS)


def f16_round_scale(scale: torch.Tensor) -> torch.Tensor:
    """Round an fp32 scale through float16 and back — the pinned dequant scale `s16`."""
    return scale.to(torch.float16).to(torch.float32)


def ternary_codes(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Return `round_half_even(weight / scale)` clipped to `{-1, 0, +1}` as int8.

    Args:
        weight: Latent weight.
        scale: fp32 scale from `absmean_scale` (per tensor or per group).

    Returns:
        int8 tensor of codes with the same shape as `weight`.
    """
    scale = scale.to(torch.float32).clamp_min(SCALE_EPS)
    codes = torch.round(_scaled(weight.to(torch.float32), scale, divide=True))
    return codes.clamp_(-1.0, 1.0).to(torch.int8)


def dequantize_codes(
    codes: torch.Tensor, scale_f16: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Return `codes * scale_f16` cast to `dtype` — the exact baked-master values."""
    values = _scaled(codes.to(torch.float32), scale_f16.to(torch.float32), divide=False)
    return values.to(dtype)


def _has_more_than_two_magnitudes(values: torch.Tensor) -> bool:
    """Cheap negative probe: a per-tensor baked weight holds only `{0, s}` magnitudes."""
    sample = values.reshape(-1)[:BAKED_PROBE_ELEMENTS]
    return bool(torch.unique(sample.abs()).numel() > 2)


def baked_codes_and_scale(
    weight: torch.Tensor, group_size: int | None = None
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Recover `(codes, s16)` from an already-baked weight, or `None` if it is latent.

    A baked tensor holds exactly `{-s16, 0, +s16}`, so its `amax` *is* `s16` and the
    codes follow without re-running the quantizer — running `absmean_scale` on such a
    tensor would instead shrink every magnitude by the non-zero code fraction.

    Args:
        weight: Candidate baked weight.
        group_size: Group size for group-scale mode, `None` for per tensor.

    Returns:
        `(codes, scale)` — int8 codes shaped like `weight` and the fp32 `s16` scale
        (0-dim per tensor, or one per group) — or `None` when the values are not
        exactly `{-s, 0, +s}`.
    """
    values = weight.detach().to(torch.float32)
    if group_size is None and _has_more_than_two_magnitudes(values):
        return None
    view = values if group_size is None else _group_view(values, group_size)
    magnitude = view.abs()
    scale = (
        magnitude.amax() if group_size is None else magnitude.amax(dim=-1)
    ).clamp_min(SCALE_EPS)
    divisor = scale if group_size is None else scale.unsqueeze(-1)
    codes = torch.round(view / divisor)
    if not bool(torch.equal(codes * divisor, view)):
        return None
    return codes.to(torch.int8).reshape(weight.shape), scale


def fake_quant_weight(
    weight: torch.Tensor,
    lambda_: float = 1.0,
    group_size: int | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return `w + λ · (codes · s16 - w)` in the weight dtype (no autograd wiring).

    Args:
        weight: Latent weight.
        lambda_: Quantization strength in `[0, 1]`.
        group_size: Group size for group-scale mode, `None` for per tensor.
        scale: Explicit fp32 scale (learnable-scale mode); derived by
            `absmean_scale` when `None`.

    Returns:
        Effective weight, same shape and dtype as `weight`.
    """
    if lambda_ <= 0.0:
        return weight
    if scale is None:
        scale = absmean_scale(weight, group_size)
    else:
        # the same floor `ternary_codes` applies, so codes and s16 share one scale;
        # an unclamped subnormal would round to an f16 zero and kill the layer
        scale = scale.to(torch.float32).clamp_min(SCALE_EPS)
    codes = ternary_codes(weight, scale)
    quantized = dequantize_codes(codes, f16_round_scale(scale), weight.dtype)
    return _interp(weight, quantized, lambda_)


def _conditional_mean(
    values: torch.Tensor, mask: torch.Tensor, fallback: torch.Tensor
) -> torch.Tensor:
    """Per-row mean of `values` over `mask`, falling back where a row selects nothing."""
    count = mask.sum(dim=-1, keepdim=True)
    total = (values * mask).sum(dim=-1, keepdim=True)
    return torch.where(count > 0, total / count.clamp_min(1), fallback)


def _ordered_dual_scales(
    scale_lo: torch.Tensor, scale_hi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    low = scale_lo.to(torch.float32).clamp_min(SCALE_EPS)
    high = scale_hi.to(torch.float32).clamp_min(SCALE_EPS)
    return torch.minimum(low, high), torch.maximum(low, high)


def dual_absmean_scales(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fp32 per-row `(s_lo, s_hi)` the five-value grid starts from.

    One conditional-mean refinement of the ternary partition: the weights a per-row
    absmean rounds to `±1` set `s_lo`, the ones it clips down from `±2` set `s_hi`.

    Args:
        weight: Latent weight of shape `(out_features, in_features)`.

    Returns:
        Two fp32 `(out_features, 1)` tensors with `s_lo <= s_hi`, both clamped to
        `SCALE_EPS`.
    """
    magnitude = weight.to(torch.float32).abs()
    centre = magnitude.mean(dim=-1, keepdim=True).clamp_min(SCALE_EPS)
    high_mask = magnitude > 1.5 * centre
    low_mask = (magnitude > 0.5 * centre) & ~high_mask
    scale_lo = _conditional_mean(magnitude, low_mask, centre).clamp_min(SCALE_EPS)
    scale_hi = _conditional_mean(magnitude, high_mask, scale_lo).clamp_min(SCALE_EPS)
    return scale_lo, torch.maximum(scale_hi, scale_lo)


def dual_codes(
    weight: torch.Tensor, scale_lo: torch.Tensor, scale_hi: torch.Tensor
) -> torch.Tensor:
    """Return the five-state codes `{-2, -1, 0, +1, +2}` nearest to `weight`.

    `±1` selects `s_lo` and `±2` selects `s_hi`. A magnitude sitting exactly on a
    boundary takes the smaller state, which is where `round_half_even` puts the
    ternary tie at `0.5·s`.

    Args:
        weight: Latent weight.
        scale_lo: fp32 low scale, broadcastable to `weight` (per row: `(out, 1)`).
        scale_hi: fp32 high scale, same shape as `scale_lo`.

    Returns:
        int8 codes shaped like `weight`.
    """
    low, high = _ordered_dual_scales(scale_lo, scale_hi)
    values = weight.to(torch.float32)
    magnitude = values.abs()
    level = (magnitude > 0.5 * low).to(torch.int8) + (
        magnitude > 0.5 * (low + high)
    ).to(torch.int8)
    return level * torch.sign(values).to(torch.int8)


def dequantize_dual_codes(
    codes: torch.Tensor,
    scale_lo_f16: torch.Tensor,
    scale_hi_f16: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return `sign(codes) · s16` for the level each code selects — the baked values."""
    magnitude = torch.where(
        codes.abs() > 1, scale_hi_f16.to(torch.float32), scale_lo_f16.to(torch.float32)
    )
    return (torch.sign(codes.to(torch.float32)) * magnitude).to(dtype)


def dual_state_planes(codes: torch.Tensor) -> torch.Tensor:
    """Split five-state codes into the two ternary planes the snapshot buffer packs.

    Plane 0 is the sign, plane 1 the level (`-1` for `s_lo`, `+1` for `s_hi`), so both
    lanes of a weight are zero exactly when its state is — which keeps
    `zero_fraction` over the pair the true fraction of zeroed weights.

    Args:
        codes: int8 five-state codes.

    Returns:
        int8 tensor of shape `(DUAL_PLANES, *codes.shape)`, every entry in `{-1, 0, 1}`.
    """
    nonzero = (codes != 0).to(torch.int8)
    level = torch.where(codes.abs() > 1, 1, -1).to(torch.int8) * nonzero
    return torch.stack((torch.sign(codes).to(torch.int8), level))


def baked_dual_codes_and_scales(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Recover `(codes, s_lo, s_hi)` from a baked five-value weight, or `None` if latent.

    A baked row holds exactly `{0, ±s_lo, ±s_hi}`, so its `amax` *is* `s_hi` and its
    smallest non-zero magnitude *is* `s_lo`; the statistic would instead shrink both.
    Rows where the two coincide (one magnitude, or none at all) recover a degenerate
    pair, which quantizes and dequantizes to the same values either way.

    Args:
        weight: Candidate baked weight of shape `(out_features, in_features)`.

    Returns:
        `(codes, s_lo, s_hi)` — int8 codes shaped like `weight` and two fp32
        `(out_features, 1)` scales — or `None` when the rows are not exactly on a
        five-value grid.
    """
    values = weight.detach().to(torch.float32)
    magnitude = values.abs()
    high = magnitude.amax(dim=-1, keepdim=True)
    low = magnitude.masked_fill(magnitude == 0, float("inf")).amin(dim=-1, keepdim=True)
    low = torch.where(torch.isinf(low), high, low).clamp_min(SCALE_EPS)
    high = high.clamp_min(SCALE_EPS)
    codes = dual_codes(values, low, high)
    if not bool(
        torch.equal(dequantize_dual_codes(codes, low, high, values.dtype), values)
    ):
        return None
    return codes, low, high


def fake_quant_weight_dual(
    weight: torch.Tensor,
    lambda_: float = 1.0,
    scale_lo: torch.Tensor | None = None,
    scale_hi: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return `w + λ · (five-value(w) - w)` in the weight dtype (no autograd wiring).

    Args:
        weight: Latent weight.
        lambda_: Quantization strength in `[0, 1]`.
        scale_lo: fp32 low scale per row; derived by `dual_absmean_scales` when `None`.
        scale_hi: fp32 high scale per row; must be `>= scale_lo` for the gradients to
            be attributed to the right one, and is ordered defensively here.

    Returns:
        Effective weight, same shape and dtype as `weight`.
    """
    if lambda_ <= 0.0:
        return weight
    if scale_lo is None or scale_hi is None:
        scale_lo, scale_hi = dual_absmean_scales(weight)
    low, high = _ordered_dual_scales(scale_lo, scale_hi)
    codes = dual_codes(weight, low, high)
    quantized = dequantize_dual_codes(
        codes, f16_round_scale(low), f16_round_scale(high), weight.dtype
    )
    return _interp(weight, quantized, lambda_)


def act_scale(x: torch.Tensor) -> torch.Tensor:
    """Return the per-token activation scale `max(|x|) / ACT_QMAX`, clamped to `SCALE_EPS`.

    Args:
        x: Activation of shape `(..., in_features)`; the last dim is the reduced one.

    Returns:
        fp32 scale of shape `(..., 1)`.
    """
    amax = x.to(torch.float32).abs().amax(dim=-1, keepdim=True)
    return (amax / ACT_QMAX).clamp_min(SCALE_EPS)


def act_quant_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to int8 codes clipped to `[-ACT_QMAX, ACT_QMAX]`.

    Args:
        x: Activation of shape `(..., in_features)`.

    Returns:
        `(codes, scale)` — int8 codes shaped like `x`, and the fp32 per-token scale
        of shape `(..., 1)`.
    """
    scale = act_scale(x)
    codes = torch.round(x.to(torch.float32) / scale).clamp_(-ACT_QMAX, ACT_QMAX)
    return codes.to(torch.int8), scale


def act_quant(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Return `x + λ · (codes · s_x - x)` in the activation dtype (no autograd wiring)."""
    if lambda_ <= 0.0:
        return x
    codes, scale = act_quant_int8(x)
    return _interp(x, (codes.to(torch.float32) * scale).to(x.dtype), lambda_)


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack ternary codes into 4-per-byte uint8, for the monitoring snapshot buffer.

    Args:
        codes: int8 codes in `{-1, 0, +1}`.

    Returns:
        Flat uint8 tensor of length `ceil(codes.numel() / CODES_PER_BYTE)`; codes are
        stored as `code + 1` in little-endian 2-bit lanes and the tail is zero-padded.
    """
    flat = (codes.reshape(-1).to(torch.int16) + 1).to(torch.uint8)
    pad = -flat.numel() % CODES_PER_BYTE
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    lanes = flat.reshape(-1, CODES_PER_BYTE)
    packed = lanes[:, 0].clone()
    for lane, shift in enumerate(_LANE_SHIFTS[1:], start=1):
        packed |= lanes[:, lane] << shift
    return packed


def unpack_codes(packed: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Inverse of `pack_codes`: unpack a uint8 snapshot back to int8 codes of `shape`."""
    lanes = torch.stack([(packed >> shift) & 3 for shift in _LANE_SHIFTS], dim=-1)
    numel = int(torch.Size(shape).numel())
    return (lanes.reshape(-1)[:numel].to(torch.int8) - 1).reshape(shape)


def flip_count(packed_prev: torch.Tensor, packed_now: torch.Tensor) -> torch.Tensor:
    """Count ternary codes that changed between two packed snapshots.

    Args:
        packed_prev: Packed snapshot from a previous step.
        packed_now: Packed snapshot from the current step, same length.

    Returns:
        0-dim int64 tensor holding the number of differing codes; padding lanes never
        contribute.
    """
    if packed_prev.shape != packed_now.shape:
        raise ValueError(
            f"snapshot length mismatch: {tuple(packed_prev.shape)} vs "
            f"{tuple(packed_now.shape)}"
        )
    diff = packed_prev ^ packed_now
    lanes = torch.stack([(diff >> shift) & 3 for shift in _LANE_SHIFTS], dim=-1)
    return (lanes != 0).sum(dtype=torch.int64)


def zero_fraction(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Return the fraction of codes equal to 0 in a packed snapshot of `numel` codes."""
    lanes = torch.stack([(packed >> shift) & 3 for shift in _LANE_SHIFTS], dim=-1)
    zeros = (lanes.reshape(-1)[:numel] == 1).sum(dtype=torch.float32)
    return zeros / max(numel, 1)


class FakeQuantWeightSTE(torch.autograd.Function):
    """Straight-through estimator for the weight fake-quant path.

    Forward applies `fake_quant_weight`; backward is the identity into the latent
    weight (and, in learnable-scale mode, the true gradient of the scale).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        weight: torch.Tensor,
        lambda_: float,
        group_size: int | None,
        scale: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the effective weight; see `fake_quant_weight`."""
        ctx.lambda_ = lambda_
        if scale is not None and scale.requires_grad and lambda_ > 0.0:
            ctx.save_for_backward(ternary_codes(weight, scale), scale)
        return fake_quant_weight(weight, lambda_, group_size, scale)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        """Pass the incoming gradient straight through to the latent weight."""
        grad_scale = None
        if ctx.saved_tensors:
            codes, scale = ctx.saved_tensors
            grad = grad_output.to(torch.float32) * codes.to(torch.float32) * ctx.lambda_
            if scale.numel() == 1:
                grad_scale = grad.sum()
            else:
                grad_scale = _group_view(grad, grad.shape[-1] // scale.shape[-1]).sum(
                    -1
                )
            grad_scale = grad_scale.reshape(scale.shape).to(scale.dtype)
        return grad_output, None, None, grad_scale


def _reduce_to(grad: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum `grad` down to the shape and dtype `target` broadcast from."""
    while grad.ndim > target.ndim:
        grad = grad.sum(0)
    for dim, size in enumerate(target.shape):
        if size == 1:
            grad = grad.sum(dim, keepdim=True)
    return grad.reshape(target.shape).to(target.dtype)


class FakeQuantWeightDualSTE(torch.autograd.Function):
    """Straight-through estimator for the five-value dual-scale weight path.

    Forward applies `fake_quant_weight_dual`; backward is the identity into the latent
    weight plus the true gradient of whichever scale each weight was assigned to.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        weight: torch.Tensor,
        lambda_: float,
        scale_lo: torch.Tensor,
        scale_hi: torch.Tensor,
    ) -> torch.Tensor:
        """Return the effective weight; see `fake_quant_weight_dual`."""
        ctx.lambda_ = lambda_
        if lambda_ > 0.0 and (scale_lo.requires_grad or scale_hi.requires_grad):
            ctx.save_for_backward(
                dual_codes(weight, scale_lo, scale_hi),
                scale_lo,
                scale_hi,
                # the forward reads the pair as (min, max); where that swapped the two,
                # each argument drove the other's level and takes the other's gradient
                scale_lo.to(torch.float32) > scale_hi.to(torch.float32),
            )
        return fake_quant_weight_dual(weight, lambda_, scale_lo, scale_hi)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        """Pass the incoming gradient through, splitting the scale grad by state."""
        if not ctx.saved_tensors:
            return grad_output, None, None, None
        codes, scale_lo, scale_hi, swapped = ctx.saved_tensors
        grad = (
            grad_output.to(torch.float32)
            * torch.sign(codes.to(torch.float32))
            * ctx.lambda_
        )
        low_level, high_level = grad * (codes.abs() == 1), grad * (codes.abs() > 1)
        grad_lo = _reduce_to(torch.where(swapped, high_level, low_level), scale_lo)
        grad_hi = _reduce_to(torch.where(swapped, low_level, high_level), scale_hi)
        return grad_output, None, grad_lo, grad_hi


class ActQuantSTE(torch.autograd.Function):
    """Straight-through estimator for the per-token activation fake-quant path."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:  # type: ignore[override]
        """Return the effective activation; see `act_quant`."""
        return act_quant(x, lambda_)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        """Pass the incoming gradient straight through to the activation."""
        return grad_output, None


def fake_quant_weight_ste(
    weight: torch.Tensor,
    lambda_: float = 1.0,
    group_size: int | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Autograd-wired `fake_quant_weight` (STE backward)."""
    return FakeQuantWeightSTE.apply(weight, lambda_, group_size, scale)


def fake_quant_weight_dual_ste(
    weight: torch.Tensor,
    lambda_: float = 1.0,
    scale_lo: torch.Tensor | None = None,
    scale_hi: torch.Tensor | None = None,
) -> torch.Tensor:
    """Autograd-wired `fake_quant_weight_dual` (STE backward)."""
    if scale_lo is None or scale_hi is None:
        scale_lo, scale_hi = dual_absmean_scales(weight.detach())
    return FakeQuantWeightDualSTE.apply(weight, lambda_, scale_lo, scale_hi)


def act_quant_ste(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Autograd-wired `act_quant` (STE backward)."""
    return ActQuantSTE.apply(x, lambda_)
