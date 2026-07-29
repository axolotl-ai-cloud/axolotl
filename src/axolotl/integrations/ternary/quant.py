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

The binary codebook (`codebook: binary`) drops the zero state instead of adding one:
codes are `sign(W)` and the scale is `mean(|W|)`, which is the least-squares optimum
rather than a threshold heuristic. The `SCALE_EPS` floor, the `f16`-rounded dequant
scale and the λ-interpolation carry over unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

SCALE_EPS: float = 1e-5
ACT_QMAX: int = 127
CODES_PER_BYTE: int = 4

# a five-value code is snapshot as two ternary planes, sign and level
DUAL_PLANES: int = 2

# non-zero magnitudes a free-sum row can hold: s2, s1 - s2, s1, s1 + s2
TRIT_PLANE_MAGNITUDES: int = 4

# the non-negative half of the free-sum codebook as `(t1, t2)` pairs, ordered by the
# magnitude each selects in `trit_plane_values`
TRIT_PLANE_STATES: torch.Tensor = torch.tensor(
    [[0, 0], [0, 1], [1, -1], [1, 0], [1, 1]], dtype=torch.int8
)

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
    return _magnitudes_exceed(values, 2)


def _magnitudes_exceed(values: torch.Tensor, limit: int) -> bool:
    sample = values.reshape(-1)[:BAKED_PROBE_ELEMENTS]
    return bool(torch.unique(sample.abs()).numel() > limit)


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


def _ordered_trit_scales(
    scale_hi: torch.Tensor, scale_lo: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the free-sum pair as `(s1, s2)` with `s1 >= s2 >= 0`, floored."""
    first = scale_hi.to(torch.float32).clamp_min(SCALE_EPS)
    second = scale_lo.to(torch.float32).clamp_min(SCALE_EPS)
    return torch.maximum(first, second), torch.minimum(first, second)


def trit_plane_values(
    scale_1: torch.Tensor, scale_2: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Return the non-negative half of the free-sum codebook, in state order.

    The grid is symmetric, so a magnitude only has to choose among
    `(0, s2, s1 - s2, s1, s1 + s2)` and the sign flips both trits at once.

    Args:
        scale_1: fp32 `(rows, 1)` larger scale.
        scale_2: fp32 `(rows, 1)` smaller scale.

    Returns:
        Five broadcastable magnitudes, aligned with `TRIT_PLANE_STATES`.
    """
    zero = torch.zeros_like(scale_1)
    return (zero, scale_2, scale_1 - scale_2, scale_1, scale_1 + scale_2)


def trit_plane_codes(
    weight: torch.Tensor, scale_1: torch.Tensor, scale_2: torch.Tensor
) -> torch.Tensor:
    """Assign every weight the nearest of the nine free-sum values `t1·s1 + t2·s2`.

    Ties go to the smaller magnitude, the same discipline `round_half_even` gives the
    single-plane grid at `0.5·s` and `dual_codes` gives the five-value one.

    Args:
        weight: Latent weight of shape `(rows, cols)`.
        scale_1: fp32 larger scale, broadcastable to `weight` (per row: `(rows, 1)`).
        scale_2: fp32 smaller scale, same shape as `scale_1`.

    Returns:
        int8 planes of shape `(DUAL_PLANES, rows, cols)`, every entry in `{-1, 0, 1}`.
    """
    first, second = _ordered_trit_scales(scale_1, scale_2)
    values = weight.to(torch.float32)
    magnitude = values.abs()

    candidates = trit_plane_values(first, second)
    best_state = torch.zeros_like(magnitude, dtype=torch.int8)
    best_distance = (magnitude - candidates[0]).abs()
    best_magnitude = candidates[0].expand_as(magnitude).clone()
    for state, candidate in enumerate(candidates[1:], start=1):
        distance = (magnitude - candidate).abs()
        closer = (distance < best_distance) | (
            (distance == best_distance) & (candidate < best_magnitude)
        )
        best_state = torch.where(closer, state, best_state.to(torch.int64)).to(
            torch.int8
        )
        best_magnitude = torch.where(
            closer, candidate.expand_as(magnitude), best_magnitude
        )
        best_distance = torch.minimum(best_distance, distance)

    sign = torch.sign(values).to(torch.int8)
    trits = TRIT_PLANE_STATES.to(weight.device)[best_state.to(torch.int64)]
    return (trits.permute(2, 0, 1) * sign).contiguous()


def dequantize_trit_plane_codes(
    planes: torch.Tensor,
    scale_1_f16: torch.Tensor,
    scale_2_f16: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return `t1·s1 + t2·s2` for the assigned trits — the baked values."""
    first = planes[0].to(torch.float32) * scale_1_f16.to(torch.float32)
    second = planes[1].to(torch.float32) * scale_2_f16.to(torch.float32)
    return (first + second).to(dtype)


def trit_plane_absmean_scales(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fp32 per-row `(s1, s2)` the free-sum grid starts from.

    The five-value seed already partitions each row into a coarse and a fine level;
    the free sum reads them as the two planes directly, which makes the nine-value
    grid a refinement of the five-value one rather than a fresh guess.
    """
    low, high = dual_absmean_scales(weight)
    return high, torch.minimum(low, high)


def trit_plane_grid_scales(
    scale_1: torch.Tensor, scale_2: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ordered pair rounded through `dtype` — the grid the master stores.

    The single-plane master rounds its scale through f16 because every value is one
    scale times a trit, so storage rounding cannot move it off the grid. A free-sum
    value is a *sum* of two scales, and `store(s1 + s2)` is not `store(s1) + store(s2)`:
    unless both scales are already exact in the master's dtype, reloading the master
    and re-deriving the pair from its magnitudes lands an ulp away and the fixed point
    is lost. Rounding here, before the assignment, keeps the codebook and the stored
    values the same set. The mode is master-only, so no packer needs the f16 scale.
    """
    first, second = _ordered_trit_scales(scale_1, scale_2)
    return first.to(dtype).to(torch.float32), second.to(dtype).to(torch.float32)


def baked_trit_plane_codes_and_scales(
    weight: torch.Tensor,
    scales: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Recover `(planes, s1, s2)` from a baked free-sum weight, or `None` if latent.

    A baked row holds values drawn from `{0, ±s2, ±(s1-s2), ±s1, ±(s1+s2)}`, so at
    most four distinct non-zero magnitudes survive and the pair is over-determined by
    any two of them. Which two, though, depends on the states the row actually used —
    a row that never takes the same-sign `(±1, ±1)` state has `s1` as its largest
    magnitude, not `s1 + s2`. Rather than infer the roles, this enumerates the closed
    forms for every plausible pairing and keeps, per row, the first whose grid
    reproduces the row *exactly*; degenerate rows (`s2 = 0`, `s1 = s2`, unused states)
    fall out of the same list.

    Args:
        weight: Candidate baked weight of shape `(rows, cols)`.
        scales: The `(s1, s2)` the master was baked with, when it persisted them.
            Value-only recovery is under-determined — a row that uses only the
            combination states stores `s1 + s2` and `s1 - s2` but never `s1` — so a
            persisted pair is checked directly instead of inferred.

    Returns:
        `(planes, s1, s2)` — int8 `(DUAL_PLANES, rows, cols)` planes and two fp32
        `(rows, 1)` scales — or `None` when some row is not on any free-sum grid.
    """
    stored = weight.detach()
    values = stored.to(torch.float32)
    if values.ndim != 2:
        return None
    if scales is not None:
        # the master persisted its grid, so there is nothing to infer
        first, second = trit_plane_grid_scales(*scales, stored.dtype)
        planes = trit_plane_codes(values, first, second)
        recovered = dequantize_trit_plane_codes(planes, first, second, stored.dtype)
        if not bool(torch.equal(recovered.to(torch.float32), values)):
            return None
        return planes, first, second
    if _too_many_magnitudes(values):
        return None

    top = _top_magnitudes(values)
    resolved_1 = torch.zeros_like(top[0])
    resolved_2 = torch.zeros_like(top[0])
    settled = torch.zeros_like(top[0], dtype=torch.bool)
    for scale_1, scale_2 in _trit_plane_hypotheses(top):
        if bool(settled.all()):
            break
        first, second = trit_plane_grid_scales(
            scale_1.clamp_min(SCALE_EPS), scale_2.clamp_min(SCALE_EPS), stored.dtype
        )
        planes = trit_plane_codes(values, first, second)
        # in the master's own dtype: a recovered scale is a sum or difference of
        # stored magnitudes, so `s1 + s2` only lands back on the stored value once
        # it is rounded the way the bake rounded it
        recovered = dequantize_trit_plane_codes(planes, first, second, stored.dtype).to(
            torch.float32
        )
        exact = (recovered == values).all(dim=-1, keepdim=True)
        take = exact & ~settled
        resolved_1 = torch.where(take, first, resolved_1)
        resolved_2 = torch.where(take, second, resolved_2)
        settled = settled | take
    if not bool(settled.all()):
        return None

    return trit_plane_codes(values, resolved_1, resolved_2), resolved_1, resolved_2


def _too_many_magnitudes(values: torch.Tensor) -> bool:
    """Cheap negative probe: a baked free-sum row holds at most four magnitudes."""
    row = values[0].abs()
    return bool(torch.unique(row).numel() > TRIT_PLANE_MAGNITUDES + 1)


def _top_magnitudes(values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return the `TRIT_PLANE_MAGNITUDES` largest distinct magnitudes per row, zero-padded."""
    magnitude = values.abs()
    top = [magnitude.amax(dim=-1, keepdim=True)]
    for _ in range(TRIT_PLANE_MAGNITUDES - 1):
        # the next magnitude strictly below the one already taken
        lower = torch.where(magnitude < top[-1], magnitude, torch.zeros_like(magnitude))
        top.append(lower.amax(dim=-1, keepdim=True))
    return tuple(top)


def _trit_plane_hypotheses(
    top: tuple[torch.Tensor, ...],
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Candidate `(s1, s2)` readings of a baked row's magnitudes, exact ones first.

    A row that uses the `(±1, 0)` and `(0, ±1)` states stores `s1` and `s2` verbatim,
    so every pairing of two observed magnitudes is tried before any pairing derived by
    arithmetic — those reconstruct a scale the master never stored and can land an ulp
    off. The arithmetic readings then cover the rows that leave a single-plane state
    unused, and the degenerate `s2 = 0` / `s1 = s2` collapses close the list.
    """
    first, second, third, fourth = top
    pairs = [
        (top[i], top[j])
        for i in range(TRIT_PLANE_MAGNITUDES)
        for j in range(i + 1, TRIT_PLANE_MAGNITUDES)
    ]
    zero = torch.zeros_like(first)
    return tuple(pairs) + (
        (second, first - second),  # m1 = s1 + s2, m2 = s1
        (first - second, second),  # m1 = s1 + s2, m2 = s2
        (first - third, third),  # m1 = s1 + s2, m3 = s2
        (0.5 * (first + second), 0.5 * (first - second)),  # m2 = s1 - s2
        (first, first - second),  # m1 = s1, m2 = s1 - s2
        (first, zero),  # a single plane carries the row
        (0.5 * first, 0.5 * first),  # s1 == s2, the sum is the only larger state
    )


def fake_quant_weight_trit_planes(
    weight: torch.Tensor,
    lambda_: float = 1.0,
    scale_1: torch.Tensor | None = None,
    scale_2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return `w + λ · (free-sum(w) - w)` in the weight dtype (no autograd wiring).

    Args:
        weight: Latent weight.
        lambda_: Quantization strength in `[0, 1]`.
        scale_1: fp32 larger scale per row; derived by `trit_plane_absmean_scales`
            when `None`.
        scale_2: fp32 smaller scale per row, ordered defensively against `scale_1`.

    Returns:
        Effective weight, same shape and dtype as `weight`.
    """
    if lambda_ <= 0.0:
        return weight
    if scale_1 is None or scale_2 is None:
        scale_1, scale_2 = trit_plane_absmean_scales(weight)
    first, second = trit_plane_grid_scales(scale_1, scale_2, weight.dtype)
    planes = trit_plane_codes(weight, first, second)
    quantized = dequantize_trit_plane_codes(planes, first, second, weight.dtype)
    return _interp(weight, quantized, lambda_)


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


# ----------------------------------------------------- binary codebook (`{-s, +s}`)


def binary_scale(weight: torch.Tensor, group_size: int | None = None) -> torch.Tensor:
    """Return the fp32 binary scale `mean(|W|)`, clamped to `SCALE_EPS`.

    Numerically the same statistic as `absmean_scale`, and a different fact. For the
    ternary grid the absmean is a *threshold heuristic* — the least-squares optimum
    depends on which weights round to zero. For a sign codebook there is no zero
    state, so `argmin_s ||W - s·sign(W)||²` has the closed form `s = mean(|W|)`
    exactly: `<W, sign(W)> = Σ|w|` and `<sign(W), sign(W)> = n`. Binary therefore
    needs no alternation to place its scale, which is why the fit converges in one
    step where the ternary solver iterates.

    Args:
        weight: Latent weight, any floating dtype.
        group_size: When set, scale per contiguous group of `group_size` input
            elements instead of per tensor.

    Returns:
        fp32 scale: a 0-dim tensor per tensor, or `(out_features, in_features //
        group_size)` in group mode.
    """
    return absmean_scale(weight, group_size)


def binary_codes(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Return the sign codes `{-1, +1}` of `weight` as int8.

    The scale is taken for signature parity with `ternary_codes` and does not enter
    the assignment: with no zero state the nearest code is `sign(w)` for every
    positive scale. Zeros go to `+1` — an arbitrary but fixed tie-break, so a baked
    binary tensor re-quantizes to itself.

    Args:
        weight: Latent weight.
        scale: fp32 scale from `binary_scale` (per tensor or per group).

    Returns:
        int8 tensor of codes with the same shape as `weight`.
    """
    del scale
    return torch.where(weight < 0, -1, 1).to(torch.int8)


def fake_quant_weight_binary(
    weight: torch.Tensor,
    lambda_: float = 1.0,
    group_size: int | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return `w + λ · (codes · s16 - w)` for the sign codebook, no autograd wiring.

    Dequantization goes through `dequantize_codes` unchanged: the codes are `±1` and
    the scale is `f16`-rounded exactly as for ternary, so a baked binary master
    re-quantizes to the same codes in every downstream packer.

    Args:
        weight: Latent weight.
        lambda_: Quantization strength in `[0, 1]`.
        group_size: Group size for group-scale mode, `None` for per tensor.
        scale: Explicit fp32 scale (learnable-scale mode); derived by `binary_scale`
            when `None`.

    Returns:
        Effective weight, same shape and dtype as `weight`.
    """
    if lambda_ <= 0.0:
        return weight
    if scale is None:
        scale = binary_scale(weight, group_size)
    else:
        scale = scale.to(torch.float32).clamp_min(SCALE_EPS)
    codes = binary_codes(weight, scale)
    quantized = dequantize_codes(codes, f16_round_scale(scale), weight.dtype)
    return _interp(weight, quantized, lambda_)


def baked_binary_codes_and_scale(
    weight: torch.Tensor, group_size: int | None = None
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Recover `(codes, s16)` from an already-baked binary weight, or `None` if latent.

    A baked binary tensor holds exactly `{-s16, +s16}` and nothing else, so its `amax`
    *is* `s16` and the codes are the signs. Two consequences the ternary probe does
    not have:

    - a *degenerate* tensor whose values all share one sign is still baked (its single
      magnitude is `s16`, every code the same); the probe must accept it rather than
      read the missing counter-sign as evidence of latent weights,
    - which makes a constant latent tensor indistinguishable from a baked one. That is
      accepted: the two agree on every value the quantizer would produce, so reading
      the constant as baked changes nothing downstream.

    An exact zero is not on the grid, so any tensor containing one is latent.

    Args:
        weight: Candidate baked weight.
        group_size: Group size for group-scale mode, `None` for per tensor.

    Returns:
        `(codes, scale)` — int8 codes in `{-1, +1}` shaped like `weight` and the fp32
        `s16` scale — or `None` when the values are not exactly `{-s, +s}`.
    """
    values = weight.detach().to(torch.float32)
    if group_size is None and _magnitudes_exceed(values, 1):
        return None
    view = values if group_size is None else _group_view(values, group_size)
    magnitude = view.abs()
    scale = (
        magnitude.amax() if group_size is None else magnitude.amax(dim=-1)
    ).clamp_min(SCALE_EPS)
    divisor = scale if group_size is None else scale.unsqueeze(-1)
    codes = binary_codes(view, scale)
    if not bool(torch.equal(codes.to(torch.float32) * divisor, view)):
        return None
    return codes.reshape(weight.shape), scale


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
        return grad_output, None, None, _single_plane_scale_grad(ctx, grad_output)


def _single_plane_scale_grad(ctx, grad_output: torch.Tensor) -> torch.Tensor | None:
    """Return the learnable-scale gradient of a one-scale grid, or `None` in statistic mode.

    The effective weight is `code · s`, so `∂/∂s` is the code, summed over whatever
    unit shares the scale.
    """
    if not ctx.saved_tensors:
        return None
    codes, scale = ctx.saved_tensors
    grad = grad_output.to(torch.float32) * codes.to(torch.float32) * ctx.lambda_
    if scale.numel() == 1:
        reduced = grad.sum()
    else:
        reduced = _group_view(grad, grad.shape[-1] // scale.shape[-1]).sum(-1)
    return reduced.reshape(scale.shape).to(scale.dtype)


def _reduce_to(grad: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum `grad` down to the shape and dtype `target` broadcast from."""
    while grad.ndim > target.ndim:
        grad = grad.sum(0)
    for dim, size in enumerate(target.shape):
        if size == 1:
            grad = grad.sum(dim, keepdim=True)
    return grad.reshape(target.shape).to(target.dtype)


class FakeQuantWeightBinarySTE(torch.autograd.Function):
    """Straight-through estimator for the sign-codebook weight path.

    Forward applies `fake_quant_weight_binary`; backward is the identity into the
    latent weight (and, in learnable-scale mode, the true gradient of the scale —
    `∂(s·code)/∂s` is the code, exactly as on the ternary path).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        weight: torch.Tensor,
        lambda_: float,
        group_size: int | None,
        scale: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the effective weight; see `fake_quant_weight_binary`."""
        ctx.lambda_ = lambda_
        if scale is not None and scale.requires_grad and lambda_ > 0.0:
            ctx.save_for_backward(binary_codes(weight, scale), scale)
        return fake_quant_weight_binary(weight, lambda_, group_size, scale)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        """Pass the incoming gradient straight through to the latent weight."""
        return grad_output, None, None, _single_plane_scale_grad(ctx, grad_output)


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


class FakeQuantWeightTritPlanesSTE(torch.autograd.Function):
    """Straight-through estimator for the nine-value free-sum weight path.

    Forward applies `fake_quant_weight_trit_planes`; backward is the identity into the
    latent weight plus, for each scale, the true gradient of the plane it multiplies —
    the effective weight is linear in both scales, so each plane's trit *is* the local
    derivative.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        weight: torch.Tensor,
        lambda_: float,
        scale_1: torch.Tensor,
        scale_2: torch.Tensor,
    ) -> torch.Tensor:
        """Return the effective weight; see `fake_quant_weight_trit_planes`."""
        ctx.lambda_ = lambda_
        if lambda_ > 0.0 and (scale_1.requires_grad or scale_2.requires_grad):
            first, second = trit_plane_grid_scales(scale_1, scale_2, weight.dtype)
            ctx.save_for_backward(
                trit_plane_codes(weight, first, second),
                scale_1,
                scale_2,
                # the forward reads the pair as (max, min); where that swapped the two,
                # each argument drove the other's plane and takes its gradient
                scale_1.to(torch.float32) < scale_2.to(torch.float32),
            )
        return fake_quant_weight_trit_planes(weight, lambda_, scale_1, scale_2)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        """Pass the incoming gradient through, splitting the scale grad by plane."""
        if not ctx.saved_tensors:
            return grad_output, None, None, None
        planes, scale_1, scale_2, swapped = ctx.saved_tensors
        grad = grad_output.to(torch.float32) * ctx.lambda_
        first = grad * planes[0].to(torch.float32)
        second = grad * planes[1].to(torch.float32)
        grad_1 = _reduce_to(torch.where(swapped, second, first), scale_1)
        grad_2 = _reduce_to(torch.where(swapped, first, second), scale_2)
        return grad_output, None, grad_1, grad_2


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


def fake_quant_weight_binary_ste(
    weight: torch.Tensor,
    lambda_: float = 1.0,
    group_size: int | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Autograd-wired `fake_quant_weight_binary` (STE backward)."""
    return FakeQuantWeightBinarySTE.apply(weight, lambda_, group_size, scale)


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


def fake_quant_weight_trit_planes_ste(
    weight: torch.Tensor,
    lambda_: float = 1.0,
    scale_1: torch.Tensor | None = None,
    scale_2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Autograd-wired `fake_quant_weight_trit_planes` (STE backward)."""
    if scale_1 is None or scale_2 is None:
        scale_1, scale_2 = trit_plane_absmean_scales(weight.detach())
    return FakeQuantWeightTritPlanesSTE.apply(weight, lambda_, scale_1, scale_2)


def act_quant_ste(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Autograd-wired `act_quant` (STE backward)."""
    return ActQuantSTE.apply(x, lambda_)


@contextmanager
def pinned_fp32_precision():
    """Pin fp32 matmuls to full precision for the duration, restoring caller state.

    Gates and calibration solves compare or invert fp32 matmul results; ambient
    TF32 state (e.g. a training config's `tf32: true` applied by normalize_config)
    silently truncates their mantissas. Usable as a decorator.
    """
    state = {
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    if hasattr(torch.backends, "fp32_precision"):
        state["fp32_precision"] = torch.backends.fp32_precision
        state["fp32_precision_matmul"] = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if "fp32_precision" in state:
            torch.backends.fp32_precision = "ieee"
            torch.backends.cuda.matmul.fp32_precision = "ieee"
        yield
    finally:
        torch.set_float32_matmul_precision(state["matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = state["allow_tf32_matmul"]
        torch.backends.cudnn.allow_tf32 = state["allow_tf32_cudnn"]
        if "fp32_precision" in state:
            torch.backends.fp32_precision = state["fp32_precision"]
            torch.backends.cuda.matmul.fp32_precision = state["fp32_precision_matmul"]
