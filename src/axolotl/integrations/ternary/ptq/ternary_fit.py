"""Data-free alternating ternary fit — the solver behind `init: ternary_fit`.

Symmetric (μ ≡ 0) single-plane collapse of the alternating scale/state solve:

1. threshold-init the codes with the TWN rule `t = sign(w)·[|w| > TWN_THRESHOLD·mean|w|]`,
2. solve the scale in closed form, `s = <W, T> / <T, T>`, over whatever unit the
   scale mode ties together (tensor, group, row) — a ridge-guarded 2×2 solve for
   `dual`, whose two planes are correlated,
3. re-assign every weight to the state minimizing `(w - s·t)²`, which for one plane
   is the nearest of `{-s, 0, +s}` and for `dual` an exhaustive search over the five
   states `{(0, 0), (±1, 0), (0, ±1)}`,
4. iterate until no code moves (or the objective stops improving by `tol`), capped at
   `MAX_ITERS`.

The fit only chooses codes and scales; the caller writes `codes · s` back into the
latent so the quantizer reproduces exactly this solution on the first forward. The
scale it stores must still survive `quant.f16_round_scale`, or the export path and
the fit disagree on the very first tensor.

`fit_binary` is the degenerate member of the same family: a sign codebook's assignment
does not depend on its scale, so step 3 is fixed before step 2 runs and the whole
alternation collapses to one closed-form solve.

Both alternations therefore run their assignment step through the *f16-rounded* scale
and score the objective against the same dequantization, which buys two properties
the heal depends on: every returned pair satisfies `codes == ternary_codes(W,
f16(scale))` — a fixed point of the plugin's canonical quantizer, not of an idealized
one — and the per-unit objective never increases from one iteration to the next,
because a unit that fails to improve keeps the pair it had.
"""

from __future__ import annotations

from typing import get_args

import torch

from .. import quant
from ..args import TWO_PLANE_SCALE_MODES, WeightScaleMode

# TWN threshold rule: codes start nonzero where |w| exceeds this fraction of mean|w|
TWN_THRESHOLD: float = 0.75

# alternating solves are monotone and converge in ~10; the cap is the safety net
MAX_ITERS: int = 30

# relative improvement of the reconstruction objective below which the fit stops
CONVERGENCE_TOL: float = 1e-6

# ridge for the dual-plane 2x2 normal equations, *relative to the system's largest
# eigenvalue*, and the condition number above which it is boosted by
# sqrt(kappa / RIDGE_KAPPA_TRIGGER) — a near-singular system means the two planes have
# collapsed onto the same state set. Relative because the same solver takes both the
# data-free normal equations (entries are code counts) and the activation-weighted ones
# (entries scale with tokens x activation^2): an absolute ridge is inert in the second.
RIDGE_LAMBDA: float = 1e-8
RIDGE_KAPPA_TRIGGER: float = 1e6

# how far the boost may take the ridge, still relative to the largest eigenvalue; a
# singular system is regularized, not replaced by the ridge
RIDGE_MAX: float = 1e-2

SCALE_MODES: frozenset[str] = frozenset(get_args(WeightScaleMode))
PER_TENSOR_MODES: frozenset[str] = frozenset({"absmean", "learnable"})

_TINY: float = 1e-30


def fit_ternary(
    weight: torch.Tensor,
    scale_mode: WeightScaleMode = "absmean",
    group_size: int | None = None,
    max_iters: int = MAX_ITERS,
    tol: float = CONVERGENCE_TOL,
    ridge_lambda: float = RIDGE_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit ternary codes and scales to `weight` without any calibration data.

    Args:
        weight: Latent weight of shape `(out_features, in_features)`, any float dtype.
        scale_mode: Which unit shares a scale — `absmean`/`learnable` per tensor,
            `group` per `group_size` block, `learnable_row` per output channel,
            `dual` two per-row scales over five states.
        group_size: Block size for `scale_mode: group`; must divide `in_features`.
        max_iters: Alternation cap.
        tol: Relative objective improvement below which the alternation stops.
        ridge_lambda: Base ridge added to the `dual` normal equations.

    Returns:
        `(codes, scales)`. For a single plane, `codes` is int8 in `{-1, 0, +1}`
        shaped like `weight` and `scales` is fp32 — 0-dim per tensor,
        `(out_features, 1)` per row, `(out_features, in_features // group_size)` per
        group. For `dual`, `codes` is int8 of shape `(2, out_features, in_features)`
        with at most one nonzero plane per weight, and `scales` is fp32
        `(out_features, 2)` holding `(s_lo, s_hi)` per row.

    Raises:
        ValueError: If `group_size` does not divide `in_features`, or the scale mode
            and `group_size` disagree.
    """
    _validate(weight, scale_mode, group_size)
    dense = weight.detach().to(torch.float32)
    if scale_mode in TWO_PLANE_SCALE_MODES:
        return _fit_two_plane(dense, scale_mode, max_iters, tol, ridge_lambda)

    view, scale_shape = _unit_view(dense, scale_mode, group_size)
    fallback = view.abs().mean(-1).clamp_min(quant.SCALE_EPS)
    seed = torch.where(
        view.abs() > TWN_THRESHOLD * fallback.unsqueeze(-1),
        torch.sign(view),
        torch.zeros_like(view),
    )
    scale = _closed_form(view, seed, fallback)
    codes = _assign(dense, scale.reshape(scale_shape), scale_mode, group_size)
    objective = _objective(view, codes, scale)

    for _ in range(max_iters):
        next_scale = _closed_form(view, codes, fallback)
        next_codes = _assign(
            dense, next_scale.reshape(scale_shape), scale_mode, group_size
        )
        next_objective = _objective(view, next_codes, next_scale)
        better = next_objective < objective
        if not bool(better.any()):
            break
        gain = torch.where(
            better,
            (objective - next_objective) / objective.clamp_min(_TINY),
            torch.zeros_like(objective),
        )
        scale = torch.where(better, next_scale, scale)
        codes = torch.where(better.unsqueeze(-1), next_codes, codes)
        objective = torch.where(better, next_objective, objective)
        if float(gain.max()) <= tol:
            break

    return (
        codes.reshape(weight.shape).to(torch.int8),
        scale.reshape(scale_shape).contiguous(),
    )


def fit_binary(
    weight: torch.Tensor,
    scale_mode: WeightScaleMode = "absmean",
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit sign codes and scales to `weight` — one pass, no alternation.

    The alternation `fit_ternary` runs exists because the ternary assignment depends
    on the scale (a weight rounds to zero or not depending where the threshold sits).
    A sign codebook has no threshold: `sign(w)` is the nearest code under every
    positive scale, so the codes are final before the scale is solved and the
    closed-form `s = <W, T> / <T, T> = mean|w|` is the global optimum of
    `‖W - s·sign(W)‖²` over that unit. Iterating could only return the same pair.

    Args:
        weight: Latent weight of shape `(out_features, in_features)`, any float dtype.
        scale_mode: Which unit shares a scale — `absmean`/`learnable` per tensor,
            `group` per `group_size` block, `learnable_row` per output channel.
        group_size: Block size for `scale_mode: group`; must divide `in_features`.

    Returns:
        `(codes, scales)` — int8 codes in `{-1, +1}` shaped like `weight`, and fp32
        scales in the layout `fit_ternary` returns for `scale_mode`.

    Raises:
        ValueError: If `group_size` does not divide `in_features`, the scale mode and
            `group_size` disagree, or the scale mode carries two planes.
    """
    _validate(weight, scale_mode, group_size)
    if scale_mode in TWO_PLANE_SCALE_MODES:
        raise ValueError(
            f"the binary codebook has one plane, so scale_mode={scale_mode!r} has "
            "nothing to put in its second one"
        )
    dense = weight.detach().to(torch.float32)
    view, scale_shape = _unit_view(dense, scale_mode, group_size)
    fallback = view.abs().mean(-1).clamp_min(quant.SCALE_EPS)
    codes = quant.binary_codes(view, fallback).to(torch.float32)
    scale = _closed_form(view, codes, fallback)
    return (
        codes.reshape(weight.shape).to(torch.int8),
        scale.reshape(scale_shape).contiguous(),
    )


def solve_scale(
    weight: torch.Tensor,
    codes: torch.Tensor,
    scale_mode: WeightScaleMode = "absmean",
    group_size: int | None = None,
    ridge_lambda: float = RIDGE_LAMBDA,
) -> torch.Tensor:
    """Return the least-squares scale for fixed `codes`, in `fit_ternary`'s layout.

    The data-free half of the alternation: `s = <W, T> / <T, T>` per unit, falling
    back to that unit's absmean where the codes are all zero.

    Under `dual` the returned pair is matched to the planes *positionally* —
    `scale[:, k]` is the scale of `codes[k]` — which is what `dequantize_fit` reads.
    It comes out ordered whenever the planes are (the assignment step puts the smaller
    magnitudes on plane 0); reordering it here without permuting the planes with it
    would return a pair that reconstructs something else entirely.

    Args:
        weight: Latent weight of shape `(out_features, in_features)`.
        codes: Codes shaped like `weight`, or `(2, ...)` for `dual`.
        scale_mode: Which unit shares a scale.
        group_size: Block size for `scale_mode: group`.
        ridge_lambda: Base ridge for the `dual` normal equations.

    Returns:
        fp32 scales, clamped to `quant.SCALE_EPS`.
    """
    _validate(weight, scale_mode, group_size)
    dense = weight.detach().to(torch.float32)
    if scale_mode in TWO_PLANE_SCALE_MODES:
        return _solve_dual(dense, codes.to(torch.float32), ridge_lambda)
    view, scale_shape = _unit_view(dense, scale_mode, group_size)
    code_view, _ = _unit_view(codes.detach().to(torch.float32), scale_mode, group_size)
    fallback = view.abs().mean(-1).clamp_min(quant.SCALE_EPS)
    return _closed_form(view, code_view, fallback).reshape(scale_shape).contiguous()


def dequantize_fit(
    codes: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    scale_mode: str = "dual",
) -> torch.Tensor:
    """Return the reconstruction `codes · scale` a fit stands for, in `dtype`.

    Args:
        codes: Codes shaped like the weight, or `(2, ...)` planes.
        scale: fp32 scales in the fit's layout.
        dtype: Reconstruction dtype.
        scale_mode: Only distinguishes the two-plane grids: `trit_planes` sums both
            planes, so its scales round through `dtype` rather than f16 — see
            `quant.trit_plane_grid_scales`.
    """
    dense = scale.detach().to(torch.float32)
    if codes.ndim == 3:
        # positional, as documented on `solve_scale`: scale column k belongs to plane k
        rounded = (
            dense.to(dtype).to(torch.float32)
            if scale_mode == "trit_planes"
            else quant.f16_round_scale(dense)
        )
        planes = codes.to(torch.float32)
        values = planes[0] * rounded[:, :1] + planes[1] * rounded[:, 1:]
        return values.to(dtype)
    return quant.dequantize_codes(codes, quant.f16_round_scale(dense), dtype)


def _validate(weight: torch.Tensor, scale_mode: str, group_size: int | None) -> None:
    if scale_mode not in SCALE_MODES:
        raise ValueError(f"unknown ternary weight_scale mode: {scale_mode!r}")
    if weight.ndim != 2:
        raise ValueError(
            f"fit_ternary expects a (out_features, in_features) weight, got shape "
            f"{tuple(weight.shape)}"
        )
    if scale_mode == "group":
        if group_size is None:
            raise ValueError("scale_mode='group' requires a group_size")
        if weight.shape[-1] % group_size:
            raise ValueError(
                f"group_size {group_size} does not divide in_features "
                f"{weight.shape[-1]}"
            )
    elif group_size is not None:
        raise ValueError(
            f"group_size is only valid with scale_mode='group', got {scale_mode!r}"
        )


def _unit_view(
    tensor: torch.Tensor, scale_mode: str, group_size: int | None
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Return `(units × elements)` view of `tensor` and the natural scale shape."""
    out_features, in_features = tensor.shape
    if scale_mode in PER_TENSOR_MODES:
        return tensor.reshape(1, -1), ()
    if scale_mode == "learnable_row":
        return tensor, (out_features, 1)
    assert group_size is not None
    groups = in_features // group_size
    return tensor.reshape(out_features * groups, -1), (out_features, groups)


def _closed_form(
    view: torch.Tensor, codes: torch.Tensor, fallback: torch.Tensor
) -> torch.Tensor:
    support = (codes * codes).sum(-1)
    scale = torch.where(
        support > 0, (view * codes).sum(-1) / support.clamp_min(1.0), fallback
    )
    return scale.clamp_min(quant.SCALE_EPS)


def _assign(
    weight: torch.Tensor,
    scale: torch.Tensor,
    scale_mode: str,
    group_size: int | None,
) -> torch.Tensor:
    codes = quant.ternary_codes(weight, quant.f16_round_scale(scale))
    return _unit_view(codes.to(torch.float32), scale_mode, group_size)[0]


def _objective(
    view: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    residual = view - codes * quant.f16_round_scale(scale).unsqueeze(-1)
    return (residual * residual).sum(-1)


def _fit_two_plane(
    weight: torch.Tensor,
    scale_mode: str,
    max_iters: int,
    tol: float,
    ridge_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alternate the 2x2 ridge solve with the mode's assignment until it stops paying.

    `dual` and `trit_planes` share every step but the assignment: the five-value grid
    admits one non-zero plane per weight, the free sum admits both. The normal
    equations are the same because both read the planes as the two columns of a
    least-squares system.
    """
    assign = _assign_trit_planes if scale_mode == "trit_planes" else _assign_dual
    _, row_scale = fit_ternary(weight, "learnable_row", max_iters=max_iters, tol=tol)
    high = row_scale.reshape(-1)
    # seeding s_lo at half s_hi makes the five-value grid a refinement of the
    # single-plane solution, so the first assignment can only reduce its error
    scales = torch.stack([0.5 * high, high], dim=-1).clamp_min(quant.SCALE_EPS)
    planes = assign(weight, scales)
    objective = _objective_two_plane(weight, planes, scales, scale_mode)

    for _ in range(max_iters):
        next_scales = torch.sort(
            _solve_dual(weight, planes, ridge_lambda), dim=-1
        ).values
        next_planes = assign(weight, next_scales)
        next_objective = _objective_two_plane(
            weight, next_planes, next_scales, scale_mode
        )
        better = next_objective < objective
        if not bool(better.any()):
            break
        gain = torch.where(
            better,
            (objective - next_objective) / objective.clamp_min(_TINY),
            torch.zeros_like(objective),
        )
        scales = torch.where(better.unsqueeze(-1), next_scales, scales)
        planes = torch.where(better.reshape(1, -1, 1), next_planes, planes)
        objective = torch.where(better, next_objective, objective)
        if float(gain.max()) <= tol:
            break

    return planes.to(torch.int8), scales.contiguous()


def _assign_dual(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Assign every weight to the nearest of `{0, ±s_lo, ±s_hi}`; ties take the lower level."""
    scale_f16 = quant.f16_round_scale(scales)
    low, high = scale_f16[:, :1], scale_f16[:, 1:]
    magnitude = weight.abs()
    to_zero = magnitude * magnitude
    to_low = (magnitude - low) ** 2
    to_high = (magnitude - high) ** 2
    take_high = (to_high < to_low) & (to_high < to_zero)
    take_low = (to_low < to_zero) & ~take_high
    sign = torch.sign(weight)
    zeros = torch.zeros_like(weight)
    return torch.stack(
        [torch.where(take_low, sign, zeros), torch.where(take_high, sign, zeros)]
    )


def _assign_trit_planes(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Assign every weight the nearest of the nine free sums `t1·s1 + t2·s2`.

    Positional, like `_assign_dual`: plane 0 carries `scales[:, 0]`, so the quantizer's
    `(s1, s2)` ordering is applied on the way in and undone on the way out.
    """
    first, second = scales[:, 1:], scales[:, :1]
    planes = quant.trit_plane_codes(weight, first, second)
    return torch.stack([planes[1], planes[0]]).to(torch.float32)


def _objective_two_plane(
    weight: torch.Tensor, planes: torch.Tensor, scales: torch.Tensor, scale_mode: str
) -> torch.Tensor:
    residual = weight - dequantize_fit(planes, scales, scale_mode=scale_mode)
    return (residual * residual).sum(-1)


def _solve_dual(
    weight: torch.Tensor, planes: torch.Tensor, ridge_lambda: float
) -> torch.Tensor:
    low, high = planes[0], planes[1]
    return solve_2x2_ridge(
        (low * low).sum(-1),
        (low * high).sum(-1),
        (high * high).sum(-1),
        (weight * low).sum(-1),
        (weight * high).sum(-1),
        ridge_lambda,
    ).clamp_min(quant.SCALE_EPS)


def solve_2x2_ridge(
    g00: torch.Tensor,
    g01: torch.Tensor,
    g11: torch.Tensor,
    r0: torch.Tensor,
    r1: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    """Solve `[[g00, g01], [g01, g11]] x = [r0, r1]` per row with a κ-adaptive ridge.

    Returns:
        fp32 `(rows, 2)` solution.
    """
    ridge = _adaptive_ridge(g00, g01, g11, ridge_lambda)
    a00, a11 = g00 + ridge, g11 + ridge
    det = (a00 * a11 - g01 * g01).clamp_min(_TINY)
    return torch.stack([(a11 * r0 - g01 * r1) / det, (a00 * r1 - g01 * r0) / det], -1)


def _adaptive_ridge(
    g00: torch.Tensor, g01: torch.Tensor, g11: torch.Tensor, base: float
) -> torch.Tensor:
    """Return the per-row ridge: `base·λmax`, boosted by `sqrt(κ / trigger)` once κ trips it.

    Both terms scale with the system, so the solve is equivariant: multiplying the
    whole 2x2 and its right-hand side by any positive constant leaves the solution
    unchanged.
    """
    wide = [tensor.to(torch.float64) for tensor in (g00, g01, g11)]
    spread = torch.sqrt((wide[0] - wide[2]) ** 2 + 4.0 * wide[1] * wide[1])
    largest = (0.5 * (wide[0] + wide[2] + spread)).clamp_min(_TINY)
    # via the determinant, not `trace - spread`: the difference of two near-equal
    # sums loses the small eigenvalue exactly where the ridge has to react
    smallest = (wide[0] * wide[2] - wide[1] * wide[1]).clamp_min(0.0) / largest
    kappa = largest / smallest.clamp_min(_TINY)
    boosted = (base * torch.sqrt(kappa / RIDGE_KAPPA_TRIGGER)).clamp(base, RIDGE_MAX)
    fraction = torch.where(
        kappa >= RIDGE_KAPPA_TRIGGER, boosted, torch.full_like(kappa, base)
    )
    return (fraction * largest).to(g00.dtype)


__all__ = [
    "CONVERGENCE_TOL",
    "MAX_ITERS",
    "PER_TENSOR_MODES",
    "RIDGE_KAPPA_TRIGGER",
    "RIDGE_LAMBDA",
    "RIDGE_MAX",
    "SCALE_MODES",
    "TWN_THRESHOLD",
    "dequantize_fit",
    "fit_binary",
    "fit_ternary",
    "solve_2x2_ridge",
    "solve_scale",
]
