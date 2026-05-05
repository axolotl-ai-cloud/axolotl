"""Effective PCIe bandwidth model for the ProTrain cost estimators (§3.3).

When ``n_swap > 0`` activation-swap traffic (forward offload, backward
prefetch) competes with chunk prefetch/offload traffic on the same PCIe
link. ProTrain's cost model derates the prefetch bandwidth so the
runtime estimator does not under-predict backward time — but ONLY for
chunks whose prefetch window temporally overlaps the active swap
traffic. Per the paper:

    §3.3: "we estimate the swapping time, identify the affected chunks,
    and use the reduced bandwidth instead."

Earlier this module returned a single scalar derate per direction and
the runtime estimator applied it uniformly to every chunk's prefetch.
That over-penalised configurations because chunks whose prefetch never
overlaps a swap operation actually run at full PCIe bandwidth.

The current model exposes two pieces:

- :func:`effective_bw` — the derated (h2d, d2h) pair used for the
  *affected* subset of non-persistent chunks. Same arithmetic as before;
  retained for backward compatibility and as the per-chunk "affected"
  bandwidth.
- :func:`n_affected_chunks` — count of non-persistent chunks whose
  prefetch overlaps swap traffic (per paper §3.3 / §A.1). The runtime
  estimator splits its per-chunk PCIe sum into ``n_affected`` chunks
  costed at the derated bandwidth and ``n_unaffected`` chunks costed at
  full PCIe bandwidth.

Refine the affected-chunk identification against measured contention if
a later test shows a >5% runtime mismatch vs. observed
``torch.cuda.Event`` timing for a workload with non-trivial ``n_swap``.

Paper references: §3.3 "bandwidth contention is modeled explicitly"; §A.1
swap-bandwidth identification.
"""

from __future__ import annotations

from axolotl.integrations.protrain.types import ChunkLayout, CostConfig, HardwareProfile
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def effective_bw(cfg: CostConfig, hw: HardwareProfile) -> tuple[float, float]:
    """Return ``(effective_h2d_bps, effective_d2h_bps)`` for AFFECTED chunks.

    When ``cfg.n_swap == 0`` the raw PCIe bandwidths are returned unchanged
    (no swap traffic, nothing to contend with). When ``cfg.n_swap > 0`` the
    effective bandwidth is reduced by a factor
    ``1 / (1 + 0.5 * min(1, n_swap / max(1, gpu_count)))``. The factor
    bottoms out at ``2/3`` when every rank has at least one swap block
    competing for the link — matching the paper's qualitative claim that
    "unlimited" swap degrades prefetch throughput by roughly a third.

    This is the bandwidth used for chunks whose prefetch window overlaps
    swap traffic — see :func:`n_affected_chunks` for the chunk-count
    selector. Chunks whose prefetch falls outside the swap window pay
    full ``hw.pcie_h2d_bps`` / ``hw.pcie_d2h_bps`` and do NOT consume
    this derated value.

    Parameters
    ----------
    cfg:
        The candidate knob configuration being costed.
    hw:
        Static hardware description; only ``pcie_h2d_bps``,
        ``pcie_d2h_bps``, and ``gpu_count`` are consulted.

    Returns
    -------
    tuple[float, float]
        Effective H2D and D2H bandwidths in bytes / second for the
        affected-chunks subset.
    """
    gpu_count = max(1, hw.gpu_count)
    if cfg.n_swap <= 0:
        return hw.pcie_h2d_bps, hw.pcie_d2h_bps

    # First-order contention model. See module docstring for refinement
    # guidance; the 0.5 slope and the clamp at gpu_count were picked to
    # keep the derate monotone in n_swap without letting a single swap
    # block on one rank halve the bandwidth for the entire cluster.
    contention = 0.5 * min(1.0, cfg.n_swap / gpu_count)
    denom = 1.0 + contention
    eff_h2d = hw.pcie_h2d_bps / denom
    eff_d2h = hw.pcie_d2h_bps / denom
    LOG.debug(
        "effective_bw: n_swap=%d gpu_count=%d derate=%.3f h2d=%.2e d2h=%.2e",
        cfg.n_swap,
        gpu_count,
        denom,
        eff_h2d,
        eff_d2h,
    )
    return eff_h2d, eff_d2h


def n_affected_chunks(cfg: CostConfig, layout: ChunkLayout) -> int:
    """How many non-persistent chunks have prefetch windows overlapping swap.

    Per paper §3.3: "we estimate the swapping time, **identify the
    affected chunks**, and use the reduced bandwidth instead." This
    function is the identification step. The runtime estimator then
    splits the per-chunk PCIe sum into ``n_affected`` chunks at the
    derated bandwidth (:func:`effective_bw`) and the remaining
    non-persistent chunks at full PCIe bandwidth.

    Formula
    -------
    ``n_affected = min(n_swap + 1, N_chunk - n_persist)``

    Rationale (paper §3.3 + ``block/layout_rules.py:121-123``):

    - Layout rule 1 ("swap-early") puts swap blocks in positions
      ``[0, n_swap)``. Active D2H traffic happens during forward(b) for
      every swap block ``b``; H2D traffic happens during backward(b).
    - The runtime prefetches the chunk for block ``b+1`` during
      block ``b``'s forward (one prefetch-depth lookahead). So during
      forward of swap blocks ``[0, n_swap)``, the chunks being
      prefetched belong to blocks ``[1, n_swap+1]`` — that's ``n_swap``
      block positions, plus one extra position past the last swap block
      where the prefetch window still trails into swap traffic. With
      ~one chunk per block in well-balanced layouts (the typical
      transformer case), this corresponds to roughly ``n_swap + 1``
      chunks.
    - Symmetric argument for backward (swap H2D + chunk prefetch
      contention) yields the same affected count.
    - Clamp at ``N_chunk - n_persist`` because persistent chunks never
      leave GPU and contribute zero prefetch traffic — they cannot be
      "affected" by definition.

    Returns 0 when ``n_swap <= 0`` (no swap traffic, no contention) so
    the runtime collapses to full-bandwidth costing for every chunk —
    matching the original no-contention behaviour.

    When ``n_swap`` is large enough that the formula saturates at
    ``N_chunk - n_persist``, every non-persistent chunk is affected and
    the per-chunk model becomes equivalent to the old flat-derate
    behaviour at full ``effective_bw``.

    Parameters
    ----------
    cfg:
        The candidate knob configuration being costed.
    layout:
        Chunk layout (``N_chunk`` is consulted).

    Returns
    -------
    int
        Number of non-persistent chunks whose prefetch overlaps swap
        traffic; clamped to ``[0, N_chunk - n_persist]``.
    """
    if cfg.n_swap <= 0:
        return 0
    n_nonpersist = max(0, layout.N_chunk - cfg.n_persist)
    return min(cfg.n_swap + 1, n_nonpersist)


__all__ = ["effective_bw", "n_affected_chunks"]
