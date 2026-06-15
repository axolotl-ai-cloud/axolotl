"""Helpers for the Q-GaLore optimizer integration.

Q-GaLore (arxiv 2407.08296) projects gradients into a low-rank subspace using a
periodically-refreshed projection matrix P. The upstream wheel
(``q-galore-torch``) exposes ``QGaLoreAdamW8bit``; it discovers which parameters
to project by reading a ``rank`` key on each ``param_group``. This module
builds those param-groups from an Axolotl config.

The companion INT8-weight-wrapping recipe from the paper is not yet wired up
(see ``check_qgalore`` in :mod:`axolotl.utils.schemas.validation`).
"""

from __future__ import annotations

import inspect
import types

from torch import nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_q_galore_for_modern_bnb() -> None:
    """bnb >=0.44 inserted (beta3, alpha) into ``optimizer_update_8bit_blockwise``
    and ``optimizer_update_32bit``; q-galore-torch==1.0 still calls the legacy
    positional layout. Swap q_galore's bnb handle for one that re-emits the
    modern layout. No-op on older bnb."""
    import bitsandbytes.functional as F
    import q_galore_torch.q_galore_adamw8bit as mod

    if "beta3" not in inspect.signature(F.optimizer_update_8bit_blockwise).parameters:
        return

    bw, fp32 = F.optimizer_update_8bit_blockwise, F.optimizer_update_32bit
    mod.F = types.SimpleNamespace(
        optimizer_update_8bit_blockwise=(
            lambda *a, **kw: bw(
                *(a[:7] + (0.0, 0.0) + a[7:] if len(a) == 15 else a), **kw
            )
        ),
        optimizer_update_32bit=(
            lambda *a, **kw: fp32(
                *(a[:10] + (0.0, 0.0) + a[10:] if len(a) == 13 else a), **kw
            )
        ),
        optimizer_update_8bit=F.optimizer_update_8bit,
        percentile_clipping=F.percentile_clipping,
    )


def build_qgalore_param_groups(
    model: nn.Module,
    target_modules: list[str],
    *,
    rank: int,
    update_proj_gap: int,
    scale: float,
    proj_type: str,
    proj_quant: bool,
    proj_bits: int,
    proj_group_size: int,
    cos_threshold: float,
    gamma_proj: int,
    queue_size: int,
) -> list[dict]:
    """Split ``model``'s trainable parameters into two groups for Q-GaLore.

    The first group carries the Q-GaLore projection settings (``rank``,
    ``update_proj_gap`` etc.). The second is a plain AdamW group for everything
    that wasn't matched by ``target_modules`` (norms, biases, embeddings, …).

    ``target_modules`` is a list of substring patterns matched against
    parameter names — identical semantics to ``optim_target_modules`` for the
    upstream HuggingFace GaLore integration.
    """
    galore, plain = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Only 2D weight matrices benefit from the low-rank projection; 1D
        # tensors (norms, biases) go to the plain AdamW group.
        if p.dim() == 2 and any(t in name for t in target_modules):
            galore.append(p)
        else:
            plain.append(p)
    if not galore:
        raise ValueError(
            "Q-GaLore: no parameters matched optim_target_modules="
            f"{target_modules!r}. Check the pattern list against the model's "
            "parameter names."
        )
    LOG.info("Q-GaLore param groups: %d projected, %d plain", len(galore), len(plain))
    return [
        {
            "params": galore,
            "rank": rank,
            "update_proj_gap": update_proj_gap,
            "scale": scale,
            "proj_type": proj_type,
            "quant": proj_quant,
            "quant_n_bit": proj_bits,
            "quant_group_size": proj_group_size,
            "cos_threshold": cos_threshold,
            "gamma_proj": gamma_proj,
            "queue_size": queue_size,
        },
        {"params": plain},
    ]
