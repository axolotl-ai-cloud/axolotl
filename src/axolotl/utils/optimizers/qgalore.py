"""Helpers for the Q-GaLore optimizer integration."""

from __future__ import annotations

import inspect
import types

from torch import nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_q_galore_for_modern_bnb() -> None:
    """q-galore-torch==1.0 targets the pre-0.44 bnb optimizer API; adapt it to
    the installed bnb:

    - bnb >=0.44 inserted (beta3, alpha) into ``optimizer_update_8bit_blockwise``
      and ``optimizer_update_32bit``: re-emit q_galore's legacy positional layout.
    - bnb >=0.50 removed the non-blockwise 8-bit path: ``F.optimizer_update_8bit``
      and ``F.percentile_clipping`` are gone, ``get_config()`` lost the
      ``percentile_clipping``/``block_wise`` keys, and — silently worst —
      ``Optimizer2State.__init__`` lost those two params so q_galore's positional
      ``super().__init__`` call lands them on ``max_unorm``/``skip_zeros``.

    No-op on bnb <0.44."""
    import bitsandbytes.functional as F
    import q_galore_torch.q_galore_adamw8bit as mod
    from bitsandbytes.optim.optimizer import Optimizer2State

    if "beta3" not in inspect.signature(F.optimizer_update_8bit_blockwise).parameters:
        return

    bw, fp32 = F.optimizer_update_8bit_blockwise, F.optimizer_update_32bit
    legacy = {
        name: getattr(F, name)
        for name in ("optimizer_update_8bit", "percentile_clipping")
        if hasattr(F, name)
    }
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
        **legacy,
    )

    ctor_params = inspect.signature(Optimizer2State.__init__).parameters
    if "percentile_clipping" in ctor_params or getattr(
        mod.AdamW8bit, "_axolotl_bnb050_patched", False
    ):
        return

    def _init(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
        is_paged=False,
    ):
        if percentile_clipping != 100 or not block_wise:
            raise ValueError(
                "bnb >=0.50 removed percentile_clipping and non-blockwise 8-bit "
                "optimizer support"
            )
        Optimizer2State.__init__(
            self,
            "adam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            8,
            args,
            min_8bit_size,
            is_paged=is_paged,
        )

    orig_get_config = mod.AdamW8bit.get_config

    def _get_config(self, gindex, pindex, group):
        config = orig_get_config(self, gindex, pindex, group)
        config.setdefault("percentile_clipping", 100)
        config.setdefault("block_wise", True)
        return config

    mod.AdamW8bit.__init__ = _init
    mod.AdamW8bit.get_config = _get_config
    mod.AdamW8bit._axolotl_bnb050_patched = True


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
    """Two param-groups: 2D weights matching ``target_modules`` get the Q-GaLore
    projection keys; everything else (norms, biases, embeddings) is plain AdamW."""
    galore, plain = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 2 and any(t in name for t in target_modules):
            galore.append(p)
        else:
            plain.append(p)
    if not galore:
        raise ValueError(
            f"Q-GaLore: no parameters matched optim_target_modules={target_modules!r}"
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
