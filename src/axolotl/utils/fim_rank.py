"""FIM-guided automatic LoRA rank allocation.

Uses the diagonal of the empirical Fisher Information Matrix (eFIM) to
measure per-layer loss sensitivity on a small calibration batch, then
reallocates LoRA ranks so that information-critical layers receive higher
rank and less sensitive layers receive lower rank — subject to a fixed
total-rank budget (mean rank preserved).

Algorithm
---------
    F_ii ≈ (1/T) Σ (∂ℓ_t / ∂θ_i)²     # eFIM diagonal (mean squared gradient)
    score_i = mean(F_i)                  # per-layer importance
    rank_i  ∝ score_i / Σ score_j × budget   # budget = n_layers × base_r
    rank_i  = clamp(rank_i, r_min, r_max)     # integer, largest-remainder

Reference: Optimal Brain Damage (LeCun et al., NeurIPS 1990).

Usage
-----
Enable in your YAML config::

    adapter: lora
    lora_r: 16
    fim_auto_rank: true          # enable FIM-guided rank allocation
    fim_calibration_batches: 8   # calibration batches (default 8)
    fim_rank_min: 1              # minimum rank per layer (default 1)
    fim_rank_max: 32             # maximum rank per layer (default 2 * lora_r)

After ``get_peft_model()`` is called, ``apply_fim_ranks()`` runs a short
calibration pass and resizes each adapter's ``lora_A``/``lora_B`` in-place
before training starts.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

LOG = logging.getLogger("axolotl.utils.fim_rank")


def _get_lora_b_params(
    model: nn.Module, adapter_name: str = "default"
) -> dict[str, torch.Tensor]:
    """Return a layer-name → lora_B parameter mapping via named_parameters().

    We track lora_B (not lora_A) because standard LoRA initialises lora_B=0,
    which makes ∂loss/∂lora_A = scaling × lora_B^T × upstream = 0 at init.
    lora_B receives a meaningful gradient immediately since lora_A is
    kaiming-initialised.
    """
    params = {}
    suffix = f".lora_B.{adapter_name}.weight"
    for param_name, param in model.named_parameters():
        if param_name.endswith(suffix):
            layer_name = param_name[: -len(suffix)]
            param.requires_grad_(True)
            params[layer_name] = param
    return params


def _accumulate_fim(
    model: nn.Module,
    dataloader: "DataLoader",
    n_batches: int,
    adapter_name: str = "default",
) -> dict[str, torch.Tensor]:
    """Run calibration forward+backward and accumulate mean squared gradients."""
    lora_b_params = _get_lora_b_params(model, adapter_name)
    if not lora_b_params:
        warnings.warn(
            "[fim_rank] No lora_B parameters found. "
            "Check that target_modules are set and the model is PEFT-wrapped.",
            stacklevel=2,
        )
        return {}

    fim_accum: dict[str, torch.Tensor] = {}
    fim_steps: dict[str, int] = defaultdict(int)
    device = next(model.parameters()).device

    was_training = model.training
    model.train()

    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= n_batches:
                break

            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            loss.backward()

            with torch.no_grad():
                for layer_name, param in lora_b_params.items():
                    if param.grad is None:
                        continue
                    grad_sq = param.grad.detach() ** 2
                    if layer_name not in fim_accum:
                        fim_accum[layer_name] = torch.zeros_like(grad_sq)
                    fim_accum[layer_name].add_(grad_sq)
                    fim_steps[layer_name] += 1

            model.zero_grad()

    finally:
        model.train(was_training)

    if not fim_accum:
        warnings.warn(
            "[fim_rank] No gradients captured during FIM calibration. "
            "Check that the dataloader produces a loss.",
            stacklevel=2,
        )
        return {}

    result = {
        name: fim_accum[name] / max(fim_steps[name], 1) for name in fim_accum
    }

    all_zero = all(v.abs().max().item() < 1e-12 for v in result.values())
    if all_zero:
        warnings.warn(
            "[fim_rank] All FIM scores are zero. Gradients did not flow through "
            "LoRA layers. Ranks will remain unchanged.",
            stacklevel=2,
        )

    return result


def _allocate_ranks(
    importance: dict[str, float],
    base_r: int,
    r_min: int,
    r_max: int,
) -> dict[str, int]:
    """Allocate integer ranks proportional to importance under a fixed budget.

    Budget = n_layers × base_r (mean rank preserved).
    Uses a water-filling + largest-remainder algorithm.
    """
    if not importance:
        return {}

    names = list(importance.keys())
    scores = {n: max(importance[n], 1e-10) for n in names}
    budget = base_r * len(names)

    result: dict[str, int] = {}
    free = list(names)
    free_budget = float(budget)

    # Phase 1: fix layers that saturate r_max
    while free:
        total = sum(scores[n] for n in free)
        raw = {n: scores[n] / total * free_budget for n in free}
        saturated = [n for n in free if math.floor(raw[n]) >= r_max]
        if not saturated:
            break
        for n in saturated:
            result[n] = r_max
        free_budget -= len(saturated) * r_max
        free = [n for n in free if n not in result]

    # Phase 2: largest-remainder rounding over remaining layers
    if free:
        total = sum(scores[n] for n in free)
        raw = {n: scores[n] / total * free_budget for n in free}
        floors = {n: math.floor(raw[n]) for n in free}
        remainder = int(free_budget) - sum(floors.values())
        by_frac = sorted(free, key=lambda n: -(raw[n] - floors[n]))
        for i, n in enumerate(by_frac):
            floors[n] += 1 if i < remainder else 0
        for n in free:
            result[n] = max(r_min, min(r_max, floors[n]))

    return result


def _resize_layer(
    layer: nn.Module,
    adapter_name: str,
    new_r: int,
    adjust_scaling: bool,
) -> None:
    """Resize lora_A and lora_B to new_r in-place, preserving existing weights."""
    if adapter_name not in layer.lora_A:
        return

    lora_A = layer.lora_A[adapter_name]
    lora_B = layer.lora_B[adapter_name]
    old_r = lora_A.weight.shape[0]
    if old_r == new_r:
        return

    device, dtype = lora_A.weight.device, lora_A.weight.dtype
    in_f, out_f = lora_A.weight.shape[1], lora_B.weight.shape[0]
    copy_r = min(old_r, new_r)

    new_A = torch.zeros(new_r, in_f, device=device, dtype=dtype)
    new_B = torch.zeros(out_f, new_r, device=device, dtype=dtype)

    with torch.no_grad():
        new_A[:copy_r].copy_(lora_A.weight[:copy_r])
        new_B[:, :copy_r].copy_(lora_B.weight[:, :copy_r])
        if new_r > old_r:
            nn.init.kaiming_uniform_(new_A[old_r:], a=math.sqrt(5))

    lora_A.weight = nn.Parameter(new_A)
    lora_B.weight = nn.Parameter(new_B)
    layer.r[adapter_name] = new_r

    if adjust_scaling and adapter_name in layer.scaling:
        layer.scaling[adapter_name] *= old_r / new_r


def apply_fim_ranks(
    model: nn.Module,
    dataloader: "DataLoader",
    base_r: int,
    n_batches: int = 8,
    r_min: int = 1,
    r_max: Optional[int] = None,
    adjust_scaling: bool = True,
    adapter_name: str = "default",
) -> dict[str, int]:
    """Reallocate LoRA ranks using FIM-guided layer importance scores.

    Call this after ``get_peft_model()`` and before training begins.

    Args:
        model: A PEFT-wrapped model with LoRA adapters.
        dataloader: Calibration dataloader — uses the first ``n_batches``
            batches from the training set.
        base_r: Base LoRA rank (defines the per-layer budget).
        n_batches: Number of calibration batches. Default 8.
        r_min: Minimum rank per layer. Default 1.
        r_max: Maximum rank per layer. Defaults to ``2 * base_r``.
        adjust_scaling: Rescale ``lora_alpha`` after rank change to preserve
            the effective ``lora_alpha / r`` scaling. Default True.
        adapter_name: Active adapter name. Default ``"default"``.

    Returns:
        Mapping from layer name to allocated rank (for logging).
    """
    try:
        from peft.tuners.lora.layer import Linear as LoraLinear
    except ImportError:
        LOG.warning("[fim_rank] PEFT not installed — skipping FIM rank allocation.")
        return {}

    r_max = r_max if r_max is not None else 2 * base_r
    LOG.info(
        "[fim_rank] Running FIM calibration (%d batches, base_r=%d, r_min=%d, r_max=%d)",
        n_batches,
        base_r,
        r_min,
        r_max,
    )

    fim_scores = _accumulate_fim(model, dataloader, n_batches, adapter_name)
    if not fim_scores:
        LOG.warning("[fim_rank] No FIM scores — ranks unchanged.")
        return {}

    importance = {name: fim.mean().item() for name, fim in fim_scores.items()}
    rank_pattern = _allocate_ranks(importance, base_r=base_r, r_min=r_min, r_max=r_max)

    # Apply resizing
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear) and name in rank_pattern:
            _resize_layer(module, adapter_name, rank_pattern[name], adjust_scaling)

    # Persist in config for serialisation
    if hasattr(model, "peft_config") and adapter_name in model.peft_config:
        cfg = model.peft_config[adapter_name]
        updates = {k: v for k, v in rank_pattern.items() if v != base_r}
        cfg.rank_pattern.update(updates)

    # Log allocation
    changed = {k: v for k, v in rank_pattern.items() if v != base_r}
    LOG.info(
        "[fim_rank] Rank allocation complete. %d/%d layers changed.",
        len(changed),
        len(rank_pattern),
    )
    for name, r in sorted(rank_pattern.items(), key=lambda x: -x[1]):
        marker = "▲" if r > base_r else ("▼" if r < base_r else "=")
        LOG.debug("[fim_rank]  %s r=%d  %s", marker, r, name)

    return rank_pattern
