"""
Detect and fix weight scale anomalies in MoE/hybrid models.

In MoE models trained with AdamW, rarely-activated experts accumulate smaller
second-moment estimates, giving them a disproportionately large effective
learning rate.  Over time this causes their weights to drift to higher
variance than the median for the same tensor across layers.

For recurrent / SSM / DeltaNet components (e.g. ``conv1d.weight`` in linear
attention layers), this drift corrupts the hidden state and degrades long-
context performance — the model "forgets" after a few tokens.

This module provides a configurable transform that detects outlier weight
scales per-tensor-pattern and rescales them to the group median.
"""

import re
from collections import defaultdict

import torch

from axolotl.utils.distributed import is_main_process
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def normalize_weight_scales(model, rules):
    """Normalize weight scales for tensor groups that have outlier variance.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded model (before adapter injection).
    rules : list[dict]
        Each rule is a dict with keys:

        - ``name_pattern`` (str): regex matched against each named parameter.
          Parameters that match are grouped together, and outliers within the
          group are rescaled.
        - ``threshold`` (float, default 1.5): a parameter is flagged when its
          std deviates from the group median by more than this factor
          (ratio > threshold or ratio < 1/threshold).
        - ``dry_run`` (bool, default False): when True, log anomalies but do
          not modify weights.

    Returns
    -------
    int
        Number of tensors that were rescaled (0 in dry-run mode).
    """
    total_fixed = 0

    for rule in rules:
        pattern = rule.get("name_pattern")
        if not pattern:
            LOG.warning("normalize_weight_scales: rule missing 'name_pattern', skipping")
            continue

        threshold = float(rule.get("threshold", 1.5))
        dry_run = bool(rule.get("dry_run", False))
        regex = re.compile(pattern)

        # Collect matching tensors
        matches = []
        for name, param in model.named_parameters():
            if regex.search(name):
                with torch.no_grad():
                    std = param.data.float().std().item()
                matches.append((name, param, std))

        if len(matches) < 3:
            if is_main_process():
                LOG.info(
                    f"normalize_weight_scales: pattern '{pattern}' matched "
                    f"{len(matches)} tensors (need >=3 to detect outliers), skipping"
                )
            continue

        # Compute group median std
        stds = [s for _, _, s in matches]
        median_std = float(sorted(stds)[len(stds) // 2])

        if median_std < 1e-10:
            continue

        # Detect and fix outliers
        outliers = []
        for name, param, std in matches:
            ratio = std / median_std
            if ratio > threshold or ratio < (1.0 / threshold):
                outliers.append((name, param, std, ratio))
                if not dry_run:
                    scale_factor = median_std / std
                    param.data.mul_(scale_factor)
                    total_fixed += 1

        # Report
        if is_main_process() and outliers:
            mode = "DRY RUN" if dry_run else "FIXED"
            LOG.warning(
                f"normalize_weight_scales [{mode}]: pattern '{pattern}' — "
                f"{len(outliers)}/{len(matches)} tensors outside "
                f"{threshold:.1f}x threshold (median std={median_std:.6f}):"
            )
            for name, _, std, ratio in outliers:
                LOG.warning(f"  {name}: std={std:.6f} ({ratio:.2f}x median)")
        elif is_main_process():
            LOG.info(
                f"normalize_weight_scales: pattern '{pattern}' — "
                f"{len(matches)} tensors, all within {threshold:.1f}x threshold "
                f"(median std={median_std:.6f})"
            )

    return total_fixed
