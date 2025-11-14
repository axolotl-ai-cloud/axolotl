"""Utilities for tagging parameters that should receive Muon updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import torch
import torch.nn as nn

from axolotl.utils.schemas.muon import MuonClipConfig


@dataclass(frozen=True)
class MuonParameterInfo:
    """Metadata describing how a parameter should be handled by MuonClip."""

    name: str
    shape: torch.Size
    use_muon: bool
    reason: str


@dataclass
class ParameterTagSummary:
    """Aggregate counts for Muon vs non-Muon parameters."""

    total: int = 0
    muon: int = 0
    non_muon: int = 0

    def record(self, use_muon: bool) -> None:
        self.total += 1
        if use_muon:
            self.muon += 1
        else:
            self.non_muon += 1


def tag_parameters_for_muon(
    module_or_named_params: nn.Module
    | Iterable[tuple[str, nn.Parameter]],
    config: MuonClipConfig,
    *,
    min_ndim: int = 2,
) -> tuple[dict[str, MuonParameterInfo], ParameterTagSummary]:
    """
    Inspect parameters and set `param.use_muon` so DeepSpeed/FSDP can split optimizer paths.

    Returns dictionaries keyed by parameter name plus a summary with aggregate counts.
    """

    named_params = _resolve_named_parameters(module_or_named_params)
    metadata: dict[str, MuonParameterInfo] = {}
    summary = ParameterTagSummary()

    include_filters = config.apply_to or []
    exclude_filters = config.exclude or []

    for name, param in named_params:
        use_muon, reason = _should_use_muon(
            name,
            param,
            include_filters=include_filters,
            exclude_filters=exclude_filters,
            min_ndim=min_ndim,
        )
        setattr(param, "use_muon", use_muon)
        info = MuonParameterInfo(
            name=name,
            shape=param.shape,
            use_muon=use_muon,
            reason=reason,
        )
        metadata[name] = info
        summary.record(use_muon)

    return metadata, summary


def _resolve_named_parameters(
    module_or_named_params: nn.Module
    | Iterable[tuple[str, nn.Parameter]],
) -> Iterator[tuple[str, nn.Parameter]]:
    if isinstance(module_or_named_params, nn.Module):
        return module_or_named_params.named_parameters()
    return iter(module_or_named_params)


def _matches_filter(name: str, filters: Sequence[str]) -> str | None:
    for flt in filters:
        if flt and flt in name:
            return flt
    return None


def _should_use_muon(
    name: str,
    param: nn.Parameter,
    *,
    include_filters: Sequence[str],
    exclude_filters: Sequence[str],
    min_ndim: int,
) -> tuple[bool, str]:
    if not param.requires_grad:
        return False, "requires_grad_false"

    include_match = _matches_filter(name, include_filters)
    exclude_match = _matches_filter(name, exclude_filters)

    if include_match:
        if param.ndim < min_ndim:
            # Explicit opt-in but still warn through reason for logging.
            return True, f"forced:{include_match}"
        return True, f"include:{include_match}"

    if exclude_match:
        return False, f"exclude:{exclude_match}"

    if param.ndim < min_ndim:
        return False, f"ndim<{min_ndim}"

    return True, "default"
