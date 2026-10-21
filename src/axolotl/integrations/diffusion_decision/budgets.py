"""Pre-mixture token-budget checks for fixed-width decision canvases."""

from __future__ import annotations

from dataclasses import dataclass

from axolotl.model_support import DiffusionLayout


@dataclass(frozen=True)
class DecisionBudget:
    logical_tokens: int
    physical_tokens: int


def decision_budget(
    *, prompt_tokens: int, canvas_tokens: int, layout: DiffusionLayout
) -> DecisionBudget:
    if prompt_tokens < 0 or canvas_tokens <= 0:
        raise ValueError("decision prompt and canvas lengths must be nonnegative")
    if layout not in {DiffusionLayout.FULL_SEQUENCE, DiffusionLayout.ENCODER_CANVAS}:
        raise ValueError(f"unsupported decision layout: {layout}")
    total = prompt_tokens + canvas_tokens
    return DecisionBudget(logical_tokens=total, physical_tokens=total)


def exceeds_budget(
    budget: DecisionBudget,
    *,
    logical_limit: int | None,
    physical_limit: int | None,
) -> str | None:
    if logical_limit is not None and budget.logical_tokens > logical_limit:
        return "logical"
    if physical_limit is not None and budget.physical_tokens > physical_limit:
        return "physical"
    return None
