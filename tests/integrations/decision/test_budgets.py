import pytest

from axolotl.integrations.decision.budgets import (
    decision_budget,
    exceeds_budget,
)
from axolotl.model_support import DiffusionLayout


@pytest.mark.parametrize(
    "layout", [DiffusionLayout.FULL_SEQUENCE, DiffusionLayout.ENCODER_CANVAS]
)
def test_fixed_canvas_cost_charges_prompt_and_full_canvas_for_both_layouts(layout):
    budget = decision_budget(prompt_tokens=19, canvas_tokens=128, layout=layout)
    assert budget.logical_tokens == 147
    assert budget.physical_tokens == 147
    assert exceeds_budget(budget, logical_limit=146, physical_limit=None) == "logical"
    assert exceeds_budget(budget, logical_limit=147, physical_limit=146) == "physical"
