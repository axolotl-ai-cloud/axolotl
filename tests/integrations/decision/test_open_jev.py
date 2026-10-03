"""Source target semantics and option-order regression checks."""

import math

import pytest

from axolotl.integrations.decision.adapters import normalize_open_jev


def record(**updates):
    result = {
        "id": "example/0",
        "group_id": "family/0",
        "source": "authored",
        "kind": "choice",
        "question": "Choose an answer",
        "options": ["a", "b", "c"],
        "target": [1.0, 0.0, 0.0],
        "state_json": '{"fact": 1}',
        "metadata_json": '{"target_basis": "rule"}',
    }
    result.update(updates)
    return result


def test_boolean_source_order_is_remapped_by_identity():
    actual = normalize_open_jev(
        record(kind="noul", options=["no", "yes"], target=[0, 1])
    )
    assert actual["labels"]["q1"] == {"kind": "hard", "gold_idx": 0}
    assert actual["questions"]["q1"]["type"] == "noul"


def test_human_distribution_preserved_and_wanli_separated():
    actual = normalize_open_jev(
        record(
            source="wanli-decisions-v1",
            kind="noul",
            options=["false", "true"],
            target=[0.3, 0.7],
            metadata_json='{"target_basis":"published_human_reviewed_label"}',
        )
    )
    assert actual["source"] == "open_jev.wanli"
    assert actual["labels"]["q1"] == {"kind": "dist", "probs": [0.7, 0.3]}


def test_uniform_optimal_actions_are_a_set_not_uncertainty():
    actual = normalize_open_jev(
        record(
            target=[0.5, 0.5, 0],
            metadata_json={
                "target_basis": "exact minimax; ties are uniform optimal actions, not win probabilities"
            },
        )
    )
    assert actual["labels"]["q1"] == {"kind": "set", "allowed_set": [0, 1]}
    stochastic = normalize_open_jev(
        record(
            target=[0.5, 0.5, 0],
            metadata_json={"target_basis": "defined_uniform_latent_worlds"},
        )
    )
    assert stochastic["labels"]["q1"]["kind"] == "dist"


def test_ir_setwise_metadata_preserves_valid_alternative_set():
    actual = normalize_open_jev(
        record(
            source="ir-control-v1",
            target=[0.5, 0, 0.5],
            metadata_json={"method": "setwise"},
        )
    )
    assert actual["labels"]["q1"] == {"kind": "set", "allowed_set": [0, 2]}


def test_ir_pairwise_uniform_target_is_an_exact_tie_set():
    actual = normalize_open_jev(
        record(
            source="ir-control-v1",
            target=[0.5, 0.5, 0],
            metadata_json={"method": "pairwise"},
        )
    )
    assert actual["labels"]["q1"] == {"kind": "set", "allowed_set": [0, 1]}
    with pytest.raises(ValueError, match="explicit target_basis"):
        normalize_open_jev(
            record(
                source="ir-control-v1",
                target=[0.25, 0.75, 0],
                metadata_json={"method": "pairwise"},
            )
        )


def test_snake_heuristic_action_ties_are_sets_but_collision_is_hard():
    basis = (
        "Choice: visible-state BFS/flood-fill heuristic, not optimal. "
        "Noul: exact one-step collision rule."
    )
    action = normalize_open_jev(
        record(target=[0.5, 0, 0.5], metadata_json={"target_basis": basis})
    )
    assert action["labels"]["q1"] == {"kind": "set", "allowed_set": [0, 2]}
    collision = normalize_open_jev(
        record(
            kind="noul",
            options=["no", "yes"],
            target=[0, 1],
            metadata_json={"target_basis": basis},
        )
    )
    assert collision["labels"]["q1"] == {"kind": "hard", "gold_idx": 0}


def test_unknown_soft_targets_require_explicit_semantics():
    row = record(target=[0.5, 0.5, 0])
    with pytest.raises(ValueError, match="explicit target_basis"):
        normalize_open_jev(row)
    actual = normalize_open_jev(row, target_basis={"rule": "set"})
    assert actual["labels"]["q1"]["allowed_set"] == [0, 1]
    with pytest.raises(ValueError, match="refusing to discard"):
        normalize_open_jev(row, target_basis={"rule": "hard"})


@pytest.mark.parametrize(
    "values", [[math.nan, 0, 0], [-0.1, 0.6, 0.5], [0.1, 0.1, 0.1]]
)
def test_invalid_probabilities_are_rejected(values):
    with pytest.raises(ValueError, match="probabilities"):
        normalize_open_jev(record(target=values))


def test_counterfactual_states_are_not_replaced_by_their_group():
    first = normalize_open_jev(record())
    second = normalize_open_jev(record(id="example/1", state_json='{"fact": 2}'))
    assert first["group"] == second["group"]
    assert first["family"] == first["group"]
    assert first["state"] != second["state"]


def test_explicit_scenario_family_is_retained_over_template_metadata():
    actual = normalize_open_jev(
        record(
            group_id="source-parent-8",
            metadata_json={
                "scenario_family": "support/damaged_order",
                "template_id": "generic-template",
            },
        )
    )
    assert actual["group"] == "source-parent-8"
    assert actual["family"] == "support/damaged_order"


def test_invalid_explicit_scenario_family_is_rejected():
    with pytest.raises(ValueError, match="scenario_family"):
        normalize_open_jev(record(metadata_json={"scenario_family": ""}))
