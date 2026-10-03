"""Leakage checks use state identity, not labels or source-specific row IDs."""

import pytest

from axolotl.integrations.decision.hygiene import (
    assert_split_isolation,
    decontaminate,
    state_fingerprint,
)


def row(state, family="family-a", source="source-a"):
    return {"state": state, "family": family, "source": source, "labels": {}}


def test_canonical_state_identity():
    assert state_fingerprint('{ "b": 2, "a": 1 }') == state_fingerprint(
        {"a": 1, "b": 2}
    )
    assert state_fingerprint("cafe\u0301 \n value") == state_fingerprint("café value")
    assert state_fingerprint({"value": [1, 2]}) != state_fingerprint({"value": [2, 1]})


def test_cross_source_decontamination_and_family_exclusion():
    evaluation = [row({"x": 1})]
    candidates = [
        row('{"x":1}', "other", "source-b"),
        row({"x": 2}),
        row({"x": 3}, "family-b"),
    ]
    kept, counts = decontaminate(iter(candidates), iter(evaluation))
    assert kept == [candidates[2]]
    assert counts == {"input": 3, "state_overlap": 1, "family_overlap": 1, "kept": 1}
    assert candidates[0]["labels"] == {}


def test_official_split_state_filter_can_preserve_shared_families():
    candidates = [row("different state")]
    kept, counts = decontaminate(
        candidates, [row("evaluation")], exclude_families=False
    )
    assert kept == candidates
    assert counts["family_overlap"] == 0


@pytest.mark.parametrize("second", [row("same", "other", "source-b"), row("different")])
def test_split_guard_rejects_state_or_family_overlap(second):
    with pytest.raises(ValueError, match="overlaps splits"):
        assert_split_isolation({"train": [row("same")], "test": [second]})


def test_family_names_are_source_scoped_and_repetitions_within_split_are_valid():
    assert_split_isolation(
        {"train": [row("a"), row("b")], "test": [row("c", source="source-b")]}
    )


def test_missing_state_or_family_is_not_silently_accepted():
    with pytest.raises(ValueError, match="state"):
        state_fingerprint(None)
    with pytest.raises(ValueError, match="family"):
        decontaminate([{"source": "x", "state": "y"}], [])
