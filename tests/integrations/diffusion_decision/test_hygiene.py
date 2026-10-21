"""Leakage checks use state identity, not labels or source-specific row IDs."""

import pytest

from axolotl.integrations.diffusion_decision.hygiene import (
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


def test_image_records_fingerprint_image_bytes(tmp_path):
    first, second, copy = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    first.write_bytes(b"image-a")
    second.write_bytes(b"image-b")
    copy.write_bytes(b"image-a")
    evaluation = [{**row("What is shown?", "eval"), "images": [str(first)]}]
    other_image = {**row("What is shown?", "t1"), "images": [str(second)]}
    same_bytes = {**row("What is shown?", "t2"), "images": [str(copy)]}
    kept, counts = decontaminate([other_image, same_bytes], evaluation)
    assert kept == [other_image]
    assert counts["state_overlap"] == 1
    with pytest.raises(ValueError, match="state overlaps"):
        assert_split_isolation({"train": [same_bytes], "eval": evaluation})
    assert_split_isolation({"train": [other_image], "eval": evaluation})
    url = {**row("What is shown?", "t3"), "images": ["https://x/a.png"]}
    assert decontaminate([url], evaluation)[0] == [url]
