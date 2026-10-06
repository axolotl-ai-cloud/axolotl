from collections import Counter

import pytest

from axolotl.integrations.diffusion_decision.mixture import (
    DeterministicMixtureSampler,
    prepare_source_pools,
)


def _record(source, group, state, questions=1):
    return {
        "source": source,
        "group": group,
        "state": state,
        "questions": {str(index): {} for index in range(questions)},
    }


def test_prepare_source_pools_decontaminates_and_caps_each_source_deterministically():
    training = [
        _record("alpha", "alpha-0", "keep-a"),
        _record("alpha", "alpha-1", "keep-b"),
        _record("alpha", "alpha-2", "overlap"),
        _record("beta", "beta-0", "keep-c"),
    ]
    evaluation = [_record("eval", "eval-0", "overlap")]

    first = prepare_source_pools(
        training, evaluation, max_examples_per_source={"alpha": 1}, seed=7
    )
    second = prepare_source_pools(
        training, evaluation, max_examples_per_source={"alpha": 1}, seed=7
    )

    assert first.records == second.records
    assert set(first.records) == {"alpha", "beta"}
    assert len(first.records["alpha"]) == 1
    assert first.counts == {
        "input": 4,
        "state_overlap": 1,
        "family_overlap": 0,
        "kept": 3,
        "capped": 1,
        "sources": 2,
    }
    assert all(
        record["state"] != "overlap"
        for pool in first.records.values()
        for record in pool
    )


def test_prepare_source_pools_removes_training_families_held_out_for_evaluation():
    training = [
        _record("alpha", "shared-family", "train-only"),
        _record("beta", "beta-family", "keep"),
    ]
    evaluation = [_record("alpha", "shared-family", "eval-only")]

    prepared = prepare_source_pools(training, evaluation)

    assert set(prepared.records) == {"beta"}
    assert prepared.counts["family_overlap"] == 1


def test_explicit_weight_draw_shares_match_within_two_percent_over_ten_thousand():
    sampler = DeterministicMixtureSampler(
        {"a": tuple(range(3)), "b": tuple(range(5)), "c": tuple(range(7))},
        weights={"a": 0.5, "b": 0.3, "c": 0.2},
        seed=19,
    )

    counts = Counter(sample.source for sample in sampler.draw(10_000))

    for source, expected in sampler.probabilities.items():
        assert abs(counts[source] / 10_000 - expected) <= 0.02


def test_temperature_uses_capped_source_sizes():
    sampler = DeterministicMixtureSampler(
        {"small": tuple(range(4)), "large": tuple(range(16))}, temperature=0.5
    )

    assert sampler.probabilities == {
        "large": pytest.approx(2 / 3),
        "small": pytest.approx(1 / 3),
    }


def test_probability_normalization_handles_finite_extreme_explicit_weights():
    sampler = DeterministicMixtureSampler(
        {"a": (1,), "b": (2,)}, weights={"a": 1e308, "b": 1e308}
    )

    assert sampler.probabilities == {"a": 0.5, "b": 0.5}


def test_temperature_zero_is_uniform_and_large_finite_temperature_is_stable():
    uniform = DeterministicMixtureSampler(
        {"small": (1,), "large": tuple(range(16))}, temperature=0
    )
    stable = DeterministicMixtureSampler(
        {"small": (1,), "large": (2,)}, temperature=1000
    )

    assert uniform.probabilities == {"large": 0.5, "small": 0.5}
    assert all(0 < probability < 1 for probability in stable.probabilities.values())


@pytest.mark.parametrize(
    ("pools", "weights", "temperature"),
    [
        (
            {"small": (1,), "large": tuple(range(2))},
            {"small": 1e-308, "large": 1e308},
            None,
        ),
        ({"small": (1,), "large": tuple(range(2))}, None, 2000),
        ({"small": (1,), "large": tuple(range(16))}, None, 1e308),
    ],
)
def test_probability_construction_rejects_source_underflow_or_overflow(
    pools, weights, temperature
):
    with pytest.raises(ValueError, match="underflowed|nonfinite"):
        DeterministicMixtureSampler(pools, weights=weights, temperature=temperature)


def test_stratified_batches_include_sources_with_at_least_one_expected_slot():
    sampler = DeterministicMixtureSampler(
        {source: tuple(range(3)) for source in ("a", "b", "c", "d")},
        weights={"a": 0.5, "b": 0.25, "c": 0.125, "d": 0.125},
        seed=3,
    )

    batches = list(sampler.batches(8, 30, stratified=True))

    assert all(len(batch) == 8 for batch in batches)
    assert all(
        {sample.source for sample in batch} == {"a", "b", "c", "d"} for batch in batches
    )
    assert all(
        Counter(sample.source for sample in batch) == {"a": 4, "b": 2, "c": 1, "d": 1}
        for batch in batches
    )


def test_fractional_stratification_converges_to_source_probabilities():
    sampler = DeterministicMixtureSampler(
        {source: tuple(range(3)) for source in ("a", "b", "c")},
        weights={"a": 0.6, "b": 0.25, "c": 0.15},
        seed=3,
    )

    counts = Counter(
        sample.source
        for batch in sampler.batches(4, 2_500, stratified=True)
        for sample in batch
    )

    for source, expected in sampler.probabilities.items():
        assert abs(counts[source] / 10_000 - expected) <= 0.02


def test_seed_reproduces_draws_and_stratified_batches():
    pools = {"a": tuple(range(4)), "b": tuple(range(6))}
    first = DeterministicMixtureSampler(pools, weights={"a": 0.4, "b": 0.6}, seed=11)
    second = DeterministicMixtureSampler(pools, weights={"a": 0.4, "b": 0.6}, seed=11)

    assert first.draw(40) == second.draw(40)
    assert list(first.batches(5, 20)) == list(second.batches(5, 20))


def test_epoch_batches_cover_every_capped_source_record_before_stopping():
    sampler = DeterministicMixtureSampler(
        {"small": tuple(range(2)), "large": tuple(range(7))},
        weights={"small": 0.5, "large": 0.5},
        seed=11,
    )

    batches = list(sampler.epoch_batches(4))
    records = {
        source: {
            sample.record
            for batch in batches
            for sample in batch
            if sample.source == source
        }
        for source in ("small", "large")
    }

    assert all(len(batch) == 4 for batch in batches)
    assert records == {"small": {0, 1}, "large": set(range(7))}


def test_non_stratified_epoch_is_seeded_and_covers_every_source_record():
    pools = {"a": tuple(range(2)), "b": tuple(range(5))}
    first = DeterministicMixtureSampler(pools, weights={"a": 0.2, "b": 0.8}, seed=7)
    second = DeterministicMixtureSampler(pools, weights={"a": 0.2, "b": 0.8}, seed=7)

    batches = list(first.epoch_batches(3, stratified=False))
    assert batches == list(second.epoch_batches(3, stratified=False))
    assert all(len(batch) == 3 for batch in batches)
    assert {
        (sample.source, sample.record) for batch in batches for sample in batch
    } >= {("a", 0), ("a", 1), *{("b", value) for value in range(5)}}


@pytest.mark.parametrize(
    ("pools", "weights", "temperature", "error"),
    [
        ({}, None, None, "at least one"),
        ({"a": ()}, None, None, "empty"),
        ({"a": (1,)}, {"missing": 1.0}, None, "every nonempty"),
        ({"a": (1,)}, {"a": 1.0}, 0.5, "or temperature"),
        ({"a": (1,)}, {"a": 0.0}, None, "positive"),
    ],
)
def test_sampler_rejects_empty_or_invalid_source_definitions(
    pools, weights, temperature, error
):
    with pytest.raises(ValueError, match=error):
        DeterministicMixtureSampler(pools, weights=weights, temperature=temperature)


def test_examples_remain_whole_records_when_sampled():
    record = _record("alpha", "family", "state", questions=3)
    sampler = DeterministicMixtureSampler({"alpha": (record,)}, seed=2)

    sample = sampler.draw(1)[0]

    assert sample.source == "alpha"
    assert sample.record is record
    assert len(sample.record["questions"]) == 3


def test_prepare_rejects_cap_for_source_removed_by_decontamination():
    training = [_record("alpha", "alpha", "overlap")]
    evaluation = [_record("eval", "eval", "overlap")]

    with pytest.raises(ValueError, match="unavailable"):
        prepare_source_pools(training, evaluation, max_examples_per_source={"alpha": 1})


def test_prepare_rejects_when_decontamination_removes_every_source():
    training = [_record("alpha", "alpha", "overlap")]
    evaluation = [_record("eval", "eval", "overlap")]

    with pytest.raises(ValueError, match="no nonempty"):
        prepare_source_pools(training, evaluation)
