"""
Tests for compensating pre-prepared KD datasets whose baked target_* columns use the
legacy alignment, and for the alignment detector that flags a mis-declared dataset.
"""

import math

import pytest
import torch

from axolotl.integrations.kd import KDPlugin, collator as collator_module
from axolotl.integrations.kd.collator import (
    DataCollatorForKD,
    KDBatchSamplerDataCollatorForSeq2Seq,
)
from axolotl.integrations.kd.utils import LOGPROB_PAD_VALUE
from axolotl.utils.dict import DictDefault

TOP_K = 4


class FakeTokenizer:
    """Minimal stand-in for the padding step the KD collator delegates."""

    padding_side = "right"

    def __init__(self):
        self.deprecation_warnings = {}

    def pad(self, features, **_kwargs):
        return {
            key: torch.as_tensor([list(feature[key]) for feature in features])
            for key in features[0]
        }


def make_collator(alignment, packed=False):
    cls = KDBatchSamplerDataCollatorForSeq2Seq if packed else DataCollatorForKD
    return cls(FakeTokenizer(), kd_prepared_targets_alignment=alignment)


def teacher_row(token_id, rank=0, top_k=TOP_K):
    """A stored top-k row whose ``rank``-th entry is ``token_id``."""
    token_ids = [900_000 + token_id * 10 + i for i in range(top_k)]
    token_ids[rank] = token_id
    logprobs = [math.log(p) for p in (0.5, 0.3, 0.15, 0.05)[:top_k]]
    if rank == 0:
        logprobs[0] = math.log(0.6)
    return token_ids, logprobs


def current_targets(input_ids, labels, ranks=None):
    """Baked targets in the current convention: row j describes input_ids[j + 1]."""
    seq_len = len(input_ids)
    target_token_ids, target_logprobs, target_mask = [], [], []
    for j in range(seq_len):
        described = j + 1
        if described >= seq_len or labels[described] == -100:
            target_token_ids.append([0] * TOP_K)
            target_logprobs.append([LOGPROB_PAD_VALUE] * TOP_K)
            target_mask.append([0] * TOP_K)
            continue
        ids, logprobs = teacher_row(
            input_ids[described], rank=0 if ranks is None else ranks[j]
        )
        target_token_ids.append(ids)
        target_logprobs.append(logprobs)
        target_mask.append([1] * TOP_K)
    return target_token_ids, target_logprobs, target_mask


def to_legacy(target_token_ids, target_logprobs, target_mask):
    """Inverse of the collator's compensation: row j describes input_ids[j]."""
    return (
        [[0] * TOP_K] + target_token_ids[:-1],
        [[LOGPROB_PAD_VALUE] * TOP_K] + target_logprobs[:-1],
        [[0] * TOP_K] + target_mask[:-1],
    )


def make_feature(input_ids, labels, alignment, ranks=None):
    ids, logprobs, mask = current_targets(input_ids, labels, ranks=ranks)
    if alignment == "legacy":
        ids, logprobs, mask = to_legacy(ids, logprobs, mask)
    return {
        "input_ids": list(input_ids),
        "labels": list(labels),
        "target_token_ids": ids,
        "target_logprobs": logprobs,
        "target_mask": mask,
    }


def simple_sequence(start=100, seq_len=8, prompt_len=2):
    input_ids = [start + i for i in range(seq_len)]
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return input_ids, labels


@pytest.fixture(name="warnings_log")
def warnings_log_fixture(monkeypatch):
    messages = []
    monkeypatch.setattr(
        collator_module.LOG,
        "warning",
        lambda msg, *args, **kwargs: messages.append(str(msg)),
    )
    return messages


def assert_pinned(batch, input_ids, labels, seq_index=0):
    """The collated batch must satisfy the loss convention: row j describes token j+1."""
    target_token_ids = batch["target_token_ids"][seq_index]
    target_mask = batch["target_mask"][seq_index]

    for j in range(len(input_ids) - 1):
        if labels[j + 1] == -100:
            assert target_mask[j].tolist() == [0] * TOP_K
            continue
        assert target_token_ids[j][0].item() == input_ids[j + 1]
        assert target_mask[j].tolist() == [1] * TOP_K

    assert target_mask[len(input_ids) - 1].tolist() == [0] * TOP_K


def test_legacy_targets_are_shifted_into_the_loss_convention(warnings_log):
    input_ids, labels = simple_sequence()
    batch = make_collator("legacy")([make_feature(input_ids, labels, "legacy")])

    assert_pinned(batch, input_ids, labels)
    assert not warnings_log


def test_current_targets_are_left_alone(warnings_log):
    input_ids, labels = simple_sequence()
    batch = make_collator("current")([make_feature(input_ids, labels, "current")])

    assert_pinned(batch, input_ids, labels)
    assert not warnings_log


def test_legacy_declared_matches_current_data_exactly():
    input_ids, labels = simple_sequence()

    legacy_batch = make_collator("legacy")([make_feature(input_ids, labels, "legacy")])
    current_batch = make_collator("current")(
        [make_feature(input_ids, labels, "current")]
    )

    assert torch.equal(
        legacy_batch["target_token_ids"], current_batch["target_token_ids"]
    )
    assert torch.equal(legacy_batch["target_mask"], current_batch["target_mask"])
    assert torch.allclose(
        legacy_batch["target_logprobs"], current_batch["target_logprobs"]
    )


def test_legacy_shift_pads_the_final_position():
    input_ids, labels = simple_sequence()
    feature = make_feature(input_ids, labels, "legacy")
    last_real_row = list(feature["target_token_ids"][-1])

    batch = make_collator("legacy")([feature])

    assert batch["target_mask"][0][-1].tolist() == [0] * TOP_K
    assert batch["target_logprobs"][0][-1].tolist() == [LOGPROB_PAD_VALUE] * TOP_K
    # the dropped row moved one position left, it was not discarded
    assert batch["target_token_ids"][0][-2].tolist() == last_real_row


def test_packed_legacy_shift_does_not_cross_sequence_boundaries():
    first_ids, first_labels = simple_sequence(start=100, seq_len=8)
    second_ids, second_labels = simple_sequence(start=500, seq_len=6)
    sub_batch = [
        make_feature(first_ids, first_labels, "legacy"),
        make_feature(second_ids, second_labels, "legacy"),
    ]

    batch = make_collator("legacy", packed=True)([sub_batch])

    target_token_ids = batch["target_token_ids"][0]
    target_mask = batch["target_mask"][0]

    for j in range(len(first_ids) - 1):
        if first_labels[j + 1] != -100:
            assert target_token_ids[j][0].item() == first_ids[j + 1]
    # the boundary row would describe the next sequence's first token if the shift had
    # run after concatenation
    boundary = len(first_ids) - 1
    assert target_mask[boundary].tolist() == [0] * TOP_K

    offset = len(first_ids)
    for j in range(len(second_ids) - 1):
        if second_labels[j + 1] != -100:
            assert target_token_ids[offset + j][0].item() == second_ids[j + 1]
    assert target_mask[offset + len(second_ids) - 1].tolist() == [0] * TOP_K


def test_packed_unpacked_agree_on_a_single_sequence():
    input_ids, labels = simple_sequence()

    packed = make_collator("legacy", packed=True)(
        [[make_feature(input_ids, labels, "legacy")]]
    )
    unpacked = make_collator("legacy")([make_feature(input_ids, labels, "legacy")])

    assert torch.equal(packed["target_token_ids"], unpacked["target_token_ids"])
    assert torch.equal(packed["target_mask"], unpacked["target_mask"])


def test_plugin_passes_the_declared_alignment_to_the_collator():
    cfg = DictDefault(
        {
            "kd_trainer": True,
            "sample_packing": True,
            "skip_prepare_dataset": True,
            "kd_prepared_targets_alignment": "legacy",
        }
    )

    collator_cls, kwargs = KDPlugin().get_collator_cls_and_kwargs(cfg)

    assert collator_cls is KDBatchSamplerDataCollatorForSeq2Seq
    assert kwargs == {"kd_prepared_targets_alignment": "legacy"}

    # the kwargs the builder would hand to the collator must actually construct one
    collator = collator_cls(FakeTokenizer(), **kwargs)
    input_ids, labels = simple_sequence()
    batch = collator([[make_feature(input_ids, labels, "legacy")]])
    assert_pinned(batch, input_ids, labels)


def sampled_generation_batch(alignment, batch_size=4, seq_len=24, prompt_len=2):
    """
    A dataset that looks like sampled (not greedy) generation: the actual next token is
    the teacher's argmax only ~40% of the time, so top-1 equality is a weak signal.
    """
    features = []
    ranks = []
    for b in range(batch_size):
        input_ids = [1000 * (b + 1) + i for i in range(seq_len)]
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        seq_ranks = [0 if (j % 5) < 2 else 1 + (j % 3) for j in range(seq_len)]
        ranks.append(seq_ranks)
        features.append(make_feature(input_ids, labels, alignment, ranks=seq_ranks))
    return features, ranks


def repetitive_batch(alignment, batch_size=4, seq_len=24, prompt_len=2):
    """
    Text with repeated tokens, so the wrong hypothesis also collects real mass and the
    detector has to weigh the two rather than see one of them at zero.
    """
    features = []
    for b in range(batch_size):
        input_ids: list[int] = []
        for i in range(seq_len):
            repeat = i % 3 == 0 and input_ids
            input_ids.append(input_ids[-1] if repeat else 1000 * (b + 1) + i)
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        seq_ranks = [0 if (j % 5) < 2 else 1 + (j % 3) for j in range(seq_len)]
        features.append(make_feature(input_ids, labels, alignment, ranks=seq_ranks))
    return features


def test_detector_is_quiet_when_the_wrong_hypothesis_also_scores(warnings_log):
    """Repeated tokens give the alternative real mass; the declared one still wins."""
    collator = make_collator("current")
    collator(repetitive_batch("current"))

    assert collator._alignment_checked
    assert not warnings_log


def test_detector_still_fires_when_both_hypotheses_score(warnings_log):
    collator = make_collator("current")
    collator(repetitive_batch("legacy"))

    assert collator._alignment_checked
    assert len(warnings_log) == 1
    assert "kd_prepared_targets_alignment: legacy" in warnings_log[0]


def test_sampled_generation_fixture_has_weak_top1_agreement():
    _, ranks = sampled_generation_batch("current")
    flat = [rank for seq in ranks for rank in seq]
    top1_rate = sum(rank == 0 for rank in flat) / len(flat)
    assert 0.3 < top1_rate < 0.5


def test_detector_is_quiet_on_correctly_declared_current_data(warnings_log):
    features, _ = sampled_generation_batch("current")
    collator = make_collator("current")
    collator(features)

    assert collator._alignment_checked
    assert not warnings_log


def test_detector_is_quiet_on_correctly_declared_legacy_data(warnings_log):
    features, _ = sampled_generation_batch("legacy")
    collator = make_collator("legacy")
    collator(features)

    assert collator._alignment_checked
    assert not warnings_log


def test_detector_fires_on_legacy_data_declared_current(warnings_log):
    features, _ = sampled_generation_batch("legacy")
    make_collator("current")(features)

    assert len(warnings_log) == 1
    message = warnings_log[0]
    assert "kd_prepared_targets_alignment: legacy" in message
    assert "containment" in message
    assert "positions" in message


def test_detector_fires_on_current_data_declared_legacy(warnings_log):
    features, _ = sampled_generation_batch("current")
    make_collator("legacy")(features)

    assert len(warnings_log) == 1
    assert "kd_prepared_targets_alignment: current" in warnings_log[0]


def test_detector_fires_for_packed_batches(warnings_log):
    features, _ = sampled_generation_batch("legacy", batch_size=4, seq_len=24)
    make_collator("current", packed=True)([features])

    assert len(warnings_log) == 1


def test_detector_runs_only_once(warnings_log):
    collator = make_collator("current")
    for _ in range(3):
        features, _ = sampled_generation_batch("legacy")
        collator(features)

    assert len(warnings_log) == 1


def test_detector_skips_batches_with_too_few_valid_positions(warnings_log):
    collator = make_collator("current")
    input_ids, labels = simple_sequence()
    collator([make_feature(input_ids, labels, "legacy")])

    # too few positions to judge, so no warning yet and the check stays armed
    assert not warnings_log
    assert not collator._alignment_checked

    features, _ = sampled_generation_batch("legacy")
    collator(features)
    assert len(warnings_log) == 1
