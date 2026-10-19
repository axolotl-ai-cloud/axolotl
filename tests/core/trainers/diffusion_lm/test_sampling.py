"""Native overflow filtering precedes batch and optimizer-step scheduling."""

import pytest
from datasets import Dataset

from axolotl.core.trainers.diffusion_lm.sampling import (
    filter_native_diffusion_dataset,
    native_packing_lengths,
    resolve_native_packing_budget,
)
from axolotl.utils.dict import DictDefault


def config(sequence_len=12, micro_batch_size=1, **options):
    return DictDefault(
        {
            "model_config_type": "diffusion_gemma",
            "sample_packing": True,
            "sequence_len": sequence_len,
            "micro_batch_size": micro_batch_size,
            "diffusion_lm": {"canvas_width": 4, "overflow_policy": "drop", **options},
        }
    )


def records():
    return Dataset.from_list(
        [
            {"input_ids": list(range(n)), "labels": [-100, -100] + list(range(2, n))}
            for n in (6, 10, 7)
        ]
    )


def test_drop_charges_encoder_and_canvas_before_scheduling():
    dataset = records()
    filtered = filter_native_diffusion_dataset(config(), dataset, split="train")
    assert list(filtered["input_ids"]) == [list(range(6)), list(range(7))]
    assert native_packing_lengths(
        filtered,
        layout="encoder_canvas",
        canvas_width=4,
        eos_tail=None,
        logical_sequence_length=None,
    ) == [10, 11]
    assert len(dataset) == 3


def test_error_reports_original_row_before_sampler():
    with pytest.raises(ValueError, match="example 1.*1 oversized"):
        filter_native_diffusion_dataset(
            config(overflow_policy="error"), records(), split="train"
        )


def test_logical_overflow_is_filtered_before_eos_cost_computation():
    filtered = filter_native_diffusion_dataset(
        config(sequence_len=6, micro_batch_size=2, eos_tail="visible_supervised"),
        records(),
        split="train",
    )
    assert len(filtered) == 1


def test_drop_all_is_an_explicit_error():
    with pytest.raises(ValueError, match="all native diffusion train"):
        filter_native_diffusion_dataset(
            config(sequence_len=2), records(), split="train"
        )


def test_packing_budget_uses_the_active_batch_size_only_for_packed_rows():
    cfg = config(sequence_len=128, micro_batch_size=2)
    assert resolve_native_packing_budget(cfg, packed=False) is None
    budget = resolve_native_packing_budget(cfg, packed=True, batch_size=3)
    assert budget is not None
    assert budget.total == 384


def test_legacy_and_missing_eval_are_unchanged():
    dataset = records()
    cfg = config(from_causal_lm=True)
    assert filter_native_diffusion_dataset(cfg, dataset, split="train") is dataset
    assert filter_native_diffusion_dataset(config(), None, split="eval") is None


@pytest.mark.parametrize(
    ("model_type", "budget", "payload", "reserved"),
    [
        ("diffusion_gemma", 384, 256, 128),
        ("Dream", 384, 384, 0),
    ],
)
def test_flex_payload_capacity_accounts_for_its_physical_streams(
    model_type, budget, payload, reserved
):
    cfg = config(sequence_len=budget)
    cfg.model_config_type = model_type
    cfg.attn_implementation = "flex_attention"
    resolved = resolve_native_packing_budget(cfg)
    assert resolved is not None
    assert resolved.total == budget
    assert resolved.payload_capacity == payload
    assert resolved.reserved_capacity == reserved
    assert resolved.bucket_size == 128


@pytest.mark.parametrize(
    ("model_type", "budget", "match"),
    [
        ("diffusion_gemma", 128, "require at least 256"),
        ("Dream", 127, "require at least 128"),
    ],
)
def test_flex_rejects_packed_rows_without_a_full_payload_bucket(
    model_type, budget, match
):
    cfg = config(sequence_len=budget)
    cfg.model_config_type = model_type
    cfg.attn_implementation = "flex_attention"
    with pytest.raises(ValueError, match=match):
        resolve_native_packing_budget(cfg)


@pytest.mark.parametrize("preprocess", [False, True])
def test_standard_preparation_filters_before_step_estimation(monkeypatch, preprocess):
    from axolotl.utils.data import sft

    dataset = records()

    class Loader:
        def __init__(self, cfg):
            pass

        def load(self, load_fn):
            return dataset, None, []

        def cleanup(self):
            pass

    observed = []

    def estimate(cfg, filtered):
        observed.append(list(filtered["input_ids"]))
        return len(filtered)

    monkeypatch.setattr(sft, "FileLockLoader", Loader)
    monkeypatch.setattr(sft, "calculate_total_num_steps", estimate)
    monkeypatch.setenv("AXOLOTL_IS_PREPROCESS", "1" if preprocess else "0")
    train, evaluation, steps, _ = sft._prepare_standard_dataset(config(), None, None)
    assert len(train) == 2
    assert evaluation is None
    assert steps == (-1 if preprocess else 2)
    assert observed == ([] if preprocess else [list(train["input_ids"])])
