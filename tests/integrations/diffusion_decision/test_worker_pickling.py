"""Decision datasets in DataLoader worker processes (spawn / forkserver / fork).

Python 3.14 made ``forkserver`` the default start method on Linux, which pickles the
dataset and collate function into every worker. These tests build the train and
eval datasets through the real loader (no patched row builder, so the workers run
the same code as the parent) and check that rows produced by workers equal the
``num_workers=0`` rows, in order, for two epochs.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import multiprocessing
import pickle
from collections.abc import Mapping
from typing import Any

import pytest
from torch.utils.data import DataLoader, SequentialSampler

from axolotl.integrations.diffusion_decision import datasets
from axolotl.integrations.diffusion_decision.slot_sampling import DecisionDraw
from axolotl.integrations.diffusion_decision.trainer import _DecisionPlanBatchSampler
from axolotl.integrations.diffusion_decision.training_collator import (
    DecisionTrainingCollator,
)
from axolotl.model_support import DiffusionLayout, DiffusionNoise

from .helpers import WorkerTokenizer, make_record, make_spec

SPEC = make_spec(noise=DiffusionNoise.ABSORBING, layout=DiffusionLayout.FULL_SEQUENCE)
EPOCHS = (0, 1)


def _canonical(value: Any) -> Any:
    """Type-tagged canonical form of a row or batch, in container order.

    Equal iff types, values and dict order are equal. Raw pickle bytes are not
    comparable across a worker hop: pickle's memo encodes which objects are
    shared, and a round trip (in or out of process) changes that layout.
    """
    import torch

    if isinstance(value, Mapping):
        return (
            "map",
            type(value).__qualname__,
            tuple((_canonical(key), _canonical(item)) for key, item in value.items()),
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            "dataclass",
            type(value).__qualname__,
            tuple(
                (field.name, _canonical(getattr(value, field.name)))
                for field in dataclasses.fields(value)
            ),
        )
    if isinstance(value, (list, tuple)):
        return (type(value).__qualname__, tuple(_canonical(item) for item in value))
    if isinstance(value, torch.Tensor):
        return ("tensor", str(value.dtype), tuple(value.shape), value.tolist())
    if isinstance(value, (str, bytes, int, float, bool, type(None), enum.Enum)):
        return (type(value).__qualname__, repr(value))
    raise TypeError(f"no canonical form for {type(value)!r}")


def _record(source: str, index: int) -> dict[str, Any]:
    identifier = f"{source}{index}"
    return make_record(
        identifier,
        source=source,
        group=identifier,
        state=f"state of {identifier}: " + "x" * (index % 5),
        questions={
            "pick": {
                "type": "choice",
                "instructions": "Pick one.",
                "options": [f"option {letter}" for letter in "abcde"],
            },
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "options": ["billing", "ops", "tech"],
            },
        },
        labels={
            "pick": {"kind": "hard", "gold_idx": index % 5},
            "team": {"kind": "hard", "gold_idx": index % 3},
        },
    )


def _write(path, records) -> str:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return str(path)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    root = tmp_path_factory.mktemp("worker-datasets")
    train = _write(
        root / "train.jsonl",
        [_record("alpha", index) for index in range(12)]
        + [_record("beta", index) for index in range(4)],
    )
    dev = _write(root / "dev.jsonl", [_record("gamma", index) for index in range(5)])
    cfg = {
        "seed": 13,
        "micro_batch_size": 2,
        "sequence_len": 4096,
        "datasets": [
            {"path": "json", "data_files": train, "type": "diffusion_decision.jsonl"}
        ],
        "test_datasets": [
            {
                "path": "json",
                "data_files": dev,
                "split": "dev",
                "type": "diffusion_decision.jsonl",
            }
        ],
        "model_config": {"vocab_size": 256, "mask_token_id": 9},
        "diffusion_lm": {"canvas_width": 128, "mask_token_id": 9},
        "diffusion_decision": {
            "max_questions_per_canvas": 20,
            "labels": {},
            "mixture": {
                "weights": {"alpha": 0.5, "beta": 0.5},
                "max_examples_per_source": {"alpha": 4},
                "per_batch_stratified": True,
            },
        },
    }
    with pytest.MonkeyPatch.context() as patch:
        # The spec resolves in the parent only; workers receive it pickled.
        patch.setattr(datasets, "require_diffusion_spec", lambda _cfg: SPEC)
        meta = datasets.load_decision_datasets(cfg, tokenizer=WorkerTokenizer())
    assert isinstance(meta.train_dataset, datasets.DecisionPlanDataset)
    assert isinstance(meta.eval_dataset, datasets.DecisionDataset)
    assert meta.train_dataset.batch_plan.reserve_records  # later epochs build lazily
    return meta.train_dataset, meta.eval_dataset


def _collator() -> DecisionTrainingCollator:
    return DecisionTrainingCollator(spec=SPEC, pad_token_id=0)


def test_train_and_eval_datasets_pickle(loaded):
    train, evals = loaded
    for dataset in (train, evals, _collator()):
        restored = pickle.loads(pickle.dumps(dataset))
        assert type(restored) is type(dataset)
    restored = pickle.loads(pickle.dumps(train))
    for epoch in EPOCHS:
        for position in range(len(train)):
            draw = DecisionDraw(
                index=position, epoch=epoch, global_draw_ordinal=position
            )
            assert _canonical(restored[draw]) == _canonical(train[draw])
    restored = pickle.loads(pickle.dumps(evals))
    assert _canonical([restored[index] for index in range(len(evals))]) == _canonical(
        [evals[index] for index in range(len(evals))]
    )


def test_pickled_builder_keeps_only_the_config_a_build_reads(loaded):
    builder = loaded[0].batch_plan._build_row
    assert isinstance(builder, datasets._PlanRowBuilder)
    state = builder.__getstate__()
    assert set(state["_cfg"]) <= set(datasets._ROW_BUILDER_CFG_KEYS)
    assert "datasets" not in state["_cfg"]


def _epoch_batches(dataset, *, workers, context=None, persistent=False, collate=list):
    """Train batches of every epoch in EPOCHS, the epoch set the way the trainer does."""
    sampler = _DecisionPlanBatchSampler(dataset.batch_plan)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=workers,
        multiprocessing_context=context,
        persistent_workers=persistent,
    )
    batches = {}
    try:
        for epoch in EPOCHS:
            sampler.set_epoch(epoch)
            batches[epoch] = list(loader)
    finally:
        sampler.set_epoch(0)
    return batches


def _eval_batches(dataset, *, workers, context=None, collate=list):
    loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=2,
        collate_fn=collate,
        num_workers=workers,
        multiprocessing_context=context,
    )
    return list(loader)


class _RowsAndBatch:
    """collate_fn yielding the raw rows and the collated batch (module-level: picklable)."""

    def __init__(self) -> None:
        self.collator = _collator()

    def __call__(self, rows):
        return list(rows), self.collator(rows)


@pytest.mark.parametrize(
    ("context", "persistent"),
    [("spawn", False), ("forkserver", False), ("forkserver", True), ("fork", False)],
)
def test_worker_rows_match_in_process_rows(loaded, context, persistent):
    """Rows and collated batches of 2 workers equal num_workers=0, in order, per epoch.

    The epoch reaches workers in each DecisionDraw the sampler yields in the parent,
    so persistent workers (whose dataset copy never sees set_epoch) agree as well.
    """
    if context not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{context} start method unavailable")
    train, evals = loaded
    collate = _RowsAndBatch()
    reference = _epoch_batches(train, workers=0, collate=collate)
    assert _canonical(reference[0]) != _canonical(reference[1])  # epoch 1 re-draws
    assert {
        row["decision_draw"].epoch
        for rows, _ in reference[1]
        for row in rows
        if "decision_draw" in row
    } <= {1}
    got = _epoch_batches(
        train, workers=2, context=context, persistent=persistent, collate=collate
    )
    for epoch in EPOCHS:
        assert len(got[epoch]) == len(reference[epoch])
        assert _canonical(got[epoch]) == _canonical(reference[epoch]), epoch
    assert _canonical(
        _eval_batches(evals, workers=2, context=context, collate=collate)
    ) == _canonical(_eval_batches(evals, workers=0, collate=collate))
