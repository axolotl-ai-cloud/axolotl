"""Canvas layer / per-epoch batch plan split of decision training data."""

from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace
from typing import Any

import pytest
from torch.utils.data import DataLoader

from axolotl.integrations.diffusion_decision import canvas, datasets, prepared_cache
from axolotl.integrations.diffusion_decision.batch_plan import (
    BudgetLimits,
    DecisionBatchPlan,
    SourcePool,
    decode_reserve,
    encode_reserve,
)
from axolotl.integrations.diffusion_decision.mixture import cap_window
from axolotl.integrations.diffusion_decision.permutation import permute_record
from axolotl.integrations.diffusion_decision.slot_sampling import DecisionDraw
from axolotl.integrations.diffusion_decision.trainer import (
    _DecisionPlanBatchSampler,
)
from axolotl.model_support import DiffusionLayout

from tests.integrations.diffusion_decision.helpers import make_record
from tests.integrations.diffusion_decision.test_datasets import (
    SPEC,
    FakeTokenizer,
    _cfg,
    _entry,
    _record,
    _unit_canvas,
)

LAYOUT = DiffusionLayout.FULL_SEQUENCE


def _row(source: str, identifier: str, prompt: int = 1) -> dict[str, Any]:
    return {
        "canvas": SimpleNamespace(prompt_ids=(1,) * prompt, canvas_ids=(2,) * 8),
        "record": {"id": identifier, "source": source},
        "source": source,
        "length": prompt + 8,
    }


def _pool(
    source: str, count: int, cap: int | None, seed: int, *, long=()
) -> SourcePool:
    """A source whose epoch-0 window is prepared and the rest held in reserve."""
    entries: list[Any] = []
    records: list[Any] = []
    window = set(cap_window(count, cap, seed, source))
    for index in range(count):
        identifier = f"{source}{index}"
        record = {"id": identifier, "source": source}
        if index in window:
            entries.append(_row(source, identifier, 20 if index in long else 1))
            records.append(record)
        else:
            entries.append(encode_reserve(record))
            records.append(None)
    return SourcePool(source=source, cap=cap, entries=entries, records=records)


class _Builder:
    """Stands in for _canvas_row: records every lazy build and its draw identity."""

    def __init__(self, long: frozenset[str] | set[str] = frozenset()) -> None:
        self.calls: list[tuple[str, str, tuple[int, int]]] = []
        self.long = long

    def __call__(self, source, record, draw):
        self.calls.append((source, record["id"], draw))
        row = _row(source, record["id"], 20 if record["id"] in self.long else 1)
        row["draw"] = draw
        return row


_NO_LIMITS = BudgetLimits(logical=None, physical=None)


def _plan(
    *,
    seed: int = 7,
    micro: int = 2,
    stratified: bool = True,
    pools=None,
    limits: BudgetLimits = _NO_LIMITS,
    builder: _Builder | None = None,
    weights=None,
    temperature=None,
) -> DecisionBatchPlan:
    pools = pools or [_pool("alpha", 12, 4, seed), _pool("beta", 3, None, seed)]
    return DecisionBatchPlan(
        pools,
        seed=seed,
        micro_batch_size=micro,
        stratified=stratified,
        weights={"alpha": 0.5, "beta": 0.5} if weights is None else weights,
        temperature=temperature,
        limits=limits,
        layout=LAYOUT,
        build_row=builder or _Builder(),
    )


def _epoch_ids(plan: DecisionBatchPlan, epoch: int) -> list[tuple[str, int]]:
    return [plan.draw(epoch, position)[:2] for position in range(plan.draws_per_epoch)]


def test_reserve_records_round_trip():
    record = make_record("r", state={"nested": ["ü", 1]})
    assert decode_reserve(encode_reserve(record)) == record


def test_cap_window_rotates_through_the_source_before_repeating():
    first = cap_window(10, 4, 3, "alpha")
    assert first == cap_window(10, 4, 3, "alpha", epoch=0)
    second = cap_window(10, 4, 3, "alpha", epoch=1)
    assert not set(first) & set(second)
    third = cap_window(10, 4, 3, "alpha", epoch=2)
    assert len(set(first) | set(second) | set(third)) == 10
    assert cap_window(3, 4, 3, "alpha", epoch=5) == [0, 1, 2]
    assert cap_window(3, None, 3, "alpha", epoch=5) == [0, 1, 2]


def test_permutation_is_seeded_by_record_draw_and_epoch():
    record = make_record(
        "row",
        questions={
            "q": {
                "type": "choice",
                "instructions": "Pick.",
                "options": [f"option {index}" for index in range(8)],
            }
        },
        labels={"q": {"kind": "hard", "gold_idx": 3}},
    )

    def order(draw):
        result = permute_record(record, seed=5, draw=draw)
        options = list(result["questions"]["q"]["options"])
        # The gold option follows its option through every permutation.
        assert options[result["labels"]["q"]["gold_idx"]] == "option 3"
        return tuple(options)

    assert order((0, 0)) == order(None)
    orders = {order((epoch, number)) for epoch in range(3) for number in range(3)}
    assert len(orders) == 9
    assert order((1, 2)) == order((1, 2))


def test_epochs_draw_fresh_records_groupings_and_option_orders():
    builder = _Builder()
    plan = _plan(builder=builder)
    first, second = _epoch_ids(plan, 0), _epoch_ids(plan, 1)
    assert len(first) == len(second) == plan.draws_per_epoch
    assert plan.batches(0) != plan.batches(1)
    alpha_first = {index for source, index in first if source == "alpha"}
    alpha_second = {index for source, index in second if source == "alpha"}
    # max_examples_per_source is re-picked: epoch 1 draws alpha's next window.
    assert alpha_first == set(cap_window(12, 4, 7, "alpha"))
    assert alpha_second == set(cap_window(12, 4, 7, "alpha", epoch=1))
    assert not alpha_first & alpha_second
    # Epoch-0 first draws are the cached rows; every later draw is built lazily
    # with its own (epoch, draw number) permutation identity.
    rows0 = [plan.row(0, position) for position in range(plan.draws_per_epoch)]
    first_draws = {
        (row["source"], row["record"]["id"]) for row in rows0 if "draw" not in row
    }
    assert len(first_draws) == len(set(first))
    assert all(draw[0] == 0 and draw[1] > 0 for _, _, draw in builder.calls)
    builder.calls.clear()
    rows1 = [plan.row(1, position) for position in range(plan.draws_per_epoch)]
    assert all("draw" in row and row["draw"][0] == 1 for row in rows1)
    numbers = Counter((source, ident) for source, ident, _ in builder.calls)
    for (source, ident), count in numbers.items():
        assert sorted(
            draw[1] for s, i, draw in builder.calls if (s, i) == (source, ident)
        ) == list(range(count))


def test_same_seed_and_epoch_is_deterministic_across_instances_and_access_order():
    left, right = _plan(), _plan()
    right.epoch_draws(3)  # a restart may plan a later epoch first
    for epoch in (2, 0, 1):
        assert _epoch_ids(left, epoch) == _epoch_ids(right, epoch)
        assert left.batches(epoch) == right.batches(epoch)
        assert [left.row(epoch, p)["record"] for p in range(4)] == [
            right.row(epoch, p)["record"] for p in range(4)
        ]
    assert _epoch_ids(_plan(seed=8), 1) != _epoch_ids(left, 1)


def test_every_epoch_keeps_the_stratified_composition_and_batch_count():
    plan = _plan(micro=4, weights={"alpha": 0.75, "beta": 0.25})
    for epoch in range(4):
        assert len(plan.batches(epoch)) == plan.batches_per_epoch
        for batch in plan.batches(epoch):
            sources = Counter(plan.draw(epoch, position)[0] for position in batch)
            assert sources == {"alpha": 3, "beta": 1}


def test_unstratified_epochs_keep_the_first_epoch_length():
    plan = _plan(stratified=False, micro=3)
    assert plan.draws_per_epoch % 3 == 0
    for epoch in range(3):
        batches = plan.batches(epoch)
        assert len(batches) == plan.batches_per_epoch
        assert sorted(p for batch in batches for p in batch) == list(
            range(plan.draws_per_epoch)
        )


def test_budget_filter_uses_cached_lengths_and_replaces_unfit_lazy_draws():
    seed = 7
    pools = [
        _pool("alpha", 12, 4, seed, long={cap_window(12, 4, seed, "alpha")[0]}),
        _pool("beta", 3, None, seed),
    ]
    reserve = cap_window(12, 4, seed, "alpha", epoch=1)
    builder = _Builder(long={f"alpha{reserve[0]}"})
    plan = _plan(
        pools=pools, limits=BudgetLimits(logical=16, physical=None), builder=builder
    )
    assert plan.budget_drops == {"logical": 1, "physical": 0}
    assert all(
        plan.row(0, position)["length"] <= 16
        for position in range(plan.draws_per_epoch)
    )
    rows = [plan.row(1, position) for position in range(plan.draws_per_epoch)]
    assert all(row["length"] <= 16 for row in rows)
    assert (
        "alpha",
        f"alpha{reserve[0]}",
    ) in {(source, ident) for source, ident, _ in builder.calls}
    assert f"alpha{reserve[0]}" not in {row["record"]["id"] for row in rows}
    # The replacement comes from the same source, so stratification holds.
    for batch in plan.batches(1):
        assert Counter(rows[p]["source"] for p in batch) == {"alpha": 1, "beta": 1}


def test_physical_budget_follows_micro_batch_size(monkeypatch):
    monkeypatch.setattr(
        canvas,
        "resolve_native_packing_budget",
        lambda cfg, *, packed, batch_size=None: SimpleNamespace(
            payload_capacity=cfg["sequence_len"] * batch_size
        ),
    )
    cfg = {"sequence_len": 64, "batch_flattening": True, "diffusion_lm": {}}
    small = datasets._budget_limits({**cfg, "micro_batch_size": 1})
    large = datasets._budget_limits({**cfg, "micro_batch_size": 4})
    assert small.logical == large.logical == 64
    assert (small.physical, large.physical) == (64, 256)


def test_plan_sampler_draws_each_epoch_and_follows_dataset_set_epoch():
    plan = _plan()
    dataset = datasets.DecisionPlanDataset(plan, {})
    sampler = _DecisionPlanBatchSampler(plan, world_size=2)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=list)

    def epoch_rows():
        return [[row["record"]["id"] for row in batch] for batch in loader]

    epoch0 = epoch_rows()
    assert len(epoch0) == len(sampler) and len(sampler) % 2 == 0
    sampler.set_epoch(1)
    epoch1 = epoch_rows()
    assert len(epoch1) == len(epoch0)
    assert epoch1 != epoch0
    # After a resume, accelerate reaches only the dataset (the skip wrapper hides
    # the sampler); a fresh plan moved that way yields the same epoch.
    restarted = _plan()
    stripped = datasets.DecisionPlanDataset(restarted, {}).remove_columns(["length"])
    stripped.set_epoch(1)
    resumed = DataLoader(
        stripped,
        batch_sampler=_DecisionPlanBatchSampler(restarted, world_size=2),
        collate_fn=list,
    )
    assert [[row["record"]["id"] for row in batch] for batch in resumed] == epoch1
    draws = [draw for batch in _DecisionPlanBatchSampler(restarted) for draw in batch]
    assert all(isinstance(draw, DecisionDraw) and draw.epoch == 1 for draw in draws)


def _local_cfg(tmp_path, records) -> dict[str, Any]:
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(record) + "\n" for record in records))
    cfg = _cfg(
        _entry("json", data_files=str(source)),
        mixture={
            "weights": {"alpha": 0.5, "beta": 0.5},
            "max_examples_per_source": {"alpha": 3},
            "per_batch_stratified": True,
        },
    )
    cfg.update(
        {
            "dataset_prepared_path": str(tmp_path / "prepared"),
            "model_config": {"vocab_size": 256, "turn_close_token_id": 2},
            "diffusion_lm": {"canvas_width": 128},
            "sequence_len": 4096,
            "gradient_accumulation_steps": 4,
            "eval_batch_size": 2,
        }
    )
    return cfg


@pytest.fixture
def local_loader(tmp_path, monkeypatch):
    root = tmp_path / "tokenizer"
    root.mkdir()
    (root / "tokenizer.json").write_text("tokenizer")
    tokenizer = FakeTokenizer(name_or_path=str(root), chat_template="t", vocab_size=256)
    built: list[tuple[str, Any]] = []

    def canvas_row(_tokenizer, row, _cfg, source_weight, **kwargs):
        built.append((row["id"], kwargs.get("draw")))
        return {
            "canvas": _unit_canvas(
                canvas_ids=(2,) * 128,
                label_positions=(3,),
                allowed=(4,),
                template_length=1,
            ),
            "record": row,
            "source": row["source"],
            "source_weight": source_weight,
            "length": 129,
            "draw": kwargs.get("draw"),
        }

    monkeypatch.setattr(datasets, "load_tokenizer", lambda _cfg: tokenizer)
    monkeypatch.setattr(datasets, "require_diffusion_spec", lambda _cfg: SPEC)
    monkeypatch.setattr(datasets, "_canvas_row", canvas_row)
    records = [
        _record("alpha", f"a{index}", f"state a{index}", f"a{index}")
        for index in range(9)
    ] + [
        _record("beta", f"b{index}", f"state b{index}", f"b{index}")
        for index in range(3)
    ]
    return _local_cfg(tmp_path, records), built


def test_batch_settings_reuse_the_canvas_layer(local_loader):
    cfg, built = local_loader
    first = datasets.load_decision_datasets(cfg)
    prepared = len(built)
    assert prepared == 3 + 3  # alpha's epoch-0 window and all of beta, eagerly
    assert first.train_dataset.manifest["dropped"]["reserve"] == 6
    for change in (
        {"micro_batch_size": 4},
        {"gradient_accumulation_steps": 16},
        {"eval_batch_size": 8},
        {"batch_flattening": True},
        {"sample_packing": False},
    ):
        result = datasets.load_decision_datasets({**cfg, **change})
        assert len(built) == prepared, change
        micro = change.get("micro_batch_size", cfg["micro_batch_size"])
        assert result.train_dataset.batch_plan.micro_batch_size == micro
    mixture = dict(
        cfg["diffusion_decision"]["mixture"], weights={"alpha": 0.25, "beta": 0.75}
    )
    reweighted = datasets.load_decision_datasets(
        {**cfg, "diffusion_decision": {**cfg["diffusion_decision"], "mixture": mixture}}
    )
    assert len(built) == prepared
    assert reweighted.train_dataset.manifest["mixture_probabilities"] == {
        "alpha": 0.25,
        "beta": 0.75,
    }
    # Data and seed still key the layer.
    datasets.load_decision_datasets({**cfg, "seed": cfg["seed"] + 1})
    assert len(built) > prepared


def test_cache_key_ignores_batch_settings_only():
    base = _cfg(_entry("train"))
    same = [
        {**base, "micro_batch_size": 9},
        {**base, "eval_batch_size": 3},
        {**base, "gradient_accumulation_steps": 32},
        {**base, "sample_packing": True, "eval_sample_packing": False},
        {**base, "batch_flattening": True},
    ]
    for cfg in same:
        assert prepared_cache._semantic_config(cfg) == prepared_cache._semantic_config(
            base
        )
    for cfg in ({**base, "seed": 99}, {**base, "sequence_len": 99}):
        assert prepared_cache._semantic_config(cfg) != prepared_cache._semantic_config(
            base
        )


def test_loaded_plan_draws_reserve_records_lazily_in_later_epochs(local_loader):
    cfg, built = local_loader
    result = datasets.load_decision_datasets(cfg)
    dataset = result.train_dataset
    plan = dataset.batch_plan
    epoch0 = {row["record"]["id"] for row in dataset}
    prepared = {identifier for identifier, _ in built}
    assert {ident for ident in epoch0 if ident.startswith("a")} <= prepared
    built.clear()
    epoch1 = [
        dataset[DecisionDraw(index=position, epoch=1, global_draw_ordinal=position)]
        for position in range(len(dataset))
    ]
    alpha1 = {row["record"]["id"] for row in epoch1 if row["source"] == "alpha"}
    assert alpha1 and not alpha1 & epoch0
    assert all(draw[0] == 1 for _, draw in built)
    assert len(epoch1) == plan.draws_per_epoch


def test_trainer_samples_a_plan_dataset_with_the_plan_sampler():
    from tests.integrations.diffusion_decision.test_trainer import _harness

    trainer = _harness(world_size=2)
    trainer.train_dataset = datasets.DecisionPlanDataset(_plan(micro=2), {})
    trainer.args.per_device_train_batch_size = 2
    trainer.args.sample_packing = False
    sampler = trainer._get_train_sampler()
    assert isinstance(sampler, _DecisionPlanBatchSampler)
    assert len(sampler) == len(list(sampler))
    trainer.args.per_device_train_batch_size = 4
    with pytest.raises(ValueError, match="microbatch size"):
        trainer._get_train_sampler()
