import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
from transformers import PretrainedConfig

from axolotl.integrations.diffusion_decision import (
    data_audit,
    datasets,
    prepared_cache,
)
from axolotl.integrations.diffusion_decision.args import DecisionMixtureConfig
from axolotl.integrations.diffusion_decision.data_audit import build_preparation_audit
from axolotl.integrations.diffusion_decision.loss import decision_example_from_canvas
from axolotl.integrations.diffusion_decision.records import (
    DecisionCanvas,
    OrdinalMetadata,
)
from axolotl.integrations.diffusion_decision.slot_sampling import (
    DecisionDraw,
    sample_slot_count,
)
from axolotl.integrations.diffusion_decision.slots import SlotPlan
from axolotl.model_support import DiffusionLayout

from tests.integrations.diffusion_decision.helpers import make_canvas, make_record

PREMIXED = {"premixed": True, "per_batch_stratified": False}
DEFAULT_MIXTURE = {
    "weights": {"alpha": 0.8, "beta": 0.2},
    "max_examples_per_source": {"alpha": 2, "beta": 2},
    "per_batch_stratified": True,
}
SPEC = SimpleNamespace(
    max_canvas=128,
    noise=datasets.DiffusionNoise.UNIFORM,
    layout=DiffusionLayout.FULL_SEQUENCE,
)
IDENTITY = {"identity": 1}


def _record(source: str, group: str, state: str, identifier: str) -> dict[str, Any]:
    return make_record(
        identifier, source=source, group=group, state=state, question_type="noul"
    )


def _premixed(draw_id: str, state: str = "state", **fields: Any) -> dict[str, Any]:
    record = _record("alpha", "same", state, "origin")
    record["source_metadata"] = {
        "premix": {
            "draw_id": draw_id,
            "origin": {"source": "alpha", "id": "origin", "group": "same"},
        }
    }
    record.update(fields)
    return record


def _entry(path: str, split: str | None = None, **fields: Any) -> dict[str, Any]:
    entry = {"path": path, "type": "diffusion_decision.jsonl", **fields}
    if split is not None:
        entry["split"] = split
    return entry


DEV_ENTRY = _entry("dev", "validation")


def _cfg(
    *entries: dict[str, Any],
    test_datasets: tuple[dict[str, Any], ...] = (),
    mixture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "seed": 13,
        "micro_batch_size": 2,
        "datasets": list(entries),
        "test_datasets": list(test_datasets),
        "diffusion_decision": {
            "labels": {},
            "mixture": DEFAULT_MIXTURE if mixture is None else mixture,
        },
    }


def _patch_loader(monkeypatch, rows_by_path: dict[str, list[dict[str, Any]]]) -> None:
    monkeypatch.setattr(datasets, "_rows", lambda entry: rows_by_path[entry["path"]])
    monkeypatch.setattr(
        datasets, "normalize_record", lambda _adapter, row, **_kwargs: dict(row)
    )
    monkeypatch.setattr(datasets, "load_tokenizer", lambda _cfg: range(131072))
    monkeypatch.setattr(datasets, "require_diffusion_spec", lambda _cfg: SPEC)
    monkeypatch.setattr(
        datasets,
        "_canvas_row",
        lambda _tokenizer, row, _cfg, source_weight, **_kwargs: {
            "canvas": SimpleNamespace(prompt_ids=(1,), canvas_ids=(2,) * 128),
            "record": row,
            "source": row["source"],
            "source_weight": source_weight,
        },
    )


def _unit_canvas(**fields: Any) -> DecisionCanvas:
    values: dict[str, Any] = {
        "prompt_ids": (1,),
        "canvas_ids": (2,),
        "label_positions": (0,),
        "allowed": (3,),
        "question_ids": ("q",),
    }
    return make_canvas(**{**values, **fields})


def _cache_row(identifier: str = "x") -> dict[str, Any]:
    return {"canvas": _unit_canvas(), "record": {"id": identifier}, "source": "x"}


class FakeTokenizer(SimpleNamespace):
    backend_tokenizer = SimpleNamespace(to_str=lambda: "backend")
    pad_token_id = bos_token_id = eos_token_id = unk_token_id = mask_token_id = 0
    vocab: dict[str, int] = {"x": 0}

    def __len__(self) -> int:
        return getattr(self, "vocab_size", len(self.vocab))

    def get_vocab(self) -> dict[str, int]:
        return self.vocab


@pytest.fixture
def tokenizer_root(tmp_path):
    root = tmp_path / "tokenizer"
    root.mkdir()
    (root / "tokenizer.json").write_text("tokenizer")
    return root


@pytest.fixture
def empty_source(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n")
    return source


@pytest.fixture
def two_source_cfg(monkeypatch):
    _patch_loader(
        monkeypatch,
        {
            "train": [_record("alpha", "a", "a", "a"), _record("beta", "b", "b", "b")],
            "dev": [_record("gamma", "g", "g", "g")],
        },
    )
    return _cfg(_entry("train"), test_datasets=(_entry("dev", "dev"),))


def test_loader_protects_official_test_calibration_and_ood(monkeypatch):
    _patch_loader(
        monkeypatch,
        {
            "train": [
                _record("alpha", "keep", "train-keep", "a0"),
                _record("alpha", "state-overlap", "shared", "a1"),
                _record("beta", "family-overlap", "train-family", "b0"),
                _record("beta", "keep-b", "train-b", "b1"),
            ],
            "cal": [_record("alpha", "cal", "shared", "c0")],
            "ood": [_record("beta", "family-overlap", "ood-state", "o0")],
            "heldout": [_record("alpha", "heldout", "heldout-state", "h0")],
            "dev": [_record("gamma", "dev", "dev-state", "d0")],
        },
    )
    cfg = _cfg(
        _entry("train", "train"),
        _entry("cal", "calibration"),
        _entry("ood", "ood"),
        test_datasets=(DEV_ENTRY, _entry("heldout", "test")),
    )

    result = datasets.load_decision_datasets(cfg)
    retained = [row["record"] for row in result.train_dataset]
    eval_rows = [row["record"] for row in result.eval_dataset]

    assert {row["id"] for row in retained} == {"a0", "b1"}
    assert {row["id"] for row in eval_rows} == {"d0"}
    assert result.train_dataset.manifest["protected_eval_rows"] == 3
    assert result.train_dataset.manifest["dropped"]["state_overlap"] == 1
    assert result.train_dataset.manifest["dropped"]["family_overlap"] == 1


def test_loader_realizes_each_capped_record_once_in_stratified_batches(monkeypatch):
    records = [
        _record(source, f"{tag}{index}", f"s{tag}{index}", f"{tag}{index}")
        for source, tag in (("alpha", "a"), ("beta", "b"))
        for index in range(3)
    ]
    _patch_loader(
        monkeypatch, {"train": records, "dev": [_record("gamma", "g", "sg", "g")]}
    )
    cfg = _cfg(_entry("train", "train"), test_datasets=(DEV_ENTRY,))

    result = datasets.load_decision_datasets(cfg)
    retained = [row["record"] for row in result.train_dataset]
    manifest = result.train_dataset.manifest

    assert len(retained) >= 4
    assert len({row["id"] for row in retained}) == 4
    assert {row["source"] for row in retained} == {"alpha", "beta"}
    assert manifest["dropped"]["capped"] == 2
    assert manifest["mixture_probabilities"] == {"alpha": 0.8, "beta": 0.2}
    assert manifest["stratified_epoch_batches"]
    assert all(
        len(batch) <= manifest["stratified_micro_batch_size"]
        for batch in manifest["stratified_epoch_batches"]
    )


def test_loader_preserves_premixed_draw_multiplicity_without_mixture_expansion(
    monkeypatch,
):
    _patch_loader(
        monkeypatch,
        {
            "train": [_premixed("draw-1"), _premixed("draw-2")],
            "dev": [_record("gamma", "dev", "dev-state", "dev")],
        },
    )
    cfg = _cfg(_entry("train", "train"), test_datasets=(DEV_ENTRY,), mixture=PREMIXED)

    result = datasets.load_decision_datasets(cfg)

    rows = list(result.train_dataset)
    assert len(rows) == 2
    assert [row["record"]["source_metadata"]["premix"]["draw_id"] for row in rows] == [
        "draw-1",
        "draw-2",
    ]
    assert result.train_dataset.manifest["premixed"] is True
    assert result.train_dataset.manifest["mixture_probabilities"] == {}
    assert result.train_dataset.manifest["stratified_epoch_batches"] == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weights": {"alpha": 1.0}},
        {"temperature": 1.0},
        {"max_examples_per_source": 1},
        {"per_batch_stratified": True},
    ],
)
def test_premixed_mixture_rejects_ignored_mixture_controls(kwargs):
    with pytest.raises(ValueError, match="premixed=true"):
        DecisionMixtureConfig(
            premixed=True, **{"per_batch_stratified": False, **kwargs}
        )


def test_premixed_mixture_accepts_unweighted_unstratified_rows():
    assert DecisionMixtureConfig(premixed=True, per_batch_stratified=False).premixed


@pytest.mark.parametrize(
    ("train", "dev", "match"),
    [
        pytest.param(
            [
                _premixed("draw-1"),
                _premixed("draw-2", labels={"q": {"kind": "hard", "gold_idx": 1}}),
            ],
            None,
            "identical content",
            id="changed-content-for-one-origin",
        ),
        pytest.param(
            [_premixed("draw-1"), _premixed("draw-1")],
            None,
            "draw_id is duplicated",
            id="duplicate-draw-id",
        ),
        pytest.param(
            [_premixed("draw-1", state="shared")],
            [_record("gamma", "dev", "shared", "dev")],
            "overlap an evaluation split",
            id="decontamination-drop",
        ),
    ],
)
def test_premixed_rejects_invalid_draws(monkeypatch, train, dev, match):
    _patch_loader(monkeypatch, {"train": train, "dev": dev or []})
    cfg = _cfg(
        _entry("train", "train"),
        test_datasets=(DEV_ENTRY,) if dev else (),
        mixture=PREMIXED,
    )

    with pytest.raises(ValueError, match=match):
        datasets.load_decision_datasets(cfg)


def test_premixed_toggle_changes_preparation_audit_config():
    regular = _cfg(_entry("train"))
    premixed = _cfg(_entry("train"), mixture=PREMIXED)

    assert (
        build_preparation_audit(regular, [], [], {})["config"]
        != build_preparation_audit(premixed, [], [], {})["config"]
    )


def test_family_split_and_protected_split_names():
    rows = [
        _record("a", "x", "one", "a"),
        _record("a", "y", "two", "b"),
        _record("b", "g", "three", "c"),
    ]
    train, dev = datasets._family_dev_split(rows, ratio=0.5)
    assert train and dev
    assert {datasets.family_key(row) for row in train}.isdisjoint(
        {datasets.family_key(row) for row in dev}
    )
    assert all(
        datasets._is_eval({"split": split}) for split in ("test", "calibration", "ood")
    )
    assert not datasets._is_eval({"split": "train"})


def test_local_jsonl_preserves_heterogeneous_nested_records_and_glob_order(tmp_path):
    (tmp_path / "01.jsonl").write_text(
        '{"state":{"scenario":"one","items":[1]},"options":["a"]}\n'
    )
    (tmp_path / "02.jsonl").write_text(
        '{"state":{"scenario":"two","items":[{"x":2}]},"options":{"yes":1}}\n'
    )

    rows = datasets._rows(
        {"path": "json", "split": "train", "data_files": str(tmp_path / "*.jsonl")}
    )

    assert [row["state"]["scenario"] for row in rows] == ["one", "two"]
    assert isinstance(rows[0]["options"], list)
    assert isinstance(rows[1]["options"], dict)


def test_local_jsonl_reports_path_and_line_for_invalid_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok":true}\n{"broken":\n')

    with pytest.raises(ValueError, match=r"invalid JSONL at .*bad\.jsonl:2:"):
        datasets._rows({"path": "json", "split": "train", "data_files": str(path)})


@pytest.mark.parametrize("name", ["https://example.org/data.jsonl", "local.json"])
def test_nonlocal_or_non_jsonl_inputs_use_hf_loader(monkeypatch, tmp_path, name):
    if name == "local.json":
        path = tmp_path / name
        path.write_text('[{"value":1}]')
        name = str(path)
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return [{"value": 1}]

    monkeypatch.setattr(datasets, "load_dataset", load)
    assert datasets._rows({"path": "json", "data_files": name, "split": "dev"}) == [
        {"value": 1}
    ]
    assert calls == [("json", {"split": "dev", "data_files": {"dev": name}})]


def test_canvas_named_invalid_question_is_not_silently_dropped(monkeypatch):
    def fail(*args, **kwargs):
        raise datasets.SchemaError(
            "question 'canvas_choice': labels do not share one template slot"
        )

    monkeypatch.setattr(datasets, "_canvas_row", fail)
    with pytest.raises(datasets.SchemaError, match="canvas_choice"):
        datasets._canvas_rows(range(131072), [{"source": "alpha"}], {}, {}, None)
    assert datasets._is_canvas_overflow(
        datasets.SchemaError("answer template is 140 tokens; the canvas holds 127")
    )


def test_parallel_canvas_rows_preserves_source_order_and_overflow_accounting(
    monkeypatch,
):
    class InlineExecutor:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, fn, values, **_kwargs):
            return [fn(value) for value in reversed(list(values))]

    def canvas_row(_tokenizer, row, _cfg, source_weight, **_kwargs):
        if row["id"] == "overflow":
            raise datasets.SchemaError(
                "answer template is 140 tokens; the canvas holds 127"
            )
        return {"id": row["id"], "source_weight": source_weight}

    records = [
        {"id": "first", "source": "alpha"},
        {"id": "overflow", "source": "alpha"},
        {"id": "last", "source": "beta"},
    ]
    monkeypatch.setattr(datasets, "_canvas_row", canvas_row)
    monkeypatch.setattr(datasets, "ThreadPoolExecutor", InlineExecutor)
    monkeypatch.setattr(datasets, "_resolve_slot_plan", lambda *_args: (None, (), ()))
    spec = SimpleNamespace(max_canvas=None)
    serial, serial_drops = datasets._canvas_rows(
        range(100), records, {"dataset_num_proc": 1}, {"alpha": 2.0}, spec
    )
    parallel, parallel_drops = datasets._canvas_rows(
        range(100), records, {"dataset_num_proc": 2}, {"alpha": 2.0}, spec
    )

    assert parallel == serial
    assert parallel_drops == serial_drops == 1


def test_budget_filter_precedes_source_probabilities_and_sampling(
    monkeypatch, two_source_cfg
):
    def canvas_row(_tokenizer, row, _cfg, source_weight, **kwargs):
        return {
            "canvas": SimpleNamespace(
                prompt_ids=(1,) * (50 if row["source"] == "alpha" else 1),
                canvas_ids=(2,) * 128,
            ),
            "record": row,
            "source": row["source"],
        }

    monkeypatch.setattr(datasets, "_canvas_row", canvas_row)
    two_source_cfg["sequence_len"] = 150
    result = datasets.load_decision_datasets(two_source_cfg)
    manifest = result.train_dataset.manifest
    assert manifest["budget_drops"]["train"] == {"logical": 1, "physical": 0}
    assert manifest["mixture_probabilities"] == {"beta": 1.0}
    assert all(row["source"] == "beta" for row in result.train_dataset)
    assert sum(map(len, manifest["stratified_epoch_batches"])) == len(
        result.train_dataset
    )


def test_budget_filter_uses_resolved_payload_capacity(monkeypatch):
    calls = []

    def resolve(cfg, *, packed, batch_size=None):
        calls.append(packed)
        return SimpleNamespace(payload_capacity=128)

    monkeypatch.setattr(datasets, "resolve_native_packing_budget", resolve)
    row = {"canvas": SimpleNamespace(prompt_ids=(1,), canvas_ids=(2,) * 128)}
    cfg = {
        "sample_packing": False,
        "batch_flattening": True,
        "sequence_len": 2048,
        "micro_batch_size": 1,
        "diffusion_lm": {},
    }
    rows, drops = datasets._filter_budget_rows([row], cfg, SPEC)
    assert not rows
    assert drops == {"logical": 0, "physical": 1}
    assert calls == [True]


def test_loader_reuses_supplied_tokenizer(monkeypatch, two_source_cfg):
    def unexpected(_cfg):
        raise AssertionError("must reuse the supplied tokenizer")

    monkeypatch.setattr(datasets, "load_tokenizer", unexpected)
    result = datasets.load_decision_datasets(two_source_cfg, tokenizer=range(131072))
    assert len(result.train_dataset) > 0


def test_local_prepared_cache_reuses_typed_rows_in_exact_order(
    tmp_path, tokenizer_root, monkeypatch
):
    source = tmp_path / "source.jsonl"

    def write_source(state: str) -> None:
        source.write_text(json.dumps(_record("alpha", "a", state, "id")) + "\n")

    write_source("state")
    tokenizer = FakeTokenizer(
        name_or_path=str(tokenizer_root), chat_template="base", vocab_size=256
    )
    cfg = _cfg(
        _entry("json", data_files=str(source)),
        mixture={
            "weights": {"alpha": 1.0},
            "max_examples_per_source": {"alpha": 1},
            "per_batch_stratified": True,
        },
    )
    cfg.update(
        {
            "dataset_prepared_path": str(tmp_path / "prepared"),
            "model_config": {"vocab_size": 256, "turn_close_token_id": 2},
            "diffusion_lm": {"canvas_width": 128},
        }
    )
    audit_path = tmp_path / "prepared" / "diffusion_decision_preparation_audit.json"
    calls = 0
    original_rows = datasets._rows

    def rows(entry):
        nonlocal calls
        calls += 1
        return original_rows(entry)

    monkeypatch.setattr(datasets, "_rows", rows)
    monkeypatch.setattr(datasets, "load_tokenizer", lambda _cfg: tokenizer)
    monkeypatch.setattr(datasets, "require_diffusion_spec", lambda _cfg: SPEC)

    def canvas_row(_tokenizer, row, _cfg, source_weight, **_kwargs):
        canvas = _unit_canvas(
            canvas_ids=(2,) * 128, label_positions=(3,), allowed=(4,), template_length=1
        )
        return {
            "canvas": canvas,
            "decision_example": decision_example_from_canvas(
                canvas, source_weight=source_weight
            ),
            "record": row,
            "source": row["source"],
            "source_weight": source_weight,
            "length": 129,
        }

    monkeypatch.setattr(datasets, "_canvas_row", canvas_row)

    first = datasets.load_decision_datasets(cfg)
    first_ids = [row["record"]["id"] for row in first.train_dataset]
    first_rows = [dict(row) for row in first.train_dataset]
    first_schedule = first.train_dataset.manifest["stratified_epoch_batches"]
    first_calls = calls
    cold_audit = json.loads(audit_path.read_text())
    cfg["seed"] = 14
    datasets.load_decision_datasets(cfg)
    assert calls > first_calls
    second_cold_calls = calls
    cfg["seed"] = 13
    audit_builds = {"cache_rows": 0, "sidecar": 0}
    cache_build = prepared_cache.build_preparation_audit
    sidecar_build = data_audit.build_preparation_audit

    def count_cache_build(_cfg, train_rows, eval_rows, manifest, **kwargs):
        if train_rows or eval_rows:
            audit_builds["cache_rows"] += 1
        return cache_build(_cfg, train_rows, eval_rows, manifest, **kwargs)

    def count_sidecar_build(*args, **kwargs):
        audit_builds["sidecar"] += 1
        return sidecar_build(*args, **kwargs)

    monkeypatch.setattr(prepared_cache, "build_preparation_audit", count_cache_build)
    monkeypatch.setattr(data_audit, "build_preparation_audit", count_sidecar_build)
    second = datasets.load_decision_datasets(cfg)
    assert calls == second_cold_calls
    assert audit_builds == {"cache_rows": 1, "sidecar": 0}
    assert [row["record"]["id"] for row in second.train_dataset] == first_ids
    assert [dict(row) for row in second.train_dataset] == first_rows
    assert second.train_dataset.manifest["stratified_epoch_batches"] == first_schedule
    assert isinstance(second.train_dataset[0]["canvas"], DecisionCanvas)
    audit = json.loads(audit_path.read_text())
    assert audit == cold_audit
    assert audit["config"]["seed"] == 13
    assert (
        audit["splits"]["train"]["full_canvas_sha256"]
        == prepared_cache.build_preparation_audit(
            cfg, first_rows, [], first.train_dataset.manifest
        )["splits"]["train"]["full_canvas_sha256"]
    )

    def requires_rebuild(mutate, restore):
        nonlocal calls
        before = calls
        mutate()
        datasets.load_decision_datasets(cfg)
        assert calls > before
        restore()
        before = calls
        datasets.load_decision_datasets(cfg)
        assert calls == before

    requires_rebuild(lambda: write_source("changed"), lambda: write_source("state"))
    requires_rebuild(
        lambda: setattr(tokenizer, "chat_template", "changed"),
        lambda: setattr(tokenizer, "chat_template", "base"),
    )
    requires_rebuild(
        lambda: cfg["diffusion_lm"].update(canvas_width=129),
        lambda: cfg["diffusion_lm"].update(canvas_width=128),
    )
    requires_rebuild(
        lambda: cfg.update(micro_batch_size=3), lambda: cfg.update(micro_batch_size=2)
    )


def test_prepared_cache_preserves_nested_typed_values_and_live_vocab_identity(
    tokenizer_root, empty_source
):
    first, second = (
        prepared_cache.identity(
            {},
            FakeTokenizer(name_or_path=str(tokenizer_root), vocab=vocab),
            SPEC,
            [empty_source],
        )
        for vocab in ({"a": 1}, {"b": 1})
    )
    assert first is not None and second is not None
    assert first[0] != second[0]
    canvas = _unit_canvas(
        slot_mask=(True,),
        ordinal_metadata=(OrdinalMetadata(("low",), ("0",), (0,)),),
    )
    row = {
        "canvas": canvas,
        "decision_example": decision_example_from_canvas(canvas, source_weight=0.25),
        "record": {"id": "x", "grouped_record_ids": ("x", "y")},
        "source": "x",
        "slot_plan": SlotPlan((9,), "thought", (True,), (False,), (False,), (9,)),
    }
    restored = prepared_cache._row_from_json(
        json.loads(json.dumps(prepared_cache._row_to_json(row)))
    )
    assert restored["canvas"] == canvas
    assert restored["slot_plan"] == row["slot_plan"]
    assert restored["decision_example"] == row["decision_example"]
    assert restored["record"] == row["record"]


def test_prepared_cache_uses_exact_cached_hub_tokenizer_snapshot(
    tmp_path, empty_source, monkeypatch
):
    import huggingface_hub

    cache = tmp_path / "hub"
    repo_id = "nvidia/Nemotron-Labs-Diffusion-8B"
    revisions = ("a" * 40, "b" * 40)
    for revision in revisions:
        snapshot = (
            cache
            / "models--nvidia--Nemotron-Labs-Diffusion-8B"
            / "snapshots"
            / revision
        )
        snapshot.mkdir(parents=True)
        (snapshot / "tokenizer_config.json").write_text("tokenizer")
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
    calls = []
    hf_hub_download = prepared_cache.hf_hub_download

    def local_tokenizer_file(*args, **kwargs):
        calls.append((args, kwargs))
        return hf_hub_download(*args, **kwargs)

    monkeypatch.setattr(prepared_cache, "hf_hub_download", local_tokenizer_file)

    def identity(cfg):
        return prepared_cache.identity(
            cfg, FakeTokenizer(name_or_path=repo_id), SPEC, [empty_source]
        )

    first = identity({"model_config": {"_commit_hash": revisions[0]}})
    second = identity({"model_config": {"_commit_hash": revisions[1]}})
    from_immutable_request = identity({"revision_of_model": revisions[0]})
    from_moving_request = identity({"revision_of_model": "main"})

    assert (
        first is not None and second is not None and from_immutable_request is not None
    )
    assert first[0] != second[0]
    assert from_moving_request is None
    assert first[1]["tokenizer"]["hub"] == {
        "repo_id": repo_id,
        "resolved_revision": revisions[0],
    }
    assert [kwargs["local_files_only"] for _, kwargs in calls] == [True, True, True]
    assert [kwargs["revision"] for _, kwargs in calls] == [*revisions, revisions[0]]
    assert [kwargs["filename"] for _, kwargs in calls] == ["tokenizer_config.json"] * 3
    validated_shape = identity(
        {
            "model_config": None,
            "overrides_of_model_config": {"_commit_hash": revisions[0]},
        }
    )
    assert validated_shape is not None
    assert validated_shape[1]["tokenizer"]["hub"]["resolved_revision"] == revisions[0]


def test_prepared_cache_identity_binds_entry_roles_and_model_controls(
    tokenizer_root, empty_source
):
    base = {
        "datasets": [_entry("json", "train", data_files=str(empty_source))],
        "test_datasets": [],
        "model_config": {"vocab_size": 8, "turn_close_token_id": 2},
    }
    role = {
        **base,
        "datasets": [],
        "test_datasets": [{**base["datasets"][0], "split": "dev"}],
    }
    controls = {**base, "model_config": {"vocab_size": 8, "turn_close_token_id": 3}}
    keys = [
        prepared_cache.identity(
            cfg, FakeTokenizer(name_or_path=str(tokenizer_root)), SPEC, [empty_source]
        )[0]
        for cfg in (base, role, controls)
    ]
    assert len(set(keys)) == 3


def test_prepared_cache_repairs_corrupt_payload_and_warms(tmp_path):
    row = _cache_row()
    prepared_cache.store(tmp_path, "key", IDENTITY, {}, [row], [], {})
    path = tmp_path / "diffusion_decision_cache" / "key.json"
    payload = json.loads(path.read_text())
    payload["content"]["manifest"] = {"bad": True}
    path.write_text(json.dumps(payload))
    assert prepared_cache.load(tmp_path, "key", IDENTITY) is None
    prepared_cache.store(tmp_path, "key", IDENTITY, {}, [row], [], {})
    first = prepared_cache.load(tmp_path, "key", IDENTITY)
    second = prepared_cache.load(tmp_path, "key", IDENTITY)
    assert first is not None and second is not None
    assert first[0][0]["canvas"] == row["canvas"]


def test_prepared_cache_serializes_pretrained_model_config(tmp_path):
    cfg = {"model_config": PretrainedConfig(vocab_size=8, pad_token_id=0)}
    identity = {"model": cfg["model_config"]}
    prepared_cache.store(tmp_path, "config", identity, cfg, [_cache_row()], [], {})
    assert prepared_cache.load(tmp_path, "config", identity) is not None


def test_prepared_cache_atomic_same_key_writers_and_readers(tmp_path):
    prepared_cache.store(tmp_path, "shared", IDENTITY, {}, [_cache_row("old")], [], {})
    barrier = threading.Barrier(3)

    def writer(identifier):
        barrier.wait()
        prepared_cache.store(
            tmp_path, "shared", IDENTITY, {}, [_cache_row(identifier)], [], {}
        )

    def reader():
        barrier.wait()
        seen = []
        for _ in range(30):
            loaded = prepared_cache.load(tmp_path, "shared", IDENTITY)
            assert loaded is not None
            seen.append(loaded[0][0]["record"]["id"])
        return seen

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(writer, "a"),
            pool.submit(writer, "b"),
            pool.submit(reader),
        ]
        seen = futures[2].result()
        futures[0].result()
        futures[1].result()
    assert set(seen) <= {"old", "a", "b"}
    assert prepared_cache.load(tmp_path, "shared", IDENTITY) is not None


def test_decision_draw_projects_maximum_row_and_preserves_weight():
    canvas = make_canvas(
        prompt_ids=(1,),
        canvas_ids=(7, 8, 9, 4, 0, 0),
        label_positions=(3,),
        allowed=(4,),
        question_ids=("q",),
        pinned_mask=(True, True, True, False, True, True),
        slot_mask=(True, True, True, False, False, False),
        template_length=4,
    )
    plan = SlotPlan((7, 8, 9), "thought", (True,) * 3, (False,) * 3, (False,) * 3, ())
    row = {
        "canvas": canvas,
        "slot_plan": plan,
        "slot_sampling": {
            "seed": 1,
            "max_slots": 3,
            "pad_token_id": 0,
            "padding_pinned": True,
        },
        "decision_example": decision_example_from_canvas(canvas, source_weight=0.25),
        "source": "x",
    }
    dataset = datasets.DecisionDataset([row], {})
    seen = {}
    for ordinal in range(100):
        count = sample_slot_count(
            seed=1, epoch=0, global_draw_ordinal=ordinal, max_slots=3
        )
        seen.setdefault(count, dataset[DecisionDraw(0, 0, ordinal)])
    assert set(seen) == {0, 1, 2, 3}
    for count, projected in seen.items():
        assert projected["decision_slot_count"] == count
        assert len(projected["slot_plan"].ids) == count
        assert projected["decision_example"].source_weight == 0.25
    assert dataset[0] is row
