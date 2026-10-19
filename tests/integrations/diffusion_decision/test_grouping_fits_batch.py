"""Batched grouping width checks (``template_fits_many``, ``_GroupingFits.many``) equal the per-canvas check."""

from __future__ import annotations

import copy
import gc
import json

import pytest

from axolotl.integrations.diffusion_decision import canvas, datasets
from axolotl.integrations.diffusion_decision.grouping import group_records
from axolotl.integrations.diffusion_decision.preprocessing import (
    template_fits,
    template_fits_many,
)

from .helpers import ChatCharacterTokenizer, make_cfg, make_spec

NEMOTRON = ("nvidia/Nemotron-Labs-Diffusion-3B", "nvidia/Nemotron-Labs-Diffusion-8B")


def _nemotron():
    transformers = pytest.importorskip("transformers")
    for name in NEMOTRON:
        try:
            return transformers.AutoTokenizer.from_pretrained(
                name, local_files_only=True, trust_remote_code=True
            )
        except Exception:  # noqa: BLE001 - not cached
            continue
    pytest.skip("no cached Nemotron-Labs-Diffusion tokenizer")


def _choice(options=3):
    return {"type": "choice", "options": [f"o{i}" for i in range(options)]}


def _questions(ids):
    return {qid: _choice() for qid in ids}


def _record(rid, ids, *, group=None, state=None):
    return {
        "id": rid,
        "source": "fixture",
        "group": group or f"g-{rid}",
        "state": state or f"state-{rid}",
        "questions": _questions(ids),
        "labels": {qid: {"kind": "hard", "gold_idx": 0} for qid in ids},
    }


def _text_length(questions):
    # Character tokenizer, lines format: "id: A" joined by newlines.
    return sum(len(qid) + 3 for qid in questions) + len(questions) - 1


class _Plain:
    """``fits`` without ``many``: the per-canvas path group_records always had."""

    def __init__(self, fits):
        self.fits = fits
        self.calls = 0

    def __call__(self, questions):
        self.calls += 1
        return self.fits(questions)


class _FastCharacterTokenizer(ChatCharacterTokenizer):
    """A character tokenizer that claims to be fast: ``many`` encodes through ``__call__``."""

    is_fast = True

    def __init__(self):
        self.batches = []

    def __call__(self, texts, add_special_tokens=True):
        assert not add_special_tokens
        self.batches.append(list(texts))
        return {"input_ids": [self.encode(text) for text in texts]}


# --- template_fits_many == template_fits ------------------------------------------------------------------------


@pytest.mark.parametrize("head,pad_slots", [((), 0), ((5, 6, 7), 0), ((5,), 2)])
def test_many_matches_template_fits_at_the_width_boundary(head, pad_slots):
    tokenizer = ChatCharacterTokenizer()
    batch = [_questions([f"q{i}" for i in range(n)]) for n in range(1, 12)]
    lengths = [_text_length(q) for q in batch]
    # Widths on both sides of every canvas's exact limit: length + 1 + slots + head.
    for width in sorted(
        {
            length + 1 + pad_slots + len(head) + d
            for length in lengths
            for d in (-1, 0, 1)
        }
    ):
        kwargs = dict(
            head=head, pad_slots=pad_slots, width=width, answer_format="lines"
        )
        expected = [template_fits(tokenizer, q, **kwargs) for q in batch]
        assert template_fits_many(tokenizer, batch, **kwargs) == expected
        # The exact boundary: a canvas whose template needs width - 1 rows fits.
        for questions, fit, length in zip(batch, expected, lengths, strict=True):
            assert fit == (length + 1 + pad_slots + len(head) <= width), questions


@pytest.mark.parametrize("answer_format", ["lines", "indexed"])
def test_many_matches_template_fits_with_a_fast_tokenizer(answer_format):
    tokenizer = _nemotron()
    batch = []
    for n in range(1, 33):
        ids = [("noul" if i % 3 == 0 else "choice") + f"_{i}" for i in range(n)]
        batch.append(
            {
                qid: {"type": "noul"} if qid.startswith("noul") else _choice(4)
                for qid in ids
            }
        )
    for width in (16, 48, 64, 96, 128, 160):
        kwargs = dict(
            head=(),
            pad_slots=0,
            width=width,
            answer_format=answer_format,
            codebook="expanded128",
        )
        expected = [template_fits(tokenizer, q, **kwargs) for q in batch]
        assert template_fits_many(tokenizer, batch, chunk_size=7, **kwargs) == expected


def test_equal_templates_are_encoded_once_per_chunk():
    tokenizer = _FastCharacterTokenizer()
    same = _questions(["a", "b"])
    batch = [same, copy.deepcopy(same), _questions(["c"]), copy.deepcopy(same)]
    result = template_fits_many(
        tokenizer, batch, head=(), pad_slots=0, width=10, answer_format="lines"
    )
    assert result == [
        template_fits(
            tokenizer, q, head=(), pad_slots=0, width=10, answer_format="lines"
        )
        for q in batch
    ]
    assert tokenizer.batches == [["a: A\nb: A", "c: A"]]


def test_many_leaves_none_where_template_fits_raises():
    tokenizer = ChatCharacterTokenizer()
    bad = [
        {"q": {"type": "mystery"}},  # SchemaError: unknown type
        {"a:b": _choice()},  # SchemaError: ':' in the id
        {"s": {"type": "span"}},  # no labels: answer_text fails
        {"q": {"type": "choice", "options": ["only"]}},  # one alternative
    ]
    good = _questions(["x", "y"])
    batch = [good, *bad, good]
    kwargs = dict(head=(), pad_slots=0, width=64, answer_format="lines")
    assert template_fits_many(tokenizer, batch, **kwargs) == [True, *[None] * 4, True]
    for questions in bad:
        with pytest.raises(Exception):  # noqa: B017 - each raises its own error
            template_fits(tokenizer, questions, **kwargs)


def test_many_defers_a_bad_answer_format_to_template_fits():
    batch = [_questions(["x"])]
    kwargs = dict(head=(), pad_slots=0, width=64, answer_format="auto")
    assert template_fits_many(ChatCharacterTokenizer(), batch, **kwargs) == [None]
    with pytest.raises(ValueError, match="explicit lines or indexed"):
        template_fits(ChatCharacterTokenizer(), batch[0], **kwargs)


def test_an_encoder_failure_falls_back_to_the_per_canvas_check():
    class Broken(_FastCharacterTokenizer):
        def __call__(self, texts, add_special_tokens=True):
            raise RuntimeError("batch encoder down")

    result = template_fits_many(
        Broken(),
        [_questions(["x"])],
        head=(),
        pad_slots=0,
        width=64,
        answer_format="lines",
    )
    assert result == [None]


# --- group_records with and without many --------------------------------------------------------------------------


def _lines_fits(width, max_questions=32, **cfg):
    return datasets._grouping_fits(
        ChatCharacterTokenizer(),
        make_cfg(
            diffusion_lm={"canvas_width": width, "mask_token_id": 9},
            diffusion_decision={
                "max_questions_per_canvas": max_questions,
                "answer_format": "lines",
            },
            **cfg,
        ),
        make_spec(),
    )


def _boundary_records(width, max_questions):
    records = []
    # Lone records on both sides of the width limit and of max_questions.
    for n in range(1, max_questions + 3):
        ids = [f"q{i}" for i in range(n)]
        records.append(_record(f"solo{n}", ids))
        records.append(_record(f"wide{n}", [f"question_{i}" for i in range(n)]))
    # Identical templates under different states (one encode serves them all).
    records += [_record(f"twin{i}", ["a", "b", "c"]) for i in range(5)]
    # A merged group that packs to width with repeated question ids.
    records += [
        _record(f"m{i}", ["a", "b", "c", "d"], group="merged", state="shared")
        for i in range(9)
    ]
    # A lone record over max_questions that also overflows the width.
    records.append(_record("big", [f"long_question_{i}" for i in range(40)]))
    return records


@pytest.mark.parametrize("width,max_questions", [(24, 4), (40, 6), (64, 8), (128, 32)])
def test_grouping_with_many_equals_per_canvas_grouping(width, max_questions):
    fits = _lines_fits(width, max_questions)
    plain = _Plain(fits)
    records = _boundary_records(width, max_questions)
    expected = group_records(
        copy.deepcopy(records), max_questions=max_questions, fits=plain
    )
    got = group_records(copy.deepcopy(records), max_questions=max_questions, fits=fits)
    assert got == expected
    assert json.dumps(got) == json.dumps(expected)
    assert [
        type(r["grouped_record_ids"]) for r in got if "grouped_record_ids" in r
    ] == [type(r["grouped_record_ids"]) for r in expected if "grouped_record_ids" in r]
    # Some lone records fit whole and some do not, at this width and count.
    lone = [r for r in got if "grouped_record_ids" not in r]
    assert lone and len(lone) < len(records)


def test_lone_records_are_measured_by_many_not_one_by_one(monkeypatch):
    fits = _lines_fits(64, 8)
    calls = []
    original = canvas._GroupingFits.__call__
    monkeypatch.setattr(
        canvas._GroupingFits,
        "__call__",
        lambda self, questions: (
            calls.append(list(questions)) or original(self, questions)
        ),
    )
    records = [
        _record(f"solo{i}", [f"q{j}" for j in range(1 + i % 5)]) for i in range(20)
    ]
    grouped = group_records(records, max_questions=8, fits=fits)
    assert len(grouped) == 20 and calls == []


@pytest.mark.parametrize(
    "first,second,message",
    [
        # The earlier record's validation error wins over a later record's schema error.
        ("labels", "schema", "question and label ids differ"),
        # A schema error raises from fits at that record, as before.
        ("schema", "labels", "unknown type"),
    ],
)
def test_errors_raise_in_input_order(first, second, message):
    def broken(kind, rid):
        record = _record(rid, ["a", "b"])
        if kind == "labels":
            record["labels"] = {"a": {"kind": "hard", "gold_idx": 0}}
        else:
            record["questions"]["b"] = {"type": "mystery"}
        return record

    records = [_record("ok", ["x"]), broken(first, "one"), broken(second, "two")]
    fits = _lines_fits(64, 8)
    with pytest.raises(ValueError, match=message) as plain_error:
        group_records(copy.deepcopy(records), max_questions=8, fits=_Plain(fits))
    with pytest.raises(ValueError, match=message) as many_error:
        group_records(copy.deepcopy(records), max_questions=8, fits=fits)
    assert type(plain_error.value) is type(many_error.value)
    assert str(plain_error.value) == str(many_error.value)


def test_merged_groups_are_packed_from_the_plan_not_one_by_one(monkeypatch):
    fits = _lines_fits(32, 6)
    records = [
        _record(f"m{i}", ["a", "b", "c"], group=f"g{i % 3}", state="s")
        for i in range(12)
    ]
    expected = group_records(copy.deepcopy(records), max_questions=6, fits=_Plain(fits))
    calls = []
    original = canvas._GroupingFits.__call__
    monkeypatch.setattr(
        canvas._GroupingFits,
        "__call__",
        lambda self, questions: (
            calls.append(list(questions)) or original(self, questions)
        ),
    )
    got = group_records(copy.deepcopy(records), max_questions=6, fits=fits)
    assert got == expected
    # The width (32 rows) closes canvases before max_questions (6) here.
    assert [len(r["questions"]) for r in got] == [5, 5, 2] * 3
    assert calls == []


def test_a_bad_question_mid_group_raises_where_the_loop_reaches_it():
    records = [_record(f"m{i}", ["a", "b"], group="g", state="s") for i in range(4)]
    records[2]["questions"]["b"] = {"type": "mystery"}
    fits = _lines_fits(128, 32)
    with pytest.raises(ValueError, match="unknown type") as plain_error:
        group_records(copy.deepcopy(records), max_questions=32, fits=_Plain(fits))
    with pytest.raises(ValueError, match="unknown type") as many_error:
        group_records(copy.deepcopy(records), max_questions=32, fits=fits)
    assert str(plain_error.value) == str(many_error.value)


@pytest.mark.parametrize("seed", range(12))
def test_randomized_grouping_with_many_equals_per_canvas_grouping(seed):
    import random

    rng = random.Random(seed)
    width = rng.choice([16, 24, 32, 48, 64, 96])
    max_questions = rng.choice([1, 2, 3, 5, 8, 32])
    records = []
    for index in range(rng.randint(5, 40)):
        # Few ids, so repeated ids get __2, __3 suffixes inside a canvas.
        ids = rng.sample(
            ["a", "b", "c", "dd", "eee", "ffff", "q_long_id"], rng.randint(1, 5)
        )
        group = rng.choice(["x", "y", "z", f"own{index}"])
        records.append(_record(f"r{index}", ids, group=group, state=f"s-{group}"))
    fits = _lines_fits(width, max_questions)
    expected = group_records(
        copy.deepcopy(records), max_questions=max_questions, fits=_Plain(fits)
    )
    got = group_records(copy.deepcopy(records), max_questions=max_questions, fits=fits)
    assert got == expected
    assert json.dumps(got) == json.dumps(expected)


def test_many_must_return_one_result_per_canvas():
    class Short(_Plain):
        def many(self, batch):
            return []

    with pytest.raises(ValueError, match="one result per canvas"):
        group_records(
            [_record("r", ["a"])], max_questions=4, fits=Short(_lines_fits(64))
        )


# --- worker processes ----------------------------------------------------------------------------------------------


def _many_batch():
    return [
        _questions(
            [f"q{i}" for i in range(1 + n % 9)] + (["bad:id"] if n % 13 == 0 else [])
        )
        for n in range(200)
    ]


def _small_pool_batches(monkeypatch):
    monkeypatch.setattr(canvas, "_FITS_CHUNK_SIZE", 16)
    monkeypatch.setattr(canvas, "_FITS_MIN_BATCH", 64)


def test_worker_pool_returns_the_serial_results_in_order(monkeypatch):
    _small_pool_batches(monkeypatch)
    pools = []

    class Spy(canvas.ProcessPoolExecutor):
        def __init__(self, *args, **kwargs):
            pools.append(kwargs["max_workers"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(canvas, "ProcessPoolExecutor", Spy)
    fits = _lines_fits(28, dataset_num_proc=3)
    assert fits.workers == 3
    batch = _many_batch()
    serial = fits.many_serial(batch)
    assert None in serial and True in serial and False in serial
    assert fits.many(batch) == serial
    assert pools == [3]
    assert canvas._FITS_WORKER_STATE is None


def test_a_failed_worker_pool_measures_serially(monkeypatch):
    _small_pool_batches(monkeypatch)

    def crash(bounds):
        raise RuntimeError("worker lost")

    monkeypatch.setattr(canvas, "_fits_range", crash)
    fits = _lines_fits(28, dataset_num_proc=2)
    batch = _many_batch()
    assert fits.many(batch) == fits.many_serial(batch)
    assert canvas._FITS_WORKER_STATE is None


@pytest.mark.parametrize("min_batch,workers", [(201, 8), (64, 1)])
def test_small_batches_and_one_worker_stay_in_process(monkeypatch, min_batch, workers):
    monkeypatch.setattr(canvas, "_FITS_CHUNK_SIZE", 16)
    monkeypatch.setattr(canvas, "_FITS_MIN_BATCH", min_batch)

    def unexpected(*args, **kwargs):
        raise AssertionError("no pool expected")

    monkeypatch.setattr(canvas, "ProcessPoolExecutor", unexpected)
    batch = _many_batch()
    fits = _lines_fits(28, dataset_num_proc=workers)
    assert fits.many(batch) == fits.many_serial(batch)


# --- garbage collector ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("enabled", [True, False])
def test_grouping_restores_the_collector_state(enabled):
    was_enabled = gc.isenabled()
    seen = []

    def fits(questions):
        seen.append(gc.isenabled())
        return True

    try:
        gc.enable() if enabled else gc.disable()
        group_records([_record("a", ["x"])], max_questions=4, fits=fits)
        assert gc.isenabled() is enabled
        with pytest.raises(ValueError, match="positive"):
            group_records([_record("a", ["x"])], max_questions=0, fits=fits)
        assert gc.isenabled() is enabled
        broken = _record("b", ["x"])
        broken["labels"] = {}
        with pytest.raises(ValueError, match="label ids differ"):
            group_records([broken], max_questions=4, fits=fits)
        assert gc.isenabled() is enabled
    finally:
        gc.enable() if was_enabled else gc.disable()
    assert seen == [False]  # grouping ran with the collector off
