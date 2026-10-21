"""Pre-tokenized canvas segments: assembled draws equal full re-tokenization."""

from __future__ import annotations

import io
import pickle  # nosec B403 - round-trips our own cache objects
from copy import deepcopy
from types import SimpleNamespace

import pytest

from axolotl.integrations.diffusion_decision import canvas, datasets, segments
from axolotl.integrations.diffusion_decision.batch_plan import (
    BudgetLimits,
    DecisionBatchPlan,
    SourcePool,
    encode_reserve,
)
from axolotl.integrations.diffusion_decision.mixture import cap_window
from axolotl.integrations.diffusion_decision.prepared_cache import _CacheUnpickler
from axolotl.integrations.diffusion_decision.template import SchemaError
from axolotl.model_support import DiffusionLayout, DiffusionNoise

from .helpers import ChatCharacterTokenizer, make_cfg, make_record, make_spec

NEMOTRON = ("nvidia/Nemotron-Labs-Diffusion-8B", "nvidia/Nemotron-Labs-Diffusion-3B")
DRAWS = [None, (0, 1), (1, 0), (1, 2), (2, 5), (7, 0)]


@pytest.fixture(scope="module")
def nemotron():
    transformers = pytest.importorskip("transformers")
    for name in NEMOTRON:
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                name, local_files_only=True, trust_remote_code=True
            )
        except Exception:  # noqa: BLE001 - not cached
            continue
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    pytest.skip("no cached Nemotron-Labs-Diffusion tokenizer")


def _cfg(**decision):
    return make_cfg(
        seed=42,
        model_config={"vocab_size": 131072, "mask_token_id": 100},
        diffusion_lm={
            "canvas_width": 128,
            "mask_token_id": 100,
            "unroll": {"k_max": 2},
        },
        diffusion_decision={
            "max_questions_per_canvas": 32,
            "labels": {"codebook": "expanded255"},
            **decision,
        },
    )


SPEC = make_spec(noise=DiffusionNoise.ABSORBING, layout=DiffusionLayout.FULL_SEQUENCE)


def _tricky_records():
    """Junction edge cases: punctuation and newlines at piece ends, slashes, digit
    and punctuation ids, special-token text, soft labels, every question type."""
    records = [
        make_record(
            "mix",
            state={"ticket": "Refund for order #12 / <|im_end|> please", "n": 3},
            instructions="/strict: answer from the state only.",
            questions={
                "1st": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "options": [
                        {"name": "billing", "description": "money (refunds)."},
                        {"name": "ops/", "description": "  trailing spaces  "},
                        {"name": "Tech\n", "description": None},
                        "plain name ",
                        "<|im_end|>",
                        "üñí ©ode",
                    ],
                },
                "_flag": {
                    "type": "noul",
                    "instructions": "Escalate?",
                    "criteria": {"true": "yes: urgent!", "false": ""},
                },
                "Score": {
                    "type": "score",
                    "instructions": "Urgency 1-5.",
                    "levels": ["1", "2", "3", "4", "5"],
                },
                "wide": {
                    "type": "score",
                    "instructions": "Fine-grained.",
                    "levels": [f"level {i}" for i in range(12)],
                },
            },
            labels={
                "1st": {
                    "kind": "dist",
                    "candidate_indices": [0, 3, 5],
                    "probs": [0.5, 0.3, 0.2],
                },
                "_flag": {"kind": "hard", "gold_idx": 1},
                "Score": {"kind": "dist", "probs": [0.1, 0.2, 0.4, 0.2, 0.1]},
                "wide": {"kind": "set", "allowed_set": [2, 3]},
            },
        ),
        make_record(
            "many",
            state="Long context. " * 400,
            questions={
                f"q{i}": {
                    "type": "choice",
                    "instructions": f"Question {i}?",
                    "options": [f"opt {j}" for j in range(3 + i % 4)],
                }
                for i in range(14)
            },
            labels={f"q{i}": {"kind": "hard", "gold_idx": i % 3} for i in range(14)},
        ),
        make_record(
            "wide",
            state="pick a class",
            questions={
                "intent": {
                    "type": "choice",
                    "instructions": "Intent.",
                    "options": [
                        {"name": f"intent_{i}", "description": f"desc {i}:"}
                        for i in range(151)
                    ],
                }
            },
            labels={"intent": {"kind": "hard", "gold_idx": 77}},
        ),
    ]
    return records


def _kwargs(tokenizer, cfg):
    plan, open_ids, close_ids = canvas._resolve_slot_plan(tokenizer, cfg, SPEC, 131072)
    return {
        "spec": SPEC,
        "vocab_size": 131072,
        "slot_plan": plan,
        "thought_open_ids": open_ids,
        "thought_close_ids": close_ids,
    }


def _row(tokenizer, record, cfg, **extra):
    try:
        return canvas._canvas_row(
            tokenizer,
            deepcopy(record),
            cfg,
            1.0,
            train=True,
            **_kwargs(tokenizer, cfg),
            **extra,
        )
    except SchemaError as error:
        return str(error)


def test_nemotron_is_segment_safe(nemotron):
    assert segments.segment_safety(nemotron) is None


def test_unproven_tokenizers_get_no_segments():
    assert (
        segments.segment_safety(ChatCharacterTokenizer()) == "no fast tokenizer backend"
    )
    other = SimpleNamespace(
        backend_tokenizer=SimpleNamespace(
            to_str=lambda: (
                '{"normalizer": null, "pre_tokenizer": {"type": "Metaspace"},'
                ' "model": {"type": "BPE"}, "added_tokens": []}'
            )
        )
    )
    assert segments.segment_safety(other) == "pre-tokenizer not proven safe"
    segment, reason = canvas._reserve_segments(
        ChatCharacterTokenizer(),
        make_record(),
        make_cfg(),
        SPEC,
        vocab_size=256,
        slot_plan=None,
        thought_open_ids=(),
        thought_close_ids=(),
    )
    assert segment is None and reason == "no fast tokenizer backend"


def test_added_token_with_whitespace_is_refused(nemotron):
    import json

    spec = json.loads(nemotron.backend_tokenizer.to_str())
    spec["added_tokens"].append(
        {"content": "a b", "lstrip": False, "rstrip": False, "single_word": False}
    )
    fake = SimpleNamespace(
        backend_tokenizer=SimpleNamespace(to_str=lambda: json.dumps(spec))
    )
    assert "straddle" in segments.segment_safety(fake)


@pytest.mark.parametrize(
    "decision",
    [
        {},
        {"answer_format": "lines"},
        {"labels": {"codebook": "expanded255", "label_shuffle_prob": 0.7}},
    ],
    ids=["auto", "lines", "shuffle"],
)
def test_assembled_draws_equal_full_tokenization(nemotron, decision):
    cfg = _cfg(**decision)
    kwargs = _kwargs(nemotron, cfg)
    before = segments.SEGMENT_STATS["assembled"]
    checked = 0
    for record in _tricky_records():
        first = _row(nemotron, record, cfg)
        prepared = reserve = None
        if isinstance(first, dict):
            prepared, reason = canvas._prepared_segments(
                nemotron, record, first, cfg, **kwargs
            )
            assert reason is None, reason
        reserve, reason = canvas._reserve_segments(nemotron, record, cfg, **kwargs)
        assert reason is None, reason
        assert reserve.suffix_stored and (
            prepared is None or not prepared.suffix_stored
        )
        for draw in DRAWS:
            full = _row(nemotron, record, cfg, draw=draw)
            got = _row(nemotron, record, cfg, draw=draw, segments=reserve)
            assert got == full, (record["id"], draw)
            if prepared is not None:
                cached = first["canvas"].prompt_ids
                got = _row(
                    nemotron,
                    record,
                    cfg,
                    draw=draw,
                    segments=prepared,
                    cached_prompt=cached,
                )
                assert got == full, (record["id"], draw)
                if isinstance(full, dict):
                    assert canvas._decision_example(
                        got["canvas"], got
                    ) == canvas._decision_example(full["canvas"], full)
            checked += 1
    assert checked == len(DRAWS) * 3
    assert segments.SEGMENT_STATS["assembled"] > before


def test_overflowing_draw_falls_back_and_raises_like_full(nemotron):
    cfg = _cfg(answer_format="lines")
    record = _tricky_records()[1]
    reserve, reason = canvas._reserve_segments(
        nemotron, record, cfg, **_kwargs(nemotron, cfg)
    )
    assert reason is None
    narrow = _cfg(answer_format="lines")
    narrow["diffusion_lm"]["canvas_width"] = 40
    before = segments.SEGMENT_STATS["fallback:answer template overflows the canvas"]
    full = _row(nemotron, record, narrow, draw=(1, 0))
    got = _row(nemotron, record, narrow, draw=(1, 0), segments=reserve)
    assert isinstance(full, str) and got == full
    assert (
        segments.SEGMENT_STATS["fallback:answer template overflows the canvas"]
        == before + 1
    )


def test_indexed_pieces_for_unstored_labels_are_encoded(nemotron):
    cfg = _cfg(labels={"codebook": "expanded255", "label_shuffle_prob": 1.0})
    record = _tricky_records()[1]  # 14 questions: indexed format
    reserve, reason = canvas._reserve_segments(
        nemotron, record, _cfg(), **_kwargs(nemotron, cfg)
    )
    assert reason is None and reserve.fmt == "indexed"
    before = segments.SEGMENT_STATS["indexed_piece_encoded"]
    assert _row(nemotron, record, cfg, draw=(3, 1), segments=reserve) == _row(
        nemotron, record, cfg, draw=(3, 1)
    )
    assert segments.SEGMENT_STATS["indexed_piece_encoded"] > before


def test_segments_survive_the_cache_unpickler(nemotron):
    cfg = _cfg()
    record = _tricky_records()[0]
    reserve, _ = canvas._reserve_segments(
        nemotron, record, cfg, **_kwargs(nemotron, cfg)
    )
    pool = SourcePool(
        source="test",
        cap=None,
        entries=[encode_reserve(record)],
        records=[None],
        segments=[reserve],
    )
    loaded = _CacheUnpickler(io.BytesIO(pickle.dumps(pool, protocol=5))).load()
    assert loaded == pool
    assert loaded.segments[0].without_suffix().suffix_stored is False


def test_batch_plan_rows_equal_with_and_without_segments(nemotron):
    cfg = _cfg(answer_format="lines")
    cfg["sequence_len"] = 900
    records = []
    for index in range(12):
        record = deepcopy(_tricky_records()[index % 3])
        record["id"] = f"r{index}"
        record["state"] = f"variant {index}: " + ("x " * (index * 60))
        records.append(record)
    window = set(cap_window(len(records), 5, 42, "test"))
    built = datasets._prepare_canvas_items(
        nemotron,
        [
            (record, "segments" if index in window else "reserve")
            for index, record in enumerate(records)
        ],
        cfg,
        {},
        SPEC,
        train=True,
    )
    entries, kept, pieces = [], [], []
    for index, (row, _, segment, reason) in enumerate(built):
        assert reason is None, reason
        if index in window:
            entries.append(row)
            kept.append(records[index])
        else:
            entries.append(encode_reserve(records[index]))
            kept.append(None)
        pieces.append(segment)
    builder = datasets._plan_row_builder(cfg, nemotron, SPEC)

    def plan(with_segments):
        return DecisionBatchPlan(
            [
                SourcePool(
                    source="test",
                    cap=5,
                    entries=list(entries),
                    records=list(kept),
                    segments=list(pieces) if with_segments else None,
                )
            ],
            seed=42,
            micro_batch_size=2,
            stratified=False,
            weights=None,
            temperature=None,
            limits=BudgetLimits(logical=900, physical=None),
            layout=SPEC.layout,
            build_row=builder,
        )

    plain, fast = plan(False), plan(True)
    assert plain.budget_drops == fast.budget_drops
    for epoch in range(3):
        for position in range(plain.draws_per_epoch):
            assert fast.row(epoch, position) == plain.row(epoch, position), (
                epoch,
                position,
            )


def test_records_with_aliased_questions_keep_full_tokenization(nemotron):
    # deepcopy keeps aliasing, so the full path permutes a shared question once per
    # key; such a record must not be assembled from per-question segments.
    cfg = _cfg(answer_format="lines")
    shared = {
        "type": "choice",
        "instructions": "Pick.",
        "options": ["alpha", "beta", "gamma", "delta"],
    }
    record = make_record(
        "alias",
        questions={"a": shared, "b": shared},
        labels={
            "a": {"kind": "hard", "gold_idx": 1},
            "b": {"kind": "hard", "gold_idx": 2},
        },
    )
    kwargs = _kwargs(nemotron, cfg)
    segment, reason = canvas._reserve_segments(nemotron, record, cfg, **kwargs)
    assert segment is None and reason == "questions or targets share objects"
    unaliased = deepcopy(record)
    unaliased["questions"]["b"] = deepcopy(shared)
    built, _ = canvas._reserve_segments(nemotron, unaliased, cfg, **kwargs)
    before = segments.SEGMENT_STATS["fallback:questions or targets share objects"]
    for draw in DRAWS:
        # Segments of the unaliased twin applied to the aliased record fall back.
        assert _row(nemotron, record, cfg, draw=draw, segments=built) == _row(
            nemotron, record, cfg, draw=draw
        )
    assert segments.SEGMENT_STATS["fallback:questions or targets share objects"] == (
        before + len(DRAWS)
    )
