"""Arrow persistence contracts for prepared decision rows."""

from __future__ import annotations

import json

from axolotl.integrations.decision import prepared_cache
from axolotl.integrations.decision.loss import decision_example_from_canvas
from axolotl.integrations.decision.records import DecisionCanvas, OrdinalMetadata
from axolotl.integrations.decision.row_codec import (
    ROW_FEATURES,
    dataset_to_rows,
    row_from_arrow,
    row_to_arrow,
    rows_to_dataset,
)


def _row(identifier: str) -> dict:
    canvas = DecisionCanvas(
        prompt_ids=(1, 2),
        canvas_ids=(3, 4, 5),
        label_positions=(0, 1, 2),
        allowed_ids=((11, 12), (21, 22), (31, 32)),
        question_ids=("hard", "soft", "ordinal"),
        targets=(
            {"kind": "hard", "gold_idx": 1},
            {"kind": "dist", "probs": [0.25, 0.75]},
            {"kind": "set", "allowed_set": [0, 1]},
        ),
        pinned_mask=(False, True, False),
        semantic_mask=(True, True, True),
        slot_mask=(False, False, False),
        template_length=3,
        ordinal_metadata=(
            None,
            None,
            OrdinalMetadata(("low", "high"), ("a", "b"), (0, 1)),
        ),
    )
    return {
        "canvas": canvas,
        "decision_example": decision_example_from_canvas(canvas, source_weight=0.25),
        "record": {
            "id": identifier,
            "grouped_record_ids": (identifier, f"{identifier}-child"),
            "provenance": {"partition": "train", "rank": 3},
        },
        "source": "jsonl",
        "length": 5,
        "source_weight": 0.25,
    }


def test_row_codec_preserves_typed_labels_and_provenance():
    original = _row("example")
    encoded = row_to_arrow(original)

    assert set(encoded) == set(ROW_FEATURES)
    assert encoded["allowed_ids"] == ((11, 12), (21, 22), (31, 32))
    assert json.loads(encoded["targets_json"])[1]["probs"] == [0.25, 0.75]
    restored = row_from_arrow(encoded)
    assert restored["canvas"] == original["canvas"]
    assert restored["decision_example"] == original["decision_example"]
    assert restored["record"] == original["record"]
    assert restored["source"] == original["source"]


def test_arrow_dataset_roundtrip_retains_split_order_and_empty_schema():
    train = [_row("a"), _row("b")]
    eval_rows = [_row("c")]
    dataset = rows_to_dataset(train, eval_rows)

    assert dataset.features == ROW_FEATURES
    assert dataset["split"] == ["train", "train", "eval"]
    restored_train, restored_eval = dataset_to_rows(dataset)
    assert [row["record"]["id"] for row in restored_train] == ["a", "b"]
    assert [row["record"]["id"] for row in restored_eval] == ["c"]
    assert restored_eval[0]["decision_example"] == eval_rows[0]["decision_example"]
    assert rows_to_dataset([], []).features == ROW_FEATURES


def test_prepared_cache_uses_core_arrow_directory_and_small_metadata(tmp_path):
    identity = {"source": "local.jsonl", "tokenizer": "revision"}
    train = [_row("a")]
    eval_rows = [_row("b")]

    prepared_cache.store(tmp_path, "key", identity, {}, train, eval_rows, {})

    dataset_dir = tmp_path / "decision_cache" / "key"
    assert (dataset_dir / "state.json").is_file()
    sidecar = json.loads((tmp_path / "decision_cache" / "key.json").read_text())
    assert "rows" not in sidecar["content"]
    restored = prepared_cache.load(tmp_path, "key", identity)
    assert restored is not None
    assert [row["record"]["id"] for row in restored[0]] == ["a"]
    assert [row["record"]["id"] for row in restored[1]] == ["b"]
    assert (
        prepared_cache.load(
            tmp_path, "key", identity, cfg={"skip_prepare_dataset": True}
        )
        is None
    )
    assert (
        prepared_cache.load(tmp_path, "key", identity, cfg={"is_preprocess": True})
        is None
    )

    arrow_file = next(dataset_dir.glob("*.arrow"))
    arrow_file.write_bytes(b"invalid Arrow data")
    assert prepared_cache.load(tmp_path, "key", identity) is None
