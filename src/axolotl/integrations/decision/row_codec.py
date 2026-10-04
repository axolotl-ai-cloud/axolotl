"""Stable Arrow columns for prepared decision canvases."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from datasets import Dataset, Features, List, Value

from .loss import decision_example_from_canvas
from .records import DecisionCanvas, OrdinalMetadata

ROW_FEATURES = Features(
    {
        "split": Value("string"),
        "canvas_prompt_ids": List(Value("int64")),
        "canvas_ids": List(Value("int64")),
        "label_positions": List(Value("int64")),
        "allowed_ids": List(List(Value("int64"))),
        "question_ids": List(Value("string")),
        "pinned_mask": List(Value("bool")),
        "semantic_mask": List(Value("bool")),
        "slot_mask": List(Value("bool")),
        "prompt_slot_mask": List(Value("bool")),
        "template_length": Value("int64"),
        "targets_json": Value("string"),
        "ordinal_metadata_json": Value("string"),
        "model_input_pixel_values": List(List(List(List(Value("float32"))))),
        "model_input_image_sizes": List(List(Value("int64"))),
        "record_json": Value("string"),
        "source": Value("string"),
        "source_weight": Value("float64"),
        "extras_json": Value("string"),
    }
)


def _json_default(item: Any) -> Any:
    if isinstance(item, Enum):
        return item.value
    if is_dataclass(item) and not isinstance(item, type):
        return asdict(item)
    if isinstance(item, Path):
        return str(item)
    if hasattr(item, "to_dict"):
        return item.to_dict()
    return str(item)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=_json_default)


def _canvas_from_json(data: Mapping[str, Any]) -> DecisionCanvas:
    ordinal = tuple(
        None
        if item is None
        else OrdinalMetadata(
            levels=tuple(item["levels"]),
            source_ids=tuple(item["source_ids"]),
            candidate_ranks=tuple(item["candidate_ranks"]),
        )
        for item in data.get("ordinal_metadata", ())
    )
    return DecisionCanvas(
        prompt_ids=tuple(data["prompt_ids"]),
        canvas_ids=tuple(data["canvas_ids"]),
        label_positions=tuple(data["label_positions"]),
        allowed_ids=tuple(tuple(value) for value in data["allowed_ids"]),
        question_ids=tuple(data["question_ids"]),
        targets=tuple(data["targets"]),
        pinned_mask=tuple(data["pinned_mask"]),
        semantic_mask=tuple(data["semantic_mask"]),
        slot_mask=tuple(data["slot_mask"]),
        template_length=int(data["template_length"]),
        prompt_slot_mask=tuple(data.get("prompt_slot_mask", ())),
        ordinal_metadata=ordinal,
        model_inputs=data.get("model_inputs", {}),
    )


def _row_to_json(row: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        key: item
        for key, item in row.items()
        if key not in {"canvas", "decision_example", "slot_plan"}
    }
    value["canvas"] = asdict(row["canvas"])
    example = row.get("decision_example")
    if example is not None:
        value["_decision_example_source_weight"] = example.source_weight
    return value


def _row_from_json(data: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(data)
    if isinstance(row.get("record"), dict) and "grouped_record_ids" in row["record"]:
        row["record"] = dict(row["record"])
        row["record"]["grouped_record_ids"] = tuple(row["record"]["grouped_record_ids"])
    row["canvas"] = _canvas_from_json(row["canvas"])
    source_weight = float(
        row.pop("_decision_example_source_weight", row.get("source_weight", 1.0))
    )
    row["decision_example"] = decision_example_from_canvas(
        row["canvas"], source_weight=source_weight
    )
    return row


def row_to_arrow(row: Mapping[str, Any], *, split: str = "train") -> dict[str, Any]:
    """Flatten stable canvas values and isolate heterogeneous fields per row."""
    value = _row_to_json(row)
    canvas = value.pop("canvas")
    record = value.pop("record", None)
    source = value.pop("source", None)
    source_weight = value.pop(
        "_decision_example_source_weight", value.get("source_weight", 1.0)
    )
    model_inputs = canvas.pop("model_inputs", {})
    pixel_values = model_inputs.get("pixel_values", ())
    image_sizes = model_inputs.get("image_sizes", ())
    if hasattr(pixel_values, "tolist"):
        pixel_values = pixel_values.tolist()
    if hasattr(image_sizes, "tolist"):
        image_sizes = image_sizes.tolist()
    return {
        "split": split,
        "canvas_prompt_ids": canvas["prompt_ids"],
        "canvas_ids": canvas["canvas_ids"],
        "label_positions": canvas["label_positions"],
        "allowed_ids": canvas["allowed_ids"],
        "question_ids": canvas["question_ids"],
        "pinned_mask": canvas["pinned_mask"],
        "semantic_mask": canvas["semantic_mask"],
        "slot_mask": canvas["slot_mask"],
        "prompt_slot_mask": canvas["prompt_slot_mask"],
        "template_length": canvas["template_length"],
        "targets_json": _json(canvas["targets"]),
        "ordinal_metadata_json": _json(canvas["ordinal_metadata"]),
        "model_input_pixel_values": pixel_values,
        "model_input_image_sizes": image_sizes,
        "record_json": _json(record),
        "source": None if source is None else str(source),
        "source_weight": float(source_weight),
        "extras_json": _json(value),
    }


def row_from_arrow(value: Mapping[str, Any]) -> dict[str, Any]:
    """Restore a typed row from one Arrow record."""
    row = json.loads(value["extras_json"])
    row["canvas"] = {
        "prompt_ids": value["canvas_prompt_ids"],
        "canvas_ids": value["canvas_ids"],
        "label_positions": value["label_positions"],
        "allowed_ids": value["allowed_ids"],
        "question_ids": value["question_ids"],
        "targets": json.loads(value["targets_json"]),
        "pinned_mask": value["pinned_mask"],
        "semantic_mask": value["semantic_mask"],
        "slot_mask": value["slot_mask"],
        "prompt_slot_mask": value["prompt_slot_mask"],
        "template_length": value["template_length"],
        "ordinal_metadata": json.loads(value["ordinal_metadata_json"]),
        "model_inputs": {
            key: media
            for key, media in {
                "pixel_values": value.get("model_input_pixel_values", ()),
                "image_sizes": value.get("model_input_image_sizes", ()),
            }.items()
            if media
        },
    }
    record = json.loads(value["record_json"])
    if record is not None:
        row["record"] = record
    if value["source"] is not None:
        row["source"] = value["source"]
    row["_decision_example_source_weight"] = value["source_weight"]
    return _row_from_json(row)


def rows_to_dataset(
    train: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]]
) -> Dataset:
    rows = [row_to_arrow(row, split="train") for row in train]
    rows.extend(row_to_arrow(row, split="eval") for row in eval_rows)
    if not rows:
        return Dataset.from_dict(
            {name: [] for name in ROW_FEATURES}, features=ROW_FEATURES
        )
    return Dataset.from_list(rows, features=ROW_FEATURES)


def dataset_to_rows(
    dataset: Dataset,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for value in dataset:
        split = value["split"]
        if split == "train":
            train.append(row_from_arrow(value))
        elif split == "eval":
            eval_rows.append(row_from_arrow(value))
        else:
            raise ValueError(f"unknown prepared decision split: {split!r}")
    return train, eval_rows
