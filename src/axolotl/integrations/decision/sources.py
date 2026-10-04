"""Raw source loading and normalization for decision datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from glob import glob
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from datasets import Dataset, load_dataset

from axolotl.utils.data.shared import load_dataset_with_config, merge_datasets
from axolotl.utils.dict import DictDefault

from .adapters import normalize_record


def _value(value: Any, key: str, default: Any = None) -> Any:
    return (
        value.get(key, default)
        if isinstance(value, Mapping)
        else getattr(value, key, default)
    )


def _workers(cfg, count):
    workers = min(int(_value(cfg, "dataset_num_proc", 1) or 1), count)
    return workers if workers > 1 else None


def local_jsonl_paths(path: str, data_files: Any, split: str) -> list[Path] | None:
    if path != "json" or data_files is None:
        return None
    selected = data_files.get(split) if isinstance(data_files, Mapping) else data_files
    names = [selected] if isinstance(selected, str) else list(selected or ())
    if not names or not all(
        isinstance(name, str) and not urlparse(name).scheme for name in names
    ):
        return None
    paths: list[Path] = []
    for name in names:
        matches = sorted(Path(match) for match in glob(name))
        if not matches:
            raise ValueError(f"local JSONL path matched no files: {name}")
        if any(path.suffix != ".jsonl" or not path.is_file() for path in matches):
            return None
        paths.extend(matches)
    return paths


def _local_jsonl(entry: Any, cfg: Any) -> Dataset | None:
    paths = local_jsonl_paths(
        str(_value(entry, "path")),
        _value(entry, "data_files"),
        str(_value(entry, "split", "train")),
    )
    if paths is None:
        return None
    partitions = []
    for path in paths:
        lines = load_dataset(
            "text",
            data_files={"train": str(path)},
            split="train",
            sample_by="line",
            keep_linebreaks=False,
        )

        def validate(row, index, filename=str(path)):
            if not row["text"].strip():
                return {"raw_json": None}
            try:
                payload = json.loads(row["text"])
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {filename}:{index + 1}: {error.msg}"
                ) from error
            if not isinstance(payload, dict):
                raise ValueError(
                    f"JSONL record at {filename}:{index + 1} must be an object"
                )
            return {"raw_json": row["text"]}

        partitions.append(
            lines.map(validate, with_indices=True, num_proc=_workers(cfg, len(lines)))
            .filter(lambda row: row["raw_json"] is not None)
            .remove_columns("text")
        )
    if not partitions:
        return Dataset.from_dict({"raw_json": []})
    return merge_datasets(
        partitions,
        DictDefault(
            {
                "curriculum_sampling": False,
                "shuffle_merged_datasets": False,
                "shuffle_before_merging_datasets": False,
                "seed": 0,
            }
        ),
    )


def load_source(entry: Any, cfg: Any = None) -> Dataset:
    local = _local_jsonl(entry, cfg)
    if local is not None:
        return local
    path, split, files = (
        str(_value(entry, "path")),
        str(_value(entry, "split", "train")),
        _value(entry, "data_files"),
    )
    if path in {"json", "parquet", "csv", "text"}:
        data_files = files if isinstance(files, Mapping) else {split: files}
        dataset = load_dataset(
            path, split=split, data_files=data_files, name=_value(entry, "name")
        )
    else:
        values = dict(entry) if isinstance(entry, Mapping) else dict(vars(entry))
        values.setdefault("split", split)
        values.setdefault("trust_remote_code", False)
        dataset = load_dataset_with_config(
            DictDefault(values), bool(_value(cfg, "hf_use_auth_token", False))
        )
    return dataset.map(
        lambda row: {
            "raw_json": json.dumps(dict(row), ensure_ascii=False, sort_keys=True)
        },
        remove_columns=dataset.column_names,
    )


def normalize_source(
    entry: Any, cfg: Any, training: bool, target_basis=None, codebook="vendored26"
) -> list[dict[str, Any]]:
    dataset = load_source(entry, cfg)
    if not len(dataset):
        return []
    adapter = str(_value(entry, "type", "jsonl"))
    split = str(_value(entry, "split", "train")).lower()

    def normalize(row):
        normalized = normalize_record(
            adapter,
            json.loads(row["raw_json"]),
            training=training,
            target_basis=target_basis,
            codebook=codebook,
            source_split=split,
        )
        return {
            "normalized": json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        }

    mapped = dataset.map(normalize, num_proc=_workers(cfg, len(dataset)))
    return [json.loads(value) for value in mapped["normalized"]]
