"""Materialize a pinned, public-only typed-decision train/dev mix.

The generator never downloads or writes protected evaluation data.  It can
optionally use locally normalized JevBench and Nimble heldouts to reject equal
states before public output is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from datasets import load_dataset

from axolotl.integrations.decision.adapters import normalize_record
from axolotl.integrations.decision.hygiene import (
    assert_split_isolation,
    decontaminate,
)

DATASET = "tasksource/procedural-typed-decisions"
REVISION = "916e6cce365a65c37d58db70c9e369795817651e"
GENERATOR_REPOSITORY_LICENSE = "CC-BY-4.0"
DATASET_CARD_LICENSE = "Apache-2.0"
CONFIGS = (
    "arithmetic",
    "entity_belief_tracking",
    "event_state_reconstruction",
    "evidence_sufficiency",
    "multi_view_adjudication",
    "needle_retrieval",
    "partial_observation_calibration",
    "policy_applicability",
    "policy_under_uncertainty",
    "record_aggregation",
    "state_perturbation",
    "table_lookup",
)
TRAIN_PER_CONFIG = 2560
DEV_PER_CONFIG = 256
SEED = "public-procedural-typed-decisions-v1"
CODEBOOK = "spreadsheet151"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"invalid JSON in {path}:{line_number}"
                        ) from error
                    if not isinstance(row, dict):
                        raise ValueError(
                            f"protected row in {path}:{line_number} is not an object"
                        )
                    rows.append(row)
    return rows


def _rank(record: Mapping[str, Any], config: str) -> str:
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("public source record requires a nonempty id")
    return hashlib.sha256(f"{SEED}\0{config}\0{record_id}".encode()).hexdigest()


def _normalize_row(
    row: Mapping[str, Any], *, config: str, split: str
) -> dict[str, str]:
    record = normalize_record("procedural", dict(row), training=True, codebook=CODEBOOK)
    record["source"] = f"public_procedural.{config}"
    record["family"] = f"{config}:{split}"
    metadata = dict(record.get("source_metadata", {}))
    metadata["provenance"] = {
        "dataset": DATASET,
        "revision": REVISION,
        "license_evidence": {
            "dataset_card_declared": DATASET_CARD_LICENSE,
            "linked_generator_repository": GENERATOR_REPOSITORY_LICENSE,
        },
        "config": config,
        "split": split,
        "source_id": record["id"],
    }
    record["source_metadata"] = metadata
    return {"normalized": _canonical(record)}


def _normalize(config: str, split: str) -> list[dict[str, Any]]:
    source = load_dataset(DATASET, name=config, split=split, revision=REVISION)
    normalized = source.map(
        _normalize_row,
        fn_kwargs={"config": config, "split": split},
        num_proc=4,
        remove_columns=source.column_names,
        keep_in_memory=True,
        desc=f"Normalizing {config} {split}",
    )
    return [json.loads(row["normalized"]) for row in normalized]


def _select(
    records: list[dict[str, Any]], config: str, limit: int
) -> list[dict[str, Any]]:
    if len(records) < limit:
        raise ValueError(f"{config} has {len(records)} rows; need {limit}")
    return sorted(records, key=lambda record: _rank(record, config))[:limit]


def _require_unique_ids(records: list[dict[str, Any]], config: str, split: str) -> None:
    ids = [record.get("id") for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{config} {split} has duplicate public source ids")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(_canonical(row) + "\n" for row in rows), encoding="utf-8")


def _assert_source_group_isolation(
    splits: Mapping[str, Iterable[Mapping[str, Any]]],
) -> None:
    origins: dict[tuple[str, str], str] = {}
    for split, records in splits.items():
        for record in records:
            source = record.get("source")
            group = record.get("group")
            if not isinstance(source, str) or not source:
                raise ValueError("decision record requires a source")
            if not isinstance(group, str) or not group:
                raise ValueError("decision record requires a group")
            key = source, group
            if key in origins and origins[key] != split:
                raise ValueError(
                    f"source/group {key!r} overlaps splits {origins[key]!r} and {split!r}"
                )
            origins[key] = split


def build(output_dir: Path, protected_paths: Iterable[Path] = ()) -> dict[str, Any]:
    protected_paths = list(protected_paths)
    protected = _read_jsonl(protected_paths)
    protected_sources = Counter(str(record.get("source", "")) for record in protected)
    missing = {"jev_bench", "nimble"} - set(protected_sources)
    if protected and missing:
        raise ValueError(
            "protected inputs must contain normalized JevBench and Nimble rows; missing "
            + ", ".join(sorted(missing))
        )

    train, dev = [], []
    for config in CONFIGS:
        candidate_train = _normalize(config, "train")
        candidate_dev = _normalize(config, "validation")
        _require_unique_ids(candidate_train, config, "train")
        _require_unique_ids(candidate_dev, config, "validation")
        kept_train, _ = decontaminate(
            candidate_train, protected, exclude_families=False
        )
        kept_dev, _ = decontaminate(candidate_dev, protected, exclude_families=False)
        train.extend(_select(list(kept_train), config, TRAIN_PER_CONFIG))
        dev.extend(_select(list(kept_dev), config, DEV_PER_CONFIG))
    assert_split_isolation({"train": train, "dev": dev})
    _assert_source_group_isolation({"train": train, "dev": dev})

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path, dev_path = output_dir / "train.jsonl", output_dir / "dev.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(dev_path, dev)
    contract = {
        "schema_version": 1,
        "name": "public-procedural-typed-decisions-v1",
        "purpose": "Public, reproducible typed-decision train/dev materialization; not a quality claim.",
        "source": {
            "dataset": DATASET,
            "revision": REVISION,
            "card": f"https://huggingface.co/datasets/{DATASET}",
            "generator_repository": "https://github.com/sileod/tasksource",
            "license_evidence": {
                "dataset_card_declared": DATASET_CARD_LICENSE,
                "linked_generator_repository": GENERATOR_REPOSITORY_LICENSE,
            },
            "redistribution_status": "dataset card declares Apache-2.0; retain this provenance when sharing materialized output",
            "configs": list(CONFIGS),
        },
        "selection": {
            "algorithm": "sha256(seed\\0config\\0id), ascending",
            "seed": SEED,
            "codebook": CODEBOOK,
            "train_per_config": TRAIN_PER_CONFIG,
            "dev_per_config": DEV_PER_CONFIG,
        },
        "protected_decontamination": {
            "status": "performed" if protected else "not_requested",
            "required_sources_when_requested": ["jev_bench", "nimble"],
            "checked_when_requested": ["canonical_state"],
            "source_family_group_check": "not comparable across sources: hygiene family keys and source/group keys include source; a cross-source semantic-group mapping is not available",
            "protected_rows_by_source": dict(sorted(protected_sources.items())),
            "protected_input_sha256": [_sha256_bytes(path) for path in protected_paths],
        },
        "files": {
            "train": {
                "path": "train.jsonl",
                "lines": len(train),
                "sha256": _sha256_bytes(train_path),
            },
            "dev": {
                "path": "dev.jsonl",
                "lines": len(dev),
                "sha256": _sha256_bytes(dev_path),
            },
        },
    }
    (output_dir / "contract.json").write_text(
        _canonical(contract) + "\n", encoding="utf-8"
    )
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--protected-normalized",
        type=Path,
        action="append",
        help="Optional untracked normalized JevBench/Nimble heldout JSONL for canonical-state exclusion; never copied to output.",
    )
    args = parser.parse_args()
    build(args.output_dir, args.protected_normalized or ())


if __name__ == "__main__":
    main()
