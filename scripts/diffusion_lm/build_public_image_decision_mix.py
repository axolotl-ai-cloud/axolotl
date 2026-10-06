"""Materialize a pinned, public-only image typed-decision train/dev/test mix.

Train and dev come from A-OKVQA and the image-bearing rows of ScienceQA; test
is CV-Bench, which is never used for training.  Images are written once per
content hash next to the JSONL files and referenced by relative path.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset, Image, load_dataset
from PIL import Image as PILImage

from axolotl.integrations.diffusion_decision.hygiene import state_fingerprint

SEED = "public-image-typed-decisions-v1"
CODEBOOK = "vendored26"
MAX_OPTIONS = 26
TRAIN_PER_SOURCE = 4096
DEV_PER_SOURCE = 256
TEST_CAP = 0
CVBENCH_ANSWER = re.compile(r"^\(([A-Z])\)$")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


@dataclass(frozen=True)
class Source:
    name: str
    dataset: str
    revision: str
    config: str
    splits: Mapping[str, str]
    license_evidence: Mapping[str, str]
    normalize: Callable[[Mapping[str, Any], int, str], dict[str, Any] | None]
    flags: Sequence[str] = field(default_factory=tuple)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank(source: str, split: str, record_id: str) -> str:
    return hashlib.sha256(
        f"{SEED}\0{source}\0{split}\0{record_id}".encode()
    ).hexdigest()


def _options(choices: Any) -> list[str] | None:
    if not isinstance(choices, list) or not choices or len(choices) > MAX_OPTIONS:
        return None
    options = [" ".join(str(choice).split()) for choice in choices]
    if any(not option for option in options) or len(set(options)) != len(options):
        return None
    return options


def _gold_index(value: Any, count: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < count:
        return None
    return value


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _record(
    *,
    source: str,
    family: str,
    record_id: str,
    group: str,
    state: Mapping[str, Any],
    instructions: str,
    options: Sequence[str],
    gold_idx: int,
    provenance: Mapping[str, Any],
    attributes: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source": source,
        "family": family,
        "id": record_id,
        "group": group,
        "state": dict(state),
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": instructions,
                "options": list(options),
            }
        },
        "labels": {"answer": {"kind": "hard", "gold_idx": gold_idx}},
        "source_metadata": {
            "provenance": dict(provenance),
            "attributes": dict(attributes),
        },
    }


def _provenance(source: Source, split: str, source_id: Any) -> dict[str, Any]:
    return {
        "dataset": source.dataset,
        "revision": source.revision,
        "config": source.config,
        "split": source.splits[split],
        "source_id": source_id,
        "license": dict(source.license_evidence),
    }


def _normalize_aokvqa(
    row: Mapping[str, Any], index: int, split: str
) -> dict[str, Any] | None:
    options = _options(row.get("choices"))
    question_id = _text(row.get("question_id"))
    question = _text(row.get("question"))
    if options is None or not question_id or not question:
        return None
    gold_idx = _gold_index(row.get("correct_choice_idx"), len(options))
    if gold_idx is None:
        return None
    return _record(
        source="public_image.aokvqa",
        family=question_id,
        record_id=question_id,
        group=question_id,
        state={"question": question},
        instructions="Choose the answer to the question about the image.",
        options=options,
        gold_idx=gold_idx,
        provenance=_provenance(AOKVQA, split, question_id),
        attributes={"row_index": index},
    )


def _normalize_scienceqa(
    row: Mapping[str, Any], index: int, split: str
) -> dict[str, Any] | None:
    options = _options(row.get("choices"))
    question = _text(row.get("question"))
    if options is None or not question:
        return None
    gold_idx = _gold_index(row.get("answer"), len(options))
    if gold_idx is None:
        return None
    state: dict[str, Any] = {"question": question}
    hint = _text(row.get("hint"))
    if hint:
        state["context"] = hint
    record_id = f"{SCIENCEQA.splits[split]}-{index}"
    return _record(
        source="public_image.scienceqa",
        family=record_id,
        record_id=record_id,
        group=record_id,
        state=state,
        instructions="Choose the answer to the question using the image and context.",
        options=options,
        gold_idx=gold_idx,
        provenance=_provenance(SCIENCEQA, split, index),
        attributes={
            key: _text(row.get(key))
            for key in ("subject", "topic", "category", "skill", "grade")
        },
    )


def _normalize_cvbench(
    row: Mapping[str, Any], index: int, split: str
) -> dict[str, Any] | None:
    options = _options(row.get("choices"))
    question = _text(row.get("question"))
    task = _text(row.get("task"))
    match = CVBENCH_ANSWER.match(_text(row.get("answer")))
    if options is None or not question or not task or match is None:
        return None
    gold_idx = _gold_index(ord(match.group(1)) - ord("A"), len(options))
    if gold_idx is None:
        return None
    record_id = f"{CVBENCH.splits[split]}-{index}"
    return _record(
        source="public_image.cvbench",
        family=task,
        record_id=record_id,
        group=record_id,
        state={"question": question},
        instructions="Choose the answer to the question about the image.",
        options=options,
        gold_idx=gold_idx,
        provenance=_provenance(CVBENCH, split, index),
        attributes={
            "task": task,
            "type": _text(row.get("type")),
            "source": _text(row.get("source")),
            "filename": _text(row.get("filename")),
        },
    )


AOKVQA = Source(
    name="aokvqa",
    dataset="HuggingFaceM4/A-OKVQA",
    revision="d1b0efa3a436e9101dfbde3752db7607da696c35",
    config="default",
    splits={"train": "train", "dev": "validation"},
    license_evidence={
        "dataset_card_declared": "unstated on HF card; A-OKVQA project license",
        "project_repository": "Apache-2.0 (github.com/allenai/aokvqa LICENSE)",
        "images": "COCO 2017 images: Flickr Terms of Use, per-image licences "
        "(CC-BY-4.0 covers the COCO annotations, not the images)",
    },
    normalize=_normalize_aokvqa,
    flags=(
        "HuggingFaceM4/A-OKVQA declares no license on its HF card; the A-OKVQA "
        "project repository is Apache-2.0. Confirm the annotation terms before "
        "redistributing materialized output.",
        "A-OKVQA images are COCO 2017 Flickr photos under the Flickr Terms of Use "
        "with per-image licences; the COCO CC-BY-4.0 licence covers annotations only.",
    ),
)
SCIENCEQA = Source(
    name="scienceqa",
    dataset="derek-thomas/ScienceQA",
    revision="f18b0a70359ebfb41f658fd564208d0355b013f4",
    config="default",
    splits={"train": "train", "dev": "validation"},
    license_evidence={
        "dataset_card_declared": "cc-by-sa-4.0",
        "dataset_card_body": "CC BY-NC-SA 4.0 (Licensing Information section)",
        "images": "collected with the questions from K-12 science curricula; "
        "the card states no image-specific licence",
    },
    normalize=_normalize_scienceqa,
    flags=(
        "derek-thomas/ScienceQA card metadata says cc-by-sa-4.0 but the card body "
        "links CC BY-NC-SA 4.0; treat the NonCommercial reading as binding until "
        "resolved.",
        "ScienceQA images were collected with the questions from K-12 science "
        "curricula and the card states no image-specific licence; the dataset "
        "licence is the only stated term.",
    ),
)
CVBENCH = Source(
    name="cvbench",
    dataset="nyu-visionx/CV-Bench",
    revision="bc284db50d036958861cb60cdd7b77612052ce0d",
    config="default",
    splits={"test": "test"},
    license_evidence={
        "dataset_card_declared": "apache-2.0",
        "images": "repurposed from COCO (Flickr Terms of Use, per-image licences), "
        "ADE20K (its own terms of use) and Omni3D (licences of its source datasets)",
    },
    normalize=_normalize_cvbench,
    flags=(
        "nyu-visionx/CV-Bench declares Apache-2.0, but its images are repurposed "
        "from COCO, ADE20K and Omni3D and keep those sources' terms.",
    ),
)
TRAIN_SOURCES = (AOKVQA, SCIENCEQA)
TEST_SOURCES = (CVBENCH,)


def _image_payload(image: Any) -> tuple[bytes, str]:
    data = None
    if isinstance(image, Mapping):
        data = image.get("bytes")
        if data is None and image.get("path"):
            data = Path(image["path"]).read_bytes()
    elif isinstance(image, (bytes, bytearray)):
        data = bytes(image)
    elif isinstance(image, PILImage.Image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
    if not data:
        raise ValueError("image row carries no bytes")
    if data.startswith(PNG_MAGIC):
        return data, "png"
    if data.startswith(JPEG_MAGIC):
        return data, "jpg"
    with PILImage.open(io.BytesIO(data)) as decoded:
        buffer = io.BytesIO()
        decoded.save(buffer, format="PNG")
    return buffer.getvalue(), "png"


def image_digest(relative: str) -> str:
    return Path(relative).stem


class ImageStore:
    def __init__(self, output_dir: Path) -> None:
        self.directory = output_dir / "images"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.written: dict[str, int] = {}

    def put(self, image: Any) -> str:
        return self.put_payload(*_image_payload(image))

    def put_payload(self, data: bytes, extension: str) -> str:
        digest = hashlib.sha256(data).hexdigest()
        relative = f"images/{digest}.{extension}"
        if digest not in self.written:
            path = self.directory / f"{digest}.{extension}"
            if not path.exists():
                path.write_bytes(data)
            self.written[digest] = len(data)
        return relative

    @property
    def count(self) -> int:
        return len(self.written)

    @property
    def total_bytes(self) -> int:
        return sum(self.written.values())


def _image_presence(dataset: Dataset) -> list[bool]:
    present: list[bool] = []
    column = dataset.select_columns(["image"]).with_format("arrow")
    for batch in column.iter(batch_size=1024):
        present.extend(not null for null in batch.column("image").is_null().to_pylist())
    return present


def _load(source: Source, split: str) -> Dataset:
    return load_dataset(
        source.dataset,
        source.config,
        split=source.splits[split],
        revision=source.revision,
    )


def _select(
    candidates: Sequence[tuple[str, int, dict[str, Any]]],
    source: Source,
    split: str,
    limit: int,
) -> list[tuple[int, dict[str, Any]]]:
    if limit and len(candidates) < limit:
        raise ValueError(
            f"{source.name} {split} has {len(candidates)} usable rows; need {limit}"
        )
    ordered = sorted(candidates, key=lambda candidate: candidate[0])
    return [(index, record) for _, index, record in ordered]


def _materialize(
    source: Source,
    split: str,
    limit: int,
    store: ImageStore,
    excluded_states: set[str],
    excluded_images: set[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    dataset = _load(source, split)
    presence = _image_presence(dataset)
    drops: Counter[str] = Counter()
    candidates: list[tuple[str, int, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(dataset.remove_columns(["image"])):
        if not presence[index]:
            drops["no_image"] += 1
            continue
        record = source.normalize(row, index, split)
        if record is None:
            drops["invalid_choices_or_answer"] += 1
            continue
        if record["id"] in seen_ids:
            raise ValueError(f"{source.name} {split} has duplicate source ids")
        seen_ids.add(record["id"])
        if state_fingerprint(record["state"]) in excluded_states:
            drops["state_overlap"] += 1
            continue
        candidates.append((_rank(record["source"], split, record["id"]), index, record))
    ordered = _select(candidates, source, split, limit)
    images = dataset.cast_column("image", Image(decode=False))
    records: list[dict[str, Any]] = []
    for index, record in ordered:
        if limit and len(records) == limit:
            break
        data, extension = _image_payload(images[index]["image"])
        if hashlib.sha256(data).hexdigest() in excluded_images:
            drops["image_overlap"] += 1
            continue
        record["images"] = [store.put_payload(data, extension)]
        records.append(record)
    if limit and len(records) < limit:
        raise ValueError(
            f"{source.name} {split} has {len(records)} usable rows after image "
            f"exclusion; need {limit}"
        )
    return records, drops


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(_canonical(row) + "\n" for row in rows), encoding="utf-8")


def _assert_source_group_isolation(
    splits: Mapping[str, Iterable[Mapping[str, Any]]],
) -> None:
    origins: dict[tuple[str, str], str] = {}
    for split, records in splits.items():
        for record in records:
            key = record["source"], record["group"]
            if key in origins and origins[key] != split:
                raise ValueError(
                    f"source/group {key!r} overlaps splits {origins[key]!r} and {split!r}"
                )
            origins[key] = split


def _assert_image_isolation(
    splits: Mapping[str, Iterable[Mapping[str, Any]]],
) -> None:
    origins: dict[str, str] = {}
    for split, records in splits.items():
        for record in records:
            for relative in record["images"]:
                digest = image_digest(relative)
                if origins.setdefault(digest, split) != split:
                    raise ValueError(
                        f"image {digest} overlaps splits {origins[digest]!r} and {split!r}"
                    )


def _states(records: Iterable[Mapping[str, Any]]) -> set[str]:
    return {state_fingerprint(record["state"]) for record in records}


def _image_digests(records: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        image_digest(relative) for record in records for relative in record["images"]
    }


def _source_contract(source: Source) -> dict[str, Any]:
    return {
        "dataset": source.dataset,
        "revision": source.revision,
        "config": source.config,
        "card": f"https://huggingface.co/datasets/{source.dataset}",
        "splits": dict(source.splits),
        "license_evidence": dict(source.license_evidence),
    }


def build(
    output_dir: Path,
    *,
    train_per_source: int = TRAIN_PER_SOURCE,
    dev_per_source: int = DEV_PER_SOURCE,
    test_cap: int = TEST_CAP,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    store = ImageStore(output_dir)
    counts: dict[str, dict[str, int]] = {}
    drops: dict[str, dict[str, dict[str, int]]] = {}

    def run(
        source: Source,
        split: str,
        limit: int,
        excluded: set[str],
        excluded_images: set[str],
    ) -> list:
        records, dropped = _materialize(
            source, split, limit, store, excluded, excluded_images
        )
        counts.setdefault(split, {})[source.name] = len(records)
        drops.setdefault(split, {})[source.name] = dict(sorted(dropped.items()))
        return records

    test: list[dict[str, Any]] = []
    for source in TEST_SOURCES:
        test.extend(run(source, "test", test_cap, set(), set()))
    excluded = _states(test)
    excluded_images = _image_digests(test)
    train: list[dict[str, Any]] = []
    for source in TRAIN_SOURCES:
        train.extend(run(source, "train", train_per_source, excluded, excluded_images))
    excluded |= _states(train)
    excluded_images |= _image_digests(train)
    dev: list[dict[str, Any]] = []
    for source in TRAIN_SOURCES:
        dev.extend(run(source, "dev", dev_per_source, excluded, excluded_images))
    splits = {"train": train, "dev": dev, "test": test}
    _assert_source_group_isolation(splits)
    _assert_image_isolation(splits)

    files = {}
    for split, records in (("train", train), ("dev", dev), ("test", test)):
        path = output_dir / f"{split}.jsonl"
        _write_jsonl(path, records)
        files[split] = {
            "path": path.name,
            "lines": len(records),
            "sha256": _sha256_bytes(path),
        }
    contract = {
        "schema_version": 1,
        "name": SEED,
        "purpose": "Public, reproducible image typed-decision train/dev/test materialization; not a quality claim.",
        "sources": {
            "train_dev": {
                source.name: _source_contract(source) for source in TRAIN_SOURCES
            },
            "test": {source.name: _source_contract(source) for source in TEST_SOURCES},
        },
        "license_flags": [
            flag for source in TRAIN_SOURCES + TEST_SOURCES for flag in source.flags
        ],
        "selection": {
            "algorithm": "sha256(seed\\0source\\0split\\0id), ascending",
            "seed": SEED,
            "codebook": CODEBOOK,
            "train_per_source": train_per_source,
            "dev_per_source": dev_per_source,
            "test_cap": test_cap,
        },
        "split_isolation": {
            "state": "test states excluded from train and dev; train states excluded from dev",
            "source_group": "disjoint across splits",
            "family": "disjoint across splits (ScienceQA uses the row id)",
            "image": "image sha256 digests disjoint across splits; test images excluded from train and dev, train images from dev",
        },
        "counts": counts,
        "dropped": drops,
        "files": files,
        "images": {
            "directory": "images",
            "naming": "images/<sha256 of file bytes>.<png|jpg>",
            "count": store.count,
            "total_bytes": store.total_bytes,
        },
    }
    (output_dir / "contract.json").write_text(
        _canonical(contract) + "\n", encoding="utf-8"
    )
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-per-source", type=int, default=TRAIN_PER_SOURCE)
    parser.add_argument("--dev-per-source", type=int, default=DEV_PER_SOURCE)
    parser.add_argument(
        "--test-cap", type=int, default=TEST_CAP, help="0 keeps every CV-Bench row."
    )
    args = parser.parse_args()
    build(
        args.output_dir,
        train_per_source=args.train_per_source,
        dev_per_source=args.dev_per_source,
        test_cap=args.test_cap,
    )


if __name__ == "__main__":
    main()
