import hashlib
import io
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from datasets import Dataset, Features, Image, List as ListFeature, Value
from PIL import Image as PILImage

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/diffusion_lm/build_public_image_decision_mix.py"
)
SPEC = spec_from_file_location("build_public_image_decision_mix", SCRIPT)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

TRAIN, DEV, TEST = 3, 2, 0


def _png(seed: int) -> bytes:
    image = PILImage.new("RGB", (4, 4), (seed % 256, (seed * 7) % 256, 9))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image(seed: int) -> dict:
    return {"bytes": _png(seed), "path": None}


def _aokvqa_rows(split: str, count: int) -> Dataset:
    rows = {
        "image": [
            _image(index + (0 if split == "train" else 50)) for index in range(count)
        ],
        "question_id": [f"{split}-q{index}" for index in range(count)],
        "question": [f"What is {split} {index} doing?" for index in range(count)],
        "choices": [["run", "sit", "jump", "fly"] for _ in range(count)],
        "correct_choice_idx": [index % 4 for index in range(count)],
        "rationales": [["because"] for _ in range(count)],
    }
    features = Features(
        {
            "image": Image(),
            "question_id": Value("string"),
            "question": Value("string"),
            "choices": ListFeature(Value("string")),
            "correct_choice_idx": Value("int8"),
            "rationales": ListFeature(Value("string")),
        }
    )
    return Dataset.from_dict(rows, features=features)


def _scienceqa_rows(split: str, count: int, *, missing: int = 1) -> Dataset:
    rows = {
        "image": [
            None if index < missing else _image(100 * len(split) + index)
            for index in range(count)
        ],
        "question": [f"Which {split} option {index}?" for index in range(count)],
        "choices": [["a", "b", "c"] for _ in range(count)],
        "answer": [index % 3 for index in range(count)],
        "hint": ["" if index % 2 else f"Hint {index}" for index in range(count)],
        "subject": ["natural science" for _ in range(count)],
        "topic": ["biology" for _ in range(count)],
        "category": ["cat" for _ in range(count)],
        "skill": ["skill" for _ in range(count)],
        "grade": ["grade3" for _ in range(count)],
    }
    features = Features(
        {
            "image": Image(),
            "question": Value("string"),
            "choices": ListFeature(Value("string")),
            "answer": Value("int8"),
            **{
                key: Value("string")
                for key in ("hint", "subject", "topic", "category", "skill", "grade")
            },
        }
    )
    return Dataset.from_dict(rows, features=features)


def _cvbench_rows(count: int) -> Dataset:
    rows = {
        "image": [_image(200 + index) for index in range(count)],
        "question": [f"How many {index}?" for index in range(count)],
        "choices": [["3", "2", "1", "0"] for _ in range(count)],
        "answer": ["(C)" for _ in range(count)],
        "task": ["Count" for _ in range(count)],
        "type": ["2D" for _ in range(count)],
        "source": ["ADE20K" for _ in range(count)],
        "filename": [f"img/{index}.png" for index in range(count)],
    }
    features = Features(
        {
            "image": Image(),
            "choices": ListFeature(Value("string")),
            **{
                key: Value("string")
                for key in ("question", "answer", "task", "type", "source", "filename")
            },
        }
    )
    return Dataset.from_dict(rows, features=features)


def _fake_load_dataset(dataset, config, split, revision):
    sources = {
        source.dataset: source for source in MODULE.TRAIN_SOURCES + MODULE.TEST_SOURCES
    }
    assert revision == sources[dataset].revision
    if dataset == MODULE.AOKVQA.dataset:
        return _aokvqa_rows(split, TRAIN + 1 if split == "train" else DEV)
    if dataset == MODULE.SCIENCEQA.dataset:
        return _scienceqa_rows(split, TRAIN + 1 if split == "train" else DEV + 1)
    return _cvbench_rows(4)


def test_aokvqa_normalizer_maps_question_and_gold():
    row = {
        "question_id": "q1",
        "question": "  What   is this? ",
        "choices": ["cat", "dog "],
        "correct_choice_idx": 1,
    }
    record = MODULE._normalize_aokvqa(row, 7, "train")
    assert record["source"] == "public_image.aokvqa"
    assert record["family"] == record["group"] == record["id"] == "q1"
    assert record["state"] == {"question": "What is this?"}
    assert record["questions"]["answer"]["type"] == "choice"
    assert record["questions"]["answer"]["options"] == ["cat", "dog"]
    assert record["labels"]["answer"] == {"kind": "hard", "gold_idx": 1}
    provenance = record["source_metadata"]["provenance"]
    assert provenance["dataset"] == MODULE.AOKVQA.dataset
    assert provenance["revision"] == MODULE.AOKVQA.revision
    assert provenance["split"] == "train"
    assert provenance["source_id"] == "q1"
    assert "license" in provenance


def test_normalizers_drop_colliding_empty_or_invalid_choices():
    base = {"question_id": "q", "question": "Q?", "correct_choice_idx": 0}
    assert MODULE._normalize_aokvqa({**base, "choices": ["a", "a"]}, 0, "train") is None
    assert MODULE._normalize_aokvqa({**base, "choices": ["a", " "]}, 0, "train") is None
    assert MODULE._normalize_aokvqa({**base, "choices": []}, 0, "train") is None
    assert (
        MODULE._normalize_aokvqa(
            {**base, "choices": ["a", "b"], "correct_choice_idx": 2}, 0, "train"
        )
        is None
    )
    assert (
        MODULE._normalize_aokvqa(
            {**base, "choices": ["a", "b"], "correct_choice_idx": None}, 0, "train"
        )
        is None
    )


def test_scienceqa_normalizer_uses_hint_as_context():
    row = {
        "question": "Which is north?",
        "choices": ["Maine", "Texas"],
        "answer": 0,
        "hint": "Look at the map.",
        "subject": "social science",
        "topic": "geography",
        "category": "c",
        "skill": "s",
        "grade": "grade2",
    }
    record = MODULE._normalize_scienceqa(row, 12, "dev")
    assert record["source"] == "public_image.scienceqa"
    assert record["family"] == record["id"] == record["group"] == "validation-12"
    assert record["state"] == {
        "question": "Which is north?",
        "context": "Look at the map.",
    }
    assert record["labels"]["answer"]["gold_idx"] == 0
    assert record["source_metadata"]["provenance"]["split"] == "validation"
    assert record["source_metadata"]["provenance"]["source_id"] == 12
    assert MODULE._normalize_scienceqa({**row, "hint": ""}, 12, "dev")["state"] == {
        "question": "Which is north?"
    }


def test_cvbench_normalizer_maps_letter_answer_to_index():
    row = {
        "question": "How many?",
        "choices": ["3", "2", "1", "0"],
        "answer": "(C)",
        "task": "Count",
        "type": "2D",
        "source": "ADE20K",
        "filename": "img/x.png",
    }
    record = MODULE._normalize_cvbench(row, 3, "test")
    assert record["source"] == "public_image.cvbench"
    assert record["family"] == "Count"
    assert record["id"] == record["group"] == "test-3"
    assert record["labels"]["answer"]["gold_idx"] == 2
    assert record["source_metadata"]["attributes"]["type"] == "2D"
    assert MODULE._normalize_cvbench({**row, "answer": "(E)"}, 3, "test") is None
    assert MODULE._normalize_cvbench({**row, "answer": "C"}, 3, "test") is None


def test_image_payload_keeps_png_jpeg_and_reencodes_other_formats():
    png = _png(1)
    assert MODULE._image_payload({"bytes": png, "path": None}) == (png, "png")
    jpeg_buffer = io.BytesIO()
    PILImage.new("RGB", (4, 4)).save(jpeg_buffer, format="JPEG")
    assert MODULE._image_payload({"bytes": jpeg_buffer.getvalue(), "path": None}) == (
        jpeg_buffer.getvalue(),
        "jpg",
    )
    bmp_buffer = io.BytesIO()
    PILImage.new("RGB", (4, 4), (1, 2, 3)).save(bmp_buffer, format="BMP")
    data, extension = MODULE._image_payload(
        {"bytes": bmp_buffer.getvalue(), "path": None}
    )
    assert extension == "png" and data.startswith(MODULE.PNG_MAGIC)
    with pytest.raises(ValueError, match="no bytes"):
        MODULE._image_payload({"bytes": None, "path": None})


def test_image_store_writes_duplicates_once(tmp_path):
    store = MODULE.ImageStore(tmp_path)
    first = store.put({"bytes": _png(5), "path": None})
    second = store.put({"bytes": _png(5), "path": None})
    third = store.put({"bytes": _png(6), "path": None})
    assert first == second != third
    assert first == f"images/{hashlib.sha256(_png(5)).hexdigest()}.png"
    assert sorted(path.name for path in (tmp_path / "images").iterdir()) == sorted(
        [first.split("/")[1], third.split("/")[1]]
    )
    assert store.count == 2
    assert store.total_bytes == len(_png(5)) + len(_png(6))


def test_image_store_rewrites_a_truncated_leftover(tmp_path):
    data = _png(5)
    digest = hashlib.sha256(data).hexdigest()
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / f"{digest}.png").write_bytes(data[:7])
    relative = MODULE.ImageStore(tmp_path).put_payload(data, "png")
    assert (tmp_path / relative).read_bytes() == data
    assert [path.name for path in (tmp_path / "images").iterdir()] == [f"{digest}.png"]


@pytest.mark.parametrize(
    "flag", ["--train-per-source", "--dev-per-source", "--test-cap"]
)
def test_cli_rejects_negative_counts(tmp_path, monkeypatch, flag):
    monkeypatch.setattr(MODULE, "build", lambda *args, **kwargs: None)
    with pytest.raises(SystemExit):
        MODULE.main([str(tmp_path), flag, "-1"])
    MODULE.main([str(tmp_path), flag, "0"])


def _build(tmp_path, monkeypatch, name):
    monkeypatch.setattr(MODULE, "load_dataset", _fake_load_dataset)
    return MODULE.build(
        tmp_path / name, train_per_source=TRAIN, dev_per_source=DEV, test_cap=TEST
    )


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_build_is_deterministic_and_drops_imageless_rows(tmp_path, monkeypatch):
    first = _build(tmp_path, monkeypatch, "first")
    second = _build(tmp_path, monkeypatch, "second")
    assert first == second
    assert first["files"]["train"]["lines"] == 2 * TRAIN
    assert first["files"]["dev"]["lines"] == 2 * DEV
    assert first["files"]["test"]["lines"] == 4
    assert first["dropped"]["train"]["scienceqa"] == {"no_image": 1}
    assert first["dropped"]["dev"]["scienceqa"] == {"no_image": 1}
    assert first["counts"] == {
        "test": {"cvbench": 4},
        "train": {"aokvqa": TRAIN, "scienceqa": TRAIN},
        "dev": {"aokvqa": DEV, "scienceqa": DEV},
    }
    out = tmp_path / "first"
    for split in ("train", "dev", "test"):
        assert (
            first["files"][split]["sha256"]
            == hashlib.sha256((out / f"{split}.jsonl").read_bytes()).hexdigest()
        )
    train = _rows(out / "train.jsonl")
    aokvqa_ids = [row["id"] for row in train if row["source"] == "public_image.aokvqa"]
    expected = sorted(
        (f"train-q{index}" for index in range(TRAIN + 1)),
        key=lambda record_id: MODULE._rank("public_image.aokvqa", "train", record_id),
    )[:TRAIN]
    assert aokvqa_ids == expected
    for row in train + _rows(out / "dev.jsonl") + _rows(out / "test.jsonl"):
        assert len(row["images"]) == 1
        relative = row["images"][0]
        assert relative.startswith("images/") and (out / relative).is_file()
        assert (
            relative.split("/")[1].split(".")[0]
            == hashlib.sha256((out / relative).read_bytes()).hexdigest()
        )
    assert first["images"]["count"] == len(list((out / "images").iterdir()))
    assert first["images"]["total_bytes"] == sum(
        path.stat().st_size for path in (out / "images").iterdir()
    )
    assert first["selection"] == {
        "algorithm": "sha256(seed\\0source\\0split\\0id), ascending",
        "seed": MODULE.SEED,
        "codebook": "vendored26",
        "train_per_source": TRAIN,
        "dev_per_source": DEV,
        "test_cap": TEST,
    }
    assert set(first["sources"]["train_dev"]) == {"aokvqa", "scienceqa"}
    assert first["sources"]["test"]["cvbench"]["revision"] == MODULE.CVBENCH.revision
    cvbench_license = first["sources"]["test"]["cvbench"]["license_evidence"]
    assert cvbench_license["dataset_card_declared"] == "apache-2.0"
    assert "Omni3D" in cvbench_license["images"]
    aokvqa_license = first["sources"]["train_dev"]["aokvqa"]["license_evidence"]
    assert "Flickr Terms of Use" in aokvqa_license["images"]
    assert "images" in first["sources"]["train_dev"]["scienceqa"]["license_evidence"]
    for name in ("A-OKVQA", "ScienceQA", "CV-Bench"):
        assert any(name in flag for flag in first["license_flags"])


def test_build_reuses_one_file_for_duplicate_images(tmp_path, monkeypatch):
    def load(dataset, config, split, revision):
        rows = _fake_load_dataset(dataset, config, split, revision)
        if dataset == MODULE.AOKVQA.dataset and split == "train":
            rows = (
                rows.cast_column("image", Image(decode=False))
                .map(lambda row: {"image": _image(999)})
                .cast_column("image", Image())
            )
        return rows

    monkeypatch.setattr(MODULE, "load_dataset", load)
    contract = MODULE.build(
        tmp_path / "out", train_per_source=TRAIN, dev_per_source=DEV, test_cap=TEST
    )
    train = _rows(tmp_path / "out" / "train.jsonl")
    paths = {
        row["images"][0] for row in train if row["source"] == "public_image.aokvqa"
    }
    assert len(paths) == 1
    assert contract["images"]["count"] == 1 + DEV + TRAIN + DEV + 4


def _selected_train_index() -> int:
    selected = min(
        (f"train-q{index}" for index in range(TRAIN + 1)),
        key=lambda record_id: MODULE._rank("public_image.aokvqa", "train", record_id),
    )
    return int(selected.rsplit("q", 1)[1])


def test_build_keeps_dev_rows_sharing_only_a_templated_train_state(
    tmp_path, monkeypatch
):
    question = f"What is train {_selected_train_index()} doing?"

    def load(dataset, config, split, revision):
        rows = _fake_load_dataset(dataset, config, split, revision)
        if dataset == MODULE.AOKVQA.dataset and split == "validation":
            rows = rows.map(lambda row: {"question": question})
        return rows

    monkeypatch.setattr(MODULE, "load_dataset", load)
    contract = MODULE.build(
        tmp_path / "out", train_per_source=TRAIN, dev_per_source=DEV, test_cap=TEST
    )
    assert contract["counts"]["dev"]["aokvqa"] == DEV
    assert contract["dropped"]["dev"]["aokvqa"] == {}


def test_build_drops_dev_rows_matching_a_train_state_and_image(tmp_path, monkeypatch):
    selected = _selected_train_index()
    question = f"What is train {selected} doing?"
    train_image = _image(selected)

    def load(dataset, config, split, revision):
        rows = _fake_load_dataset(dataset, config, split, revision)
        if dataset == MODULE.AOKVQA.dataset and split == "validation":
            rows = rows.cast_column("image", Image(decode=False)).map(
                lambda row, index: (
                    {"image": train_image, "question": question} if index == 0 else row
                ),
                with_indices=True,
            )
            rows = rows.cast_column("image", Image())
        return rows

    monkeypatch.setattr(MODULE, "load_dataset", load)
    contract = MODULE.build(
        tmp_path / "out", train_per_source=TRAIN, dev_per_source=0, test_cap=TEST
    )
    assert contract["dropped"]["dev"]["aokvqa"] == {"state_overlap": 1}
    assert contract["scanned"]["dev"]["aokvqa"] == {
        "split_rows": DEV,
        "ranked_scanned": DEV,
    }


def test_build_excludes_test_images_and_states_from_train(tmp_path, monkeypatch):
    def load(dataset, config, split, revision):
        rows = _fake_load_dataset(dataset, config, split, revision)
        if dataset == MODULE.AOKVQA.dataset and split == "train":
            rows = rows.cast_column("image", Image(decode=False)).map(
                lambda row, index: (
                    {"image": _image(200)}
                    if index == 0
                    else {"image": _image(201), "question": "How many 1?"}
                    if index == 1
                    else {"question": "How many 2?"}
                    if index == 2
                    else row
                ),
                with_indices=True,
            )
            rows = rows.cast_column("image", Image())
        return rows

    monkeypatch.setattr(MODULE, "load_dataset", load)
    contract = MODULE.build(
        tmp_path / "out", train_per_source=0, dev_per_source=DEV, test_cap=TEST
    )
    assert contract["dropped"]["train"]["aokvqa"] == {
        "image_overlap": 1,
        "state_overlap": 1,
    }
    assert contract["counts"]["train"]["aokvqa"] == TRAIN + 1 - 2
    out = tmp_path / "out"
    train = _rows(out / "train.jsonl")
    test = _rows(out / "test.jsonl")
    assert not {row["images"][0] for row in train} & {row["images"][0] for row in test}
    assert any(row["state"] == {"question": "How many 2?"} for row in train)


def test_build_rejects_short_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "load_dataset", _fake_load_dataset)
    with pytest.raises(ValueError, match="aokvqa train has 4 usable rows; need 5"):
        MODULE.build(tmp_path / "out", train_per_source=5, dev_per_source=DEV)


def test_emitted_row_round_trips_through_normalize_jsonl(tmp_path, monkeypatch):
    from axolotl.integrations.diffusion_decision.adapters.jsonl import normalize_jsonl

    _build(tmp_path, monkeypatch, "out")
    row = _rows(tmp_path / "out" / "train.jsonl")[0]
    result = normalize_jsonl(row)
    assert result["images"] == row["images"]
    assert result["labels"]["answer"]["gold_idx"] == row["labels"]["answer"]["gold_idx"]


def test_build_drops_dev_rows_whose_image_is_in_train(tmp_path, monkeypatch):
    selected = min(
        (f"train-q{index}" for index in range(TRAIN + 1)),
        key=lambda record_id: MODULE._rank("public_image.aokvqa", "train", record_id),
    )
    train_image = _image(int(selected.rsplit("q", 1)[1]))

    def load(dataset, config, split, revision):
        rows = _fake_load_dataset(dataset, config, split, revision)
        if dataset == MODULE.AOKVQA.dataset and split == "validation":
            rows = rows.cast_column("image", Image(decode=False)).map(
                lambda row, index: {
                    "image": train_image if index == 0 else row["image"]
                },
                with_indices=True,
            )
            rows = rows.cast_column("image", Image())
        return rows

    monkeypatch.setattr(MODULE, "load_dataset", load)
    contract = MODULE.build(
        tmp_path / "out", train_per_source=TRAIN, dev_per_source=0, test_cap=TEST
    )
    assert contract["dropped"]["dev"]["aokvqa"] == {"image_overlap": 1}
    assert contract["counts"]["dev"]["aokvqa"] == DEV - 1
    out = tmp_path / "out"
    digests = {
        split: {
            MODULE.image_digest(row["images"][0])
            for row in _rows(out / f"{split}.jsonl")
        }
        for split in ("train", "dev", "test")
    }
    assert not digests["train"] & digests["dev"]
    assert not (digests["train"] | digests["dev"]) & digests["test"]


def test_image_isolation_assert_rejects_shared_digests():
    shared = {"images": ["images/abc.png"]}
    MODULE._assert_image_isolation({"train": [shared, shared], "dev": []})
    with pytest.raises(ValueError, match="image abc overlaps splits 'train' and 'dev'"):
        MODULE._assert_image_isolation({"train": [shared], "dev": [shared]})


def test_built_scienceqa_train_rows_survive_plugin_decontamination(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from axolotl.integrations.diffusion_decision import datasets
    from axolotl.model_support import DiffusionLayout

    _build(tmp_path, monkeypatch, "out")
    out = tmp_path / "out"
    monkeypatch.setattr(datasets, "load_tokenizer", lambda _cfg: range(131072))
    monkeypatch.setattr(
        datasets,
        "require_diffusion_spec",
        lambda _cfg: SimpleNamespace(
            max_canvas=128,
            noise=datasets.DiffusionNoise.ABSORBING,
            layout=DiffusionLayout.FULL_SEQUENCE,
        ),
    )
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

    def entry(split):
        return {
            "path": "json",
            "data_files": str(out / f"{split}.jsonl"),
            "split": split,
            "type": "diffusion_decision.jsonl",
        }

    cfg = {
        "seed": 42,
        "micro_batch_size": 2,
        "datasets": [entry("train")],
        "test_datasets": [entry("dev")],
        "diffusion_decision": {"labels": {}, "mixture": {"temperature": 1.0}},
    }
    result = datasets.load_decision_datasets(cfg)
    manifest = result.train_dataset.manifest
    assert manifest["dropped"]["family_overlap"] == 0
    assert manifest["dropped"]["state_overlap"] == 0
    built = {row["id"] for row in _rows(out / "train.jsonl")}
    retained = {
        record_id
        for row in result.train_dataset
        for record_id in row["record"].get("grouped_record_ids")
        or (row["record"]["id"],)
    }
    assert retained == built
    scienceqa = {
        row["id"]
        for row in _rows(out / "train.jsonl")
        if row["source"] == "public_image.scienceqa"
    }
    assert len(scienceqa) == TRAIN and scienceqa <= retained
