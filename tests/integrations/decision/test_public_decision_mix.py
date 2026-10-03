import hashlib
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from datasets import Dataset

SCRIPT = (
    Path(__file__).parents[3]
    / "src/axolotl/integrations/decision/scripts/build_public_decision_mix.py"
)
SPEC = spec_from_file_location("build_public_decision_mix", SCRIPT)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _raw(config: str, split: str, index: int) -> dict:
    return {
        "id": f"{config}:{split}:{index}",
        "state": json.dumps({"config": config, "split": split, "index": index}),
        "questions": json.dumps(
            {"ok": {"type": "noul", "instructions": "", "criteria": {}}}
        ),
        "answers": json.dumps({"ok": {"noul": 1.0}}),
    }


def test_build_public_mix_is_deterministic_and_decontaminates_heldouts(
    tmp_path, monkeypatch
):
    def fake_load_dataset(dataset, name, split, revision):
        assert (dataset, revision) == (MODULE.DATASET, MODULE.REVISION)
        count = (
            MODULE.TRAIN_PER_CONFIG + 1 if split == "train" else MODULE.DEV_PER_CONFIG
        )
        return Dataset.from_list([_raw(name, split, index) for index in range(count)])

    monkeypatch.setattr(MODULE, "load_dataset", fake_load_dataset)
    protected = tmp_path / "protected.jsonl"
    colliding_id = min(
        (f"arithmetic:train:{index}" for index in range(MODULE.TRAIN_PER_CONFIG + 1)),
        key=lambda record_id: hashlib.sha256(
            f"{MODULE.SEED}\0arithmetic\0{record_id}".encode()
        ).hexdigest(),
    )
    colliding_index = int(colliding_id.rsplit(":", 1)[1])
    protected.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"source": "jev_bench", "family": "heldout", "state": {"x": "j"}},
                {"source": "nimble", "family": "heldout", "state": {"x": "n"}},
                {
                    "source": "jev_bench",
                    "family": "heldout",
                    "state": {
                        "config": "arithmetic",
                        "split": "train",
                        "index": colliding_index,
                    },
                },
            )
        )
        + "\n"
    )
    first = MODULE.build(tmp_path / "first", [protected])
    second = MODULE.build(tmp_path / "second", [protected])

    assert first["files"] == second["files"]
    assert (
        first["files"]["train"]["lines"]
        == len(MODULE.CONFIGS) * MODULE.TRAIN_PER_CONFIG
    )
    assert first["files"]["dev"]["lines"] == len(MODULE.CONFIGS) * MODULE.DEV_PER_CONFIG
    assert (
        first["files"]["train"]["sha256"]
        == hashlib.sha256((tmp_path / "first" / "train.jsonl").read_bytes()).hexdigest()
    )
    assert (
        not (tmp_path / "first" / "train.jsonl").read_text().find('"source":"nimble"')
        >= 0
    )
    train_ids = {
        json.loads(line)["id"]
        for line in (tmp_path / "first" / "train.jsonl").read_text().splitlines()
    }
    assert colliding_id not in train_ids
    assert len(train_ids) == first["files"]["train"]["lines"]


def test_build_public_mix_is_usable_without_protected_rows(tmp_path, monkeypatch):
    def fake_load_dataset(dataset, name, split, revision):
        count = MODULE.TRAIN_PER_CONFIG if split == "train" else MODULE.DEV_PER_CONFIG
        return Dataset.from_list([_raw(name, split, index) for index in range(count)])

    monkeypatch.setattr(MODULE, "load_dataset", fake_load_dataset)
    contract = MODULE.build(tmp_path / "out")
    assert contract["protected_decontamination"]["status"] == "not_requested"


def test_normalize_uses_recipe_codebook(monkeypatch):
    options = {str(index): f"option {index}" for index in range(27)}
    raw = {
        "id": "arithmetic:train:0",
        "state": json.dumps({"x": 1}),
        "questions": json.dumps(
            {"choice": {"type": "choice", "instructions": "", "criteria": options}}
        ),
        "answers": json.dumps(
            {
                "choice": {
                    "probabilities": {
                        name: 0.5 if name in {"0", "1"} else 0.0 for name in options
                    }
                }
            }
        ),
    }

    monkeypatch.setattr(
        MODULE, "load_dataset", lambda *args, **kwargs: Dataset.from_list([raw])
    )
    record = MODULE._normalize("arithmetic", "train")[0]
    assert record["labels"]["choice"]["probs"][0] == 0.5


def test_source_group_overlap_is_rejected():
    record = {"source": "public_procedural.arithmetic", "group": "origin-1"}
    with pytest.raises(ValueError, match="source/group"):
        MODULE._assert_source_group_isolation({"train": [record], "dev": [record]})


def test_build_public_mix_rejects_missing_nimble(tmp_path):
    protected = tmp_path / "protected.jsonl"
    protected.write_text(
        json.dumps({"source": "jev_bench", "family": "x", "state": {}}) + "\n"
    )
    with pytest.raises(ValueError, match="Nimble"):
        MODULE.build(tmp_path / "out", [protected])


def test_build_public_mix_rejects_duplicate_source_ids(tmp_path, monkeypatch):
    def fake_load_dataset(dataset, name, split, revision):
        count = MODULE.TRAIN_PER_CONFIG if split == "train" else MODULE.DEV_PER_CONFIG
        rows = [_raw(name, split, index) for index in range(count)]
        if name == "arithmetic" and split == "train":
            rows[-1]["id"] = rows[0]["id"]
        return Dataset.from_list(rows)

    monkeypatch.setattr(MODULE, "load_dataset", fake_load_dataset)
    protected = tmp_path / "protected.jsonl"
    protected.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"source": "jev_bench", "family": "heldout", "state": {"x": "j"}},
                {"source": "nimble", "family": "heldout", "state": {"x": "n"}},
            )
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="duplicate public source ids"):
        MODULE.build(tmp_path / "out", [protected])
