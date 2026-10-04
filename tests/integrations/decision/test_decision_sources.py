import json

import pytest

from axolotl.integrations.decision.sources import load_source, normalize_source


def test_local_jsonl_keeps_heterogeneous_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"source":"x","group":"g","state":"s","questions":{"q":{"type":"choice","options":["a","b"]}},"labels":{"q":{"kind":"hard","gold_idx":1}}}\n{"nested":[1,{"x":2}]}\n'
    )
    data = load_source({"path": "json", "data_files": str(path)})
    assert all(isinstance(value, str) for value in data["raw_json"])


def test_invalid_local_jsonl_reports_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{bad}\n")
    with pytest.raises(ValueError, match=r"bad.jsonl:1"):
        load_source({"path": "json", "data_files": str(path)})


def test_normalize_source_keeps_declared_split(tmp_path):
    row = {
        "source": "x",
        "group": "g",
        "state": "s",
        "questions": {"q": {"type": "choice", "options": ["a", "b"]}},
        "labels": {"q": {"kind": "hard", "gold_idx": 1}},
    }
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n")
    result = normalize_source(
        {"path": "json", "data_files": str(path), "type": "jsonl", "split": "train"},
        {"dataset_num_proc": 1},
        True,
    )
    assert result[0]["source"] == "x"


def test_builtin_preserves_split_mapping(monkeypatch):
    from datasets import Dataset

    from axolotl.integrations.decision import sources

    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return Dataset.from_dict({"value": [1]})

    monkeypatch.setattr(sources, "load_dataset", load)
    result = load_source(
        {"path": "parquet", "data_files": {"dev": "dev.parquet"}, "split": "dev"}
    )
    assert calls[0][1]["data_files"] == {"dev": "dev.parquet"}
    assert json.loads(result[0]["raw_json"]) == {"value": 1}


def test_hub_reuses_core_config_and_auth(monkeypatch):
    from datasets import Dataset

    from axolotl.integrations.decision import sources

    calls = []

    def load(cfg, auth):
        calls.append((cfg, auth))
        return Dataset.from_dict({"value": [1]})

    monkeypatch.setattr(sources, "load_dataset_with_config", load)
    load_source(
        {"path": "org/data", "revision": "revision", "split": "test"},
        {"hf_use_auth_token": True},
    )
    assert calls[0][0].revision == "revision"
    assert calls[0][0].split == "test"
    assert calls[0][1] is True


def test_empty_normalization(monkeypatch):
    from datasets import Dataset

    from axolotl.integrations.decision import sources

    monkeypatch.setattr(
        sources, "load_source", lambda *args: Dataset.from_dict({"raw_json": []})
    )
    assert normalize_source({}, {}, True) == []
