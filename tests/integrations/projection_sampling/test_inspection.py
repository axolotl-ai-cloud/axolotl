"""Readable export retention, atomic publication, and distributed writers."""

import json
from pathlib import Path

import pytest

from axolotl.integrations.projection_sampling.args import ProjectionSamplingConfig
from axolotl.integrations.projection_sampling.inspection import export_dataset
from axolotl.integrations.projection_sampling.plugin import (
    ProjectionSamplingPlugin,
    cache_path,
)
from axolotl.utils.dict import DictDefault


def test_export_retains_tools_and_skip_metadata_without_token_arrays(tmp_path):
    row = {
        "messages": [{"role": "assistant", "content": "rewritten"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "input_ids": [1, 2],
        "labels": [-100, 2],
        "attention_mask": [1, 1],
        "sampling": [
            {"message_index": 0, "sampled_token_ids": [2], "expert_response": "expert"},
            {"message_index": 1, "skipped": "partial_training_mask"},
        ],
    }
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps(row) + "\n")
    destination = export_dataset(cache, str(tmp_path / "output"), seed=7)
    readable = json.loads(destination.read_text())
    assert readable["messages"] == row["messages"]
    assert readable["tools"] == row["tools"]
    assert readable["sampling"][-1] == row["sampling"][-1]
    assert not {"input_ids", "labels", "attention_mask"} & readable.keys()
    assert "sampled_token_ids" not in readable["sampling"][0]
    assert json.loads(cache.read_text()) == row


def test_failed_export_keeps_previous_complete_dataset(tmp_path):
    cache = tmp_path / "cache.jsonl"
    row = {
        "prompt": "question",
        "response": "rewritten",
        "expert_response": "expert",
        "sampling": {"finished": True},
    }
    cache.write_text(json.dumps(row) + "\n")
    destination = export_dataset(cache, str(tmp_path / "output"), seed=7)
    previous = destination.read_bytes()
    cache.write_text(json.dumps(row) + "\ninvalid json\n")
    with pytest.raises(json.JSONDecodeError):
        export_dataset(cache, str(tmp_path / "output"), seed=8)
    assert destination.read_bytes() == previous
    assert list(destination.parent.glob("*.jsonl")) == [destination]


def test_training_rank_one_does_not_export(tmp_path, monkeypatch):
    import axolotl.common.datasets as common
    import axolotl.integrations.projection_sampling.plugin as plugin_module

    cfg = DictDefault(
        datasets=[{"path": "expert.jsonl"}],
        seed=7,
        output_dir=str(tmp_path / "output"),
        projection_sampling={"cache_dir": str(tmp_path / "cache")},
    )
    cache = cache_path(
        cfg, ProjectionSamplingConfig.model_validate(cfg.projection_sampling)
    )
    cache.parent.mkdir()
    cache.write_text("{}\n")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setattr(
        plugin_module,
        "export_dataset",
        lambda *args, **kwargs: pytest.fail("non-main writer"),
    )
    monkeypatch.setattr(common, "load_datasets", lambda **kwargs: "prepared")
    assert ProjectionSamplingPlugin().load_datasets(cfg) == "prepared"
    assert not Path(cfg.output_dir).exists()
