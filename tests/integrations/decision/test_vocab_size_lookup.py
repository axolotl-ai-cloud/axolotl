"""Avoid expensive tokenizer vocabulary scans when model metadata is available."""

from types import SimpleNamespace

from axolotl.integrations.decision import datasets


def test_canvas_row_uses_model_vocabulary_without_counting_tokenizer(monkeypatch):
    class Tokenizer:
        eos_token_id = 11
        pad_token_id = 11

        def __len__(self):
            raise AssertionError("unexpected full tokenizer vocabulary count")

    captured = {}

    def build(_tokenizer, _record, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(prompt_ids=(1,), canvas_ids=(100,) * 128)

    monkeypatch.setattr(datasets, "build_decision_canvas", build)
    monkeypatch.setattr(datasets, "permute_record", lambda record, **kwargs: record)
    monkeypatch.setattr(
        datasets, "decision_example_from_canvas", lambda *args, **kwargs: None
    )
    cfg = {
        "model_config": {"vocab_size": 131072},
        "diffusion": {"mask_token_id": 100},
        "decision": {},
    }
    row = datasets._canvas_row(
        Tokenizer(),
        {"source": "test"},
        cfg,
        1.0,
        spec=SimpleNamespace(max_canvas=128, noise=datasets.DiffusionNoise.ABSORBING),
    )
    assert captured["vocab_size"] == 131072
    assert row["length"] == 129


def test_missing_model_metadata_counts_vocabulary_once_per_pool(monkeypatch):
    class Tokenizer:
        calls = 0

        def __len__(self):
            self.calls += 1
            return 131072

    tokenizer = Tokenizer()
    seen = []

    def build_row(_tokenizer, record, cfg, weight, **kwargs):
        seen.append(kwargs["vocab_size"])
        return record

    monkeypatch.setattr(datasets, "_canvas_row", build_row)
    rows = [{"source": "test", "id": str(index)} for index in range(20)]
    actual, dropped = datasets._canvas_rows(tokenizer, rows, {}, {}, None)
    assert actual == rows
    assert dropped == 0
    assert seen == [131072] * 20
    assert tokenizer.calls == 1
