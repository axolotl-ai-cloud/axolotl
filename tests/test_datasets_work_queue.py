"""Tests for load-balanced multiprocess tokenization."""

import pytest
from datasets import Dataset

from axolotl.datasets_work_queue import tokenize_with_work_queue


class FakeStrategy:
    """Minimal stand-in for a PromptTokenizingStrategy."""

    supports_batched = False

    def tokenize_prompt(self, row):
        n = len(row["text"])
        return {
            "input_ids": list(range(n)),
            "attention_mask": [1] * n,
            "labels": list(range(n)),
        }


class EmptyFirstRowStrategy(FakeStrategy):
    """Row 0 tokenizes to nothing, as an empty prompt does."""

    def tokenize_prompt(self, row):
        if row["text"] == "":
            return {"input_ids": [], "attention_mask": [], "labels": []}
        return super().tokenize_prompt(row)


class DroppedFirstRowStrategy(FakeStrategy):
    """Row 0 is dropped entirely, as chat_template does when a batch empties."""

    def tokenize_prompt(self, row):
        if row["text"] == "":
            return {}
        return super().tokenize_prompt(row)


class VaryingKeysStrategy(FakeStrategy):
    """Every row but the first emits an extra column."""

    def tokenize_prompt(self, row):
        out = super().tokenize_prompt(row)
        if row["text"] != "row0":
            out["position_ids"] = [0]
        return out


@pytest.fixture(name="dataset")
def dataset_fixture():
    # Large enough to span several chunks at num_proc=2.
    return Dataset.from_dict({"text": [f"row{i}" for i in range(200)]})


def test_matches_map(dataset):
    expected = dataset.map(
        FakeStrategy().tokenize_prompt,
        num_proc=2,
        remove_columns=dataset.column_names,
    )
    actual = tokenize_with_work_queue(
        FakeStrategy(), dataset, num_proc=2, keep_in_memory=True
    )

    assert actual.column_names == expected.column_names
    assert actual.to_dict() == expected.to_dict()


def test_empty_first_row_does_not_poison_schema():
    """An empty first row used to fix the Arrow schema to null and crash."""
    dataset = Dataset.from_dict({"text": [""] + [f"row{i}" for i in range(199)]})

    result = tokenize_with_work_queue(
        EmptyFirstRowStrategy(), dataset, num_proc=2, keep_in_memory=True
    )

    assert result.num_rows == 200
    assert result[0]["input_ids"] == []
    assert result[1]["input_ids"] == list(range(len("row1")))


def test_dropped_row_does_not_drop_its_chunk():
    """A row returning {} used to delete every other row in its chunk."""
    dataset = Dataset.from_dict({"text": [""] + [f"row{i}" for i in range(199)]})

    result = tokenize_with_work_queue(
        DroppedFirstRowStrategy(), dataset, num_proc=2, keep_in_memory=True
    )

    assert result.num_rows == 199


def test_inconsistent_keys_raise(dataset):
    """Varying columns used to be silently dropped or raise a bare KeyError."""
    with pytest.raises(ValueError, match="same set of columns"):
        tokenize_with_work_queue(
            VaryingKeysStrategy(), dataset, num_proc=2, keep_in_memory=True
        )


def test_all_rows_dropped_raises_actionable_error(dataset):
    class DropEverything(FakeStrategy):
        def tokenize_prompt(self, row):
            return {}

    with pytest.raises(ValueError, match="produced no rows"):
        tokenize_with_work_queue(
            DropEverything(), dataset, num_proc=2, keep_in_memory=True
        )


def test_cache_round_trip(dataset, tmp_path):
    """Second call must load from the Arrow cache and match the first."""
    dataset = dataset.map(lambda row: row, cache_file_name=str(tmp_path / "base.arrow"))

    first = tokenize_with_work_queue(FakeStrategy(), dataset, num_proc=2)
    second = tokenize_with_work_queue(FakeStrategy(), dataset, num_proc=2)

    assert first.to_dict() == second.to_dict()
    assert second.cache_files
