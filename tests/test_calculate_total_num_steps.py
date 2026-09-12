"""Tests for token counting in calculate_total_num_steps"""

import math

import pytest
from datasets import Dataset, Features, LargeList, Sequence, Value, load_from_disk

from axolotl.utils.dict import DictDefault
from axolotl.utils.trainer import calculate_total_num_steps

NUM_ROWS = 3000

FEATURES = Features(
    {
        "input_ids": Sequence(Value("int64")),
        "labels": Sequence(Value("int64")),
    }
)


def build_columns(num_rows):
    input_ids = []
    labels = []
    for i in range(num_rows):
        seq_len = 5 + (i % 50)
        input_ids.append(list(range(seq_len)))
        labels.append([-100 if j < seq_len // 2 else j for j in range(seq_len)])
    return {"input_ids": input_ids, "labels": labels}


@pytest.fixture(name="columns")
def fixture_columns():
    return build_columns(NUM_ROWS)


@pytest.fixture(name="expected_counts")
def fixture_expected_counts(columns):
    total_tokens = sum(len(row) for row in columns["input_ids"])
    supervised_tokens = sum(
        1 for row in columns["labels"] for token in row if token != -100
    )
    return total_tokens, supervised_tokens


def make_cfg():
    return DictDefault(
        {
            "num_epochs": 1,
            "batch_size": 2,
            "micro_batch_size": 1,
        }
    )


def assert_counts(dataset, expected_counts):
    cfg = make_cfg()
    total_num_steps = calculate_total_num_steps(cfg, dataset)
    expected_tokens, expected_supervised = expected_counts
    assert cfg.total_num_tokens == expected_tokens
    assert cfg.total_supervised_tokens == expected_supervised
    assert total_num_steps == math.ceil(len(dataset) / cfg.batch_size)


class TestCalculateTotalNumSteps:
    """
    Test token counting against hand-computed ground truth across Arrow layouts
    """

    def test_in_memory_single_chunk(self, columns, expected_counts):
        dataset = Dataset.from_dict(columns, features=FEATURES)
        # a single chunk larger than the to_batches chunksize is the layout
        # where ListArray.values (vs .flatten()) silently over-counts
        assert dataset.data.column("labels").num_chunks == 1
        assert len(dataset) > 1024
        assert_counts(dataset, expected_counts)

    def test_disk_round_trip(self, columns, expected_counts, tmp_path):
        dataset = Dataset.from_dict(columns, features=FEATURES)
        dataset.save_to_disk(str(tmp_path / "ds"))
        dataset = load_from_disk(str(tmp_path / "ds"))
        assert_counts(dataset, expected_counts)

    def test_large_list(self, columns, expected_counts):
        dataset = Dataset.from_dict(columns, features=FEATURES).cast(
            Features(
                {
                    "input_ids": LargeList(Value("int64")),
                    "labels": LargeList(Value("int64")),
                }
            )
        )
        assert_counts(dataset, expected_counts)

    def test_length_column_takes_precedence(self, columns, expected_counts):
        columns = dict(columns)
        # offset lengths by one so the result proves which branch ran
        columns["length"] = [len(row) + 1 for row in columns["input_ids"]]
        dataset = Dataset.from_dict(columns, features=None)
        cfg = make_cfg()
        calculate_total_num_steps(cfg, dataset)
        expected_tokens, expected_supervised = expected_counts
        assert cfg.total_num_tokens == expected_tokens + NUM_ROWS
        assert cfg.total_supervised_tokens == expected_supervised

    def test_empty_dataset(self):
        dataset = Dataset.from_dict({"input_ids": [], "labels": []}, features=FEATURES)
        cfg = make_cfg()
        total_num_steps = calculate_total_num_steps(cfg, dataset)
        assert not cfg.total_num_tokens
        assert not cfg.total_supervised_tokens
        assert total_num_steps == 0

    def test_update_false_does_not_mutate_cfg(self, columns):
        dataset = Dataset.from_dict(columns, features=FEATURES)
        cfg = make_cfg()
        calculate_total_num_steps(cfg, dataset, update=False)
        assert cfg.total_num_tokens is None
        assert cfg.total_supervised_tokens is None
