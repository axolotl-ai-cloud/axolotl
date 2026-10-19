"""Tests that the buffered MM packing loaders resolve every supported dataset source."""

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from axolotl.utils.data.sft import (
    _load_skip_prepare_mm_dataset,
    _load_streaming_mm_dataset,
)
from axolotl.utils.dict import DictDefault


@pytest.fixture(name="local_parquet")
def fixture_local_parquet(tmp_path):
    path = str(tmp_path / "train.parquet")
    Dataset.from_dict(
        {"messages": [[{"role": "user", "content": "hi"}]] * 2}
    ).to_parquet(path)
    return path


def _cfg():
    return DictDefault({"hf_use_auth_token": None})


def _first_row(dataset):
    return next(iter(dataset))


class TestBufferedLoaderPathForms:
    """Local file paths must work like they do in the standard loaders."""

    def load(self, loader, dataset_config):
        with patch(
            "axolotl.utils.data.mm_streaming.build_streaming_mm_dataset",
            side_effect=lambda raw, cfg, tok, proc: raw,
        ):
            return loader(DictDefault(dataset_config), _cfg(), MagicMock(), MagicMock())

    def test_skip_prepare_local_file_path(self, local_parquet):
        raw = self.load(
            _load_skip_prepare_mm_dataset,
            {"path": local_parquet, "type": "chat_template", "split": "train"},
        )
        assert "messages" in _first_row(raw)

    def test_streaming_local_file_path(self, local_parquet):
        raw = self.load(
            _load_streaming_mm_dataset,
            {"path": local_parquet, "type": "chat_template", "split": "train"},
        )
        assert "messages" in _first_row(raw)

    def test_local_directory_path(self, local_parquet, tmp_path):
        # directory form resolves through the same local-path branch
        raw = self.load(
            _load_skip_prepare_mm_dataset,
            {"path": str(tmp_path), "type": "chat_template", "split": "train"},
        )
        assert "messages" in _first_row(raw)
