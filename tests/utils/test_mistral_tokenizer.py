"""Tests for the mistral-common tokenizer wrapper."""

from unittest.mock import patch

import pytest

from axolotl.utils.mistral.mistral_tokenizer import HFMistralTokenizer


@pytest.fixture(name="captured_init")
def captured_init_fixture():
    """Capture the kwargs `from_pretrained` builds without constructing a tokenizer."""
    captured: dict = {}

    def fake_init(self, **kwargs):  # pylint: disable=unused-argument
        captured.update(kwargs)

    with patch.object(HFMistralTokenizer, "__init__", fake_init):
        yield captured


class TestHFMistralTokenizerFromPretrained:
    """Resolution of `pretrained_model_name_or_path` to a tokenizer file."""

    def test_local_dir_resolves_tokenizer_file(self, tmp_path, captured_init):
        """A local dir (e.g. merge-lora output) is read without hitting the Hub."""
        (tmp_path / "tekken.json").write_text("{}")
        (tmp_path / "config.json").write_text("{}")

        with patch(
            "axolotl.utils.mistral.mistral_tokenizer.download_tokenizer_from_hf_hub"
        ) as mock_download:
            HFMistralTokenizer.from_pretrained(str(tmp_path))

        mock_download.assert_not_called()
        assert captured_init["tokenizer_path"] == str(tmp_path / "tekken.json")
        assert captured_init["name_or_path"] == str(tmp_path)

    def test_local_file_is_used_directly(self, tmp_path, captured_init):
        tokenizer_file = tmp_path / "tekken.json"
        tokenizer_file.write_text("{}")

        with patch(
            "axolotl.utils.mistral.mistral_tokenizer.download_tokenizer_from_hf_hub"
        ) as mock_download:
            HFMistralTokenizer.from_pretrained(str(tokenizer_file))

        mock_download.assert_not_called()
        assert captured_init["tokenizer_path"] == str(tokenizer_file)

    def test_repo_id_downloads_from_hub(self, captured_init):
        with patch(
            "axolotl.utils.mistral.mistral_tokenizer.download_tokenizer_from_hf_hub",
            return_value="/cache/tekken.json",
        ) as mock_download:
            HFMistralTokenizer.from_pretrained("mistralai/Shieldstral-1.0-3B")

        mock_download.assert_called_once()
        assert captured_init["tokenizer_path"] == "/cache/tekken.json"
