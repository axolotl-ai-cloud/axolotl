"""Tests for the mistral-common tokenizer wrapper."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.normalize import InstructRequestNormalizerV7
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from pydantic import BaseModel, ValidationError, create_model

from axolotl.utils.mistral.mistral_tokenizer import HFMistralTokenizer


@pytest.mark.parametrize("legacy_continuation", [False, True])
@pytest.mark.parametrize("continue_final_message", [False, True])
def test_normalizer_defaults_match_request_schema(
    legacy_continuation, continue_final_message
):
    normalizer = InstructRequestNormalizerV7.normalizer()
    if legacy_continuation:
        normalizer._instruct_request_class = create_model(
            "LegacyInstructRequest",
            __base__=normalizer._instruct_request_class,
            continue_final_message=(bool, ...),
        )

    class MissingDefaults(BaseModel):
        truncate_at_max_tokens: int | None
        continue_final_message: bool

    def broken_normalize(_request):
        MissingDefaults.model_validate({})

    normalizer.from_chat_completion_request = broken_normalize
    tokenizer = object.__new__(HFMistralTokenizer)
    tokenizer.tokenizer = SimpleNamespace(_instruct_request_normalizer=normalizer)
    tokenizer._patch_instruct_request_normalizer()
    request = SimpleNamespace(
        messages=[UserMessage(content="Hello")],
        tools=None,
        continue_final_message=continue_final_message,
    )
    result = normalizer.from_chat_completion_request(request)
    assert result.messages == request.messages
    assert result.truncate_at_max_tokens is None
    if legacy_continuation:
        assert result.continue_final_message is continue_final_message
    else:
        assert "continue_final_message" not in result.model_dump()


def test_normalizer_preserves_unrelated_validation_errors():
    normalizer = InstructRequestNormalizerV7.normalizer()

    class RequiredMessages(BaseModel):
        messages: list

    def broken_normalize(_request):
        RequiredMessages.model_validate({})

    normalizer.from_chat_completion_request = broken_normalize
    tokenizer = object.__new__(HFMistralTokenizer)
    tokenizer.tokenizer = SimpleNamespace(_instruct_request_normalizer=normalizer)
    tokenizer._patch_instruct_request_normalizer()
    with pytest.raises(ValidationError, match="messages"):
        normalizer.from_chat_completion_request(
            ChatCompletionRequest(messages=[UserMessage(content="Hello")])
        )


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
