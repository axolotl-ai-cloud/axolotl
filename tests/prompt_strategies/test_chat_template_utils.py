"""
Tests for utils in axolotl.utils.chat_templates
"""

import unittest

import pytest
from transformers import AutoTokenizer

from axolotl.utils.chat_templates import (
    _CHAT_TEMPLATES,
    extract_chat_template_args,
    get_chat_template,
)

from tests.hf_offline_utils import enable_hf_offline


@pytest.fixture(name="llama3_tokenizer")
@enable_hf_offline
def fixture_llama3_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")

    return tokenizer


class TestGetChatTemplateUtils:
    """
    Tests the get_chat_template function.
    """

    def test_known_chat_template(self):
        chat_template_str = get_chat_template("llama3")
        assert chat_template_str == _CHAT_TEMPLATES["llama3"]

    def test_invalid_chat_template(self):
        with pytest.raises(ValueError) as exc:
            get_chat_template("invalid_template")
            assert str(exc) == "Template 'invalid_template' not found."

    def test_tokenizer_default_no_tokenizer(self):
        with pytest.raises(ValueError):
            get_chat_template("tokenizer_default", tokenizer=None)

    def test_tokenizer_default_no_chat_template_on_tokenizer(self, llama3_tokenizer):
        with pytest.raises(ValueError):
            get_chat_template("tokenizer_default", tokenizer=llama3_tokenizer)

    def test_tokenizer_default_with_chat_template_on_tokenizer(self, llama3_tokenizer):
        llama3_tokenizer.chat_template = "test_template"
        chat_template_str = get_chat_template(
            "tokenizer_default", tokenizer=llama3_tokenizer
        )
        assert chat_template_str == "test_template"

    def test_tokenizer_default_fallback_no_tokenizer(self):
        with pytest.raises(ValueError):
            get_chat_template("tokenizer_default_fallback_test", tokenizer=None)

    def test_tokenizer_default_fallback_no_chat_template_on_tokenizer(
        self, llama3_tokenizer
    ):
        chat_template_str = get_chat_template(
            "tokenizer_default_fallback_chatml", tokenizer=llama3_tokenizer
        )
        assert chat_template_str == get_chat_template("chatml")

    def test_tokenizer_default_fallback_with_chat_template_on_tokenizer(
        self, llama3_tokenizer
    ):
        llama3_tokenizer.chat_template = "test_template"
        chat_template_str = get_chat_template(
            "tokenizer_default_fallback_chatml", tokenizer=llama3_tokenizer
        )
        assert chat_template_str == "test_template"

    def test_jinja_template_mode(self):
        jinja_template = "example_jinja_template"
        chat_template_str = get_chat_template("jinja", jinja_template=jinja_template)
        assert chat_template_str == jinja_template

    def test_jinja_template_mode_no_jinja_template(self):
        with pytest.raises(ValueError):
            get_chat_template("jinja", jinja_template=None)

    def test_extract_chat_template_args(self):
        # No ds_cfg
        chat_template_choice, chat_template_jinja = extract_chat_template_args(
            cfg={"chat_template": "chatml"},
        )
        assert chat_template_choice == "chatml"
        assert chat_template_jinja is None

        # ds_cfg provided
        chat_template_choice, chat_template_jinja = extract_chat_template_args(
            cfg={
                "chat_template": "jinja",
                "chat_template_jinja": "global_jinja_template",
            },
            ds_cfg={"chat_template": "llama3", "chat_template_jinja": None},
        )
        assert chat_template_choice == "llama3"
        assert chat_template_jinja is None

        # ds_cfg provided with jinja template
        chat_template_choice, chat_template_jinja = extract_chat_template_args(
            cfg={"chat_template": "chatml", "chat_template_jinja": None},
            ds_cfg={
                "chat_template": "jinja",
                "chat_template_jinja": "ds_jinja_template",
            },
        )
        assert chat_template_choice == "jinja"
        assert chat_template_jinja == "ds_jinja_template"

        # ds_cfg provided with no chat_template
        chat_template_choice, chat_template_jinja = extract_chat_template_args(
            cfg={
                "chat_template": "jinja",
                "chat_template_jinja": "global_jinja_template",
            },
            ds_cfg={"chat_template": None, "chat_template_jinja": "ds_jinja_template"},
        )
        assert chat_template_choice == "jinja"
        assert chat_template_jinja == "global_jinja_template"


if __name__ == "__main__":
    unittest.main()
