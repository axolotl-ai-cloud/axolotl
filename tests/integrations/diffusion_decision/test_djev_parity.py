"""Exact upstream fixtures generated with pinned Gemma and Dream tokenizers."""

import json
import re
from pathlib import Path

import pytest

from axolotl.integrations.diffusion_decision import template

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "djev_template_e5841cf.json").read_text()
)


class RecordedTokenizer:
    def __init__(self, encodings):
        self.encodings = encodings

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return self.encodings[text]


def json_value(value):
    return json.loads(json.dumps(value))


@pytest.mark.parametrize("family", ["gemma", "dream"])
@pytest.mark.parametrize("index", range(22))
def test_schema_system_answer_and_resolver_match_pinned_djev(family, index):
    reference = _FIXTURE["tokenizers"][family]
    case = reference["cases"][index]
    schema = template.parse_schema(case["schema"])
    assert json_value(schema) == case["parsed"]
    assert template.system_text(schema, chunked=case["chunked"]) == case["system_text"]
    assert (
        template.answer_text(
            schema["questions"], case["answer_indices"], schema["format"]
        )
        == case["answer_text"]
    )
    args = {
        "tokenizer": RecordedTokenizer(reference["encodings"]),
        "qs": schema["questions"],
        "head": case["head"],
        "lead": case["lead"],
        "fmt": schema["format"],
        "width": case["width"],
    }
    if "template_error" in case:
        with pytest.raises(
            template.SchemaError, match="^" + re.escape(case["template_error"]) + "$"
        ):
            template.resolve_template(**args)
    else:
        assert json_value(template.resolve_template(**args)) == case["template"]


@pytest.mark.parametrize("case", _FIXTURE["invalid_schemas"])
def test_schema_rejection_matches_pinned_djev(case):
    with pytest.raises(
        template.SchemaError, match="^" + re.escape(case["error"]) + "$"
    ):
        template.parse_schema(case["schema"])


def test_fixture_includes_successful_native_tokenization_and_boundary_cases():
    assert _FIXTURE["source_revision"] == "e5841cf41e9211608e698492658685c36e24e77a"
    for reference in _FIXTURE["tokenizers"].values():
        assert sum("template" in case for case in reference["cases"]) >= 20
        assert {case["parsed"]["format"] for case in reference["cases"]} == {
            "lines",
            "indexed",
        }
        assert any(
            len(case["parsed"]["questions"]) == 64 for case in reference["cases"]
        )
        assert reference["tokenizer_files_sha256"]
