"""Exact upstream fixtures generated with pinned Gemma and Dream tokenizers."""

import json
import re

import pytest

from axolotl.integrations.diffusion_decision import template

from tests.integrations.diffusion_decision.helpers import (
    DJEV_FIXTURE,
    RecordedTokenizer,
)


def json_value(value):
    return json.loads(json.dumps(value))


@pytest.mark.parametrize("family", ["gemma", "dream"])
@pytest.mark.parametrize("index", range(22))
def test_schema_system_answer_and_resolver_match_pinned_djev(family, index):
    reference = DJEV_FIXTURE["tokenizers"][family]
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


@pytest.mark.parametrize("case", DJEV_FIXTURE["invalid_schemas"])
def test_schema_rejection_matches_pinned_djev(case):
    with pytest.raises(
        template.SchemaError, match="^" + re.escape(case["error"]) + "$"
    ):
        template.parse_schema(case["schema"])


def test_fixture_includes_successful_native_tokenization_and_boundary_cases():
    assert DJEV_FIXTURE["source_revision"] == "e5841cf41e9211608e698492658685c36e24e77a"
    for reference in DJEV_FIXTURE["tokenizers"].values():
        assert sum("template" in case for case in reference["cases"]) >= 20
        assert {case["parsed"]["format"] for case in reference["cases"]} == {
            "lines",
            "indexed",
        }
        assert any(
            len(case["parsed"]["questions"]) == 64 for case in reference["cases"]
        )
        assert reference["tokenizer_files_sha256"]
