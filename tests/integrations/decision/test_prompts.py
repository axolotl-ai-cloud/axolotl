import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from axolotl.integrations.decision import prompts, template


class RecordingTokenizer:
    def __init__(self, output: Any):
        self.output = output
        self.messages: Any = None
        self.kwargs: dict[str, Any] | None = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return self.output


def test_serializes_state_like_pinned_djev():
    assert prompts.serialize_state("raw °") == "raw °"
    assert prompts.serialize_state({"temperature": "20°"}) == '{"temperature": "20°"}'
    with pytest.raises(ValueError, match="state is required"):
        prompts.serialize_state(None)


def test_structured_content_and_dict_tokenizer_output():
    tokenizer = RecordingTokenizer({"input_ids": [[4, 5, 6]]})

    ids = prompts.decision_prompt_ids(
        tokenizer,
        "system",
        {"value": "20°"},
        thinking=True,
        structured_content=True,
    )

    assert ids == (4, 5, 6)
    assert tokenizer.messages == [
        {"role": "system", "content": [{"type": "text", "text": "system"}]},
        {
            "role": "user",
            "content": [{"type": "text", "text": '{"value": "20°"}'}],
        },
    ]
    assert tokenizer.kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": True,
    }


def test_defaults_to_source_faithful_string_content():
    tokenizer = RecordingTokenizer([7, 8])

    assert prompts.decision_prompt_ids(tokenizer, "system", "raw state") == (7, 8)
    assert tokenizer.messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "raw state"},
    ]


def test_rejects_multiple_prompts_or_non_token_output():
    with pytest.raises(ValueError, match="more than one prompt"):
        prompts.decision_prompt_ids(
            RecordingTokenizer([[1], [2]]), "system", "state", structured_content=False
        )
    with pytest.raises(TypeError, match="token ids"):
        prompts.decision_prompt_ids(
            RecordingTokenizer(object()), "system", "state", structured_content=False
        )
    with pytest.raises(TypeError, match="integer token ids"):
        prompts.decision_prompt_ids(
            RecordingTokenizer([1.5]), "system", "state", structured_content=False
        )


def _djev_questions(questions):
    result = []
    for question_id, question in questions.items():
        item = {
            "id": question_id,
            "type": question["kind"] if "kind" in question else question["type"],
            "instructions": question.get("instructions", ""),
        }
        if item["type"] == "choice":
            item["options"] = (
                [
                    {"name": option, "description": description}
                    if isinstance(option, str)
                    else option
                    for option, description in question.get("options", {}).items()
                ]
                if isinstance(question.get("options"), dict)
                else list(question["options"])
            )
        elif item["type"] == "score":
            item["levels"] = question["levels"]
        elif item["type"] == "noul":
            item["criteria"] = question.get("criteria", {})
        else:
            raise AssertionError(item["type"])
        result.append(item)
    return result


@pytest.mark.skipif(
    not Path("/tmp/axolotl-diffusion-results/m2-djev-paired-canvases-50.jsonl").exists()
    or not Path("/tmp/diffusion-m2/open-jev-train-first50-normalized.jsonl").exists()
    or not Path(
        "/mnt/data/hf_cache/hub/models--google--diffusiongemma-26B-A4B-it/"
        "snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b"
    ).exists(),
    reason="local Gemma parity fixture and tokenizer snapshot are unavailable",
)
def test_local_gemma_prompt_ids_match_all_serialized_first50_fixtures():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/data/hf_cache/hub/models--google--diffusiongemma-26B-A4B-it/"
        "snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b",
        local_files_only=True,
    )
    fixture_path = Path(
        "/tmp/axolotl-diffusion-results/m2-djev-paired-canvases-50.jsonl"
    )
    records_path = Path("/tmp/diffusion-m2/open-jev-train-first50-normalized.jsonl")
    fixtures = [json.loads(line) for line in fixture_path.read_text().splitlines()]
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert len(fixtures) == len(records) == 50

    for fixture, record in zip(fixtures, records, strict=True):
        assert fixture["id"] == record["id"]
        schema = template.parse_schema(
            {"questions": _djev_questions(record["questions"]), "samples": 1}
        )
        ids = prompts.decision_prompt_ids(
            tokenizer,
            template.system_text(schema),
            record["state"],
        )
        assert ids == tuple(fixture["prompt_ids"])


@pytest.mark.parametrize(
    ("name", "snapshot"),
    [
        (
            "gemma",
            "/mnt/data/hf_cache/hub/models--google--diffusiongemma-26B-A4B-it/"
            "snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b",
        ),
        (
            "dream",
            "/mnt/data/hf_cache/hub/models--Dream-org--Dream-v0-Instruct-7B/"
            "snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae",
        ),
    ],
)
def test_local_tokenizer_matches_pinned_djev_chat_prompt_ids(name, snapshot):
    if (
        not Path(snapshot).exists()
        or not Path("/tmp/diffusion-m0/src/djev-structured_server.py").exists()
    ):
        pytest.skip("local pinned source or tokenizer snapshot is unavailable")

    from transformers import AutoTokenizer

    try:
        source_module = importlib.import_module("axolotl.integrations.decision.scripts.baselines")
    except ModuleNotFoundError as error:
        if error.name == "axolotl.integrations.decision.scripts.baselines":
            pytest.skip("local pinned baseline reference is unavailable")
        raise
    pinned_djev = source_module.pinned_djev

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=True
    )
    source = pinned_djev(tokenizer)
    system_text = "System °"
    state = {"value": "20°"}
    expected = source.chat_prompt_ids(system_text, prompts.serialize_state(state))

    assert prompts.decision_prompt_ids(tokenizer, system_text, state) == tuple(expected)
