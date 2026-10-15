"""Preference strategies keep an assistant turn's reasoning_content for templates with a thinking channel."""

from axolotl.prompt_strategies.dpo.chat_template import default
from axolotl.prompt_strategies.orpo.chat_template import (
    Message,
    ORPODatasetParsingStrategy,
)
from axolotl.utils.dict import DictDefault


class _Tokenizer:
    """Renders role, content and any reasoning_content, like a Harmony-style template."""

    chat_template = "stub"

    def apply_chat_template(self, messages, add_generation_prompt=False, **_):
        parts = []
        for m in messages:
            reasoning = m.get("reasoning_content")
            parts.append(
                f"<{m['role']}>"
                + (f"[think:{reasoning}]" if reasoning else "")
                + m["content"]
            )
        return "".join(parts) + ("<assistant>" if add_generation_prompt else "")


def test_dpo_default_carries_mapped_reasoning_content():
    cfg = DictDefault(
        {
            "datasets": [
                {
                    "path": "x",
                    "type": "chat_template.default",
                    "field_messages": "prompt",
                    "message_property_mappings": {
                        "role": "role",
                        "content": "content",
                        "reasoning_content": "reasoning_content",
                    },
                }
            ]
        }
    )
    transform = default(cfg)
    if isinstance(transform, tuple):
        transform = transform[0]
    sample = {
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [
            {"role": "assistant", "content": "good", "reasoning_content": "why"}
        ],
        "rejected": [{"role": "assistant", "content": "bad"}],
    }
    out = transform(sample, tokenizer=_Tokenizer())
    assert "[think:why]good" in out["chosen"]
    assert "[think:" not in out["rejected"] and "bad" in out["rejected"]


def test_orpo_message_keeps_reasoning_and_drops_label():
    parser = ORPODatasetParsingStrategy()
    sample = {
        "system": "s",
        "chosen": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "good", "reasoning_content": "why"},
        ],
        "rejected": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "bad"},
        ],
    }
    chosen = parser.get_chosen(sample).messages[-1]
    assert chosen.reasoning_content == "why"
    assert chosen.to_chat() == {
        "role": "assistant",
        "content": "good",
        "reasoning_content": "why",
    }
    rejected = parser.get_rejected(sample).messages[-1]
    assert rejected.to_chat() == {"role": "assistant", "content": "bad"}
    assert Message(role="user", content="q").to_chat() == {
        "role": "user",
        "content": "q",
    }


def test_orpo_argilla_rebuilds_the_rendered_columns():
    from axolotl.prompt_strategies.orpo.chat_template import argilla
    from axolotl.utils.dict import DictDefault

    cfg = DictDefault(
        {"chat_template": "tokenizer_default", "datasets": [{"path": "x"}]}
    )
    transform, map_kwargs = argilla(cfg)
    assert map_kwargs == {"remove_columns": ["chosen", "rejected"]}
    sample = {
        "system": "s",
        "chosen": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "good", "reasoning_content": "why"},
        ],
        "rejected": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "bad"},
        ],
    }
    out = transform(sample, tokenizer=_Tokenizer())
    assert out["prompt"].endswith("<assistant>")
    assert out["chosen"].startswith("[think:why]good") and out["rejected"].startswith(
        "bad"
    )
