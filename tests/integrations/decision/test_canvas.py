"""Decision canvas visibility and atomic budget contracts."""

import pytest

from axolotl.integrations.decision.template import resolve_template


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


def test_template_indexed_and_overflow():
    questions = [{"id": str(index), "labels": ["a", "b"]} for index in range(64)]
    _, slots = resolve_template(
        CharacterTokenizer(), questions, fmt="indexed", width=256
    )
    assert len(slots) == 64
    assert all(len(slot["label_ids"]) == 2 for slot in slots)
    with pytest.raises(ValueError):
        resolve_template(CharacterTokenizer(), questions, fmt="indexed", width=8)
