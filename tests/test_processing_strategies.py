"""
Unit tests for ``ProcessingStrategy`` (multimodal collator's processing entry).

These cover the configurable ``field_messages`` plumbing introduced for the
multimodal path: the helpers (``_normalize_field_messages``,
``_is_legacy_schema``, ``_get_messages_field``) and the end-to-end ``__call__``
behavior across the OpenAI / ShareGPT / custom-field combinations.

Tests use a ``MagicMock`` processor with no ``image_token`` attribute so the
``__init__`` does not require a real tokenizer; this keeps the suite fast and
cpu-only.
"""

import unittest
from unittest.mock import MagicMock

from axolotl.processing_strategies import ProcessingStrategy


def _make_processor() -> MagicMock:
    """Minimal processor mock that skips the ``image_token`` branch in __init__."""
    processor = MagicMock()
    del processor.image_token
    return processor


def _openai_messages() -> list[dict]:
    return [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def _sharegpt_messages() -> list[dict]:
    return [
        {"from": "human", "value": "hello"},
        {"from": "gpt", "value": "hi"},
    ]


class TestNormalizeFieldMessages(unittest.TestCase):
    """``_normalize_field_messages`` should turn the raw config into a tuple."""

    def test_none_defaults_to_messages(self):
        self.assertEqual(
            ProcessingStrategy._normalize_field_messages(None), ("messages",)
        )

    def test_string_wrapped_in_tuple(self):
        self.assertEqual(
            ProcessingStrategy._normalize_field_messages("dialogue"),
            ("dialogue",),
        )

    def test_list_preserved(self):
        self.assertEqual(
            ProcessingStrategy._normalize_field_messages(["foo", "bar"]),
            ("foo", "bar"),
        )

    def test_falsy_entries_dropped(self):
        self.assertEqual(
            ProcessingStrategy._normalize_field_messages(["foo", "", None, "bar"]),
            ("foo", "bar"),
        )


class TestIsLegacySchema(unittest.TestCase):
    """``_is_legacy_schema`` requires both ``from`` and ``value`` to flag legacy."""

    def test_sharegpt_pair_detected(self):
        self.assertTrue(
            ProcessingStrategy._is_legacy_schema([{"from": "human", "value": "hi"}])
        )

    def test_only_from_is_not_enough(self):
        # A row whose first message merely carries a `from` key (e.g., custom
        # metadata) must not be treated as ShareGPT.
        self.assertFalse(
            ProcessingStrategy._is_legacy_schema([{"from": "human", "extra": "x"}])
        )

    def test_openai_schema_returns_false(self):
        self.assertFalse(
            ProcessingStrategy._is_legacy_schema([{"role": "user", "content": "hi"}])
        )

    def test_empty_list_returns_false(self):
        self.assertFalse(ProcessingStrategy._is_legacy_schema([]))

    def test_non_list_returns_false(self):
        self.assertFalse(ProcessingStrategy._is_legacy_schema(None))
        self.assertFalse(ProcessingStrategy._is_legacy_schema("not a list"))


class TestGetMessagesField(unittest.TestCase):
    """``_get_messages_field`` resolves which key in the example to read from."""

    def test_default_returns_messages(self):
        strategy = ProcessingStrategy(processor=_make_processor())
        self.assertEqual(
            strategy._get_messages_field({"messages": _openai_messages()}),
            "messages",
        )

    def test_custom_field_returns_custom(self):
        strategy = ProcessingStrategy(
            processor=_make_processor(), field_messages="my_col"
        )
        self.assertEqual(
            strategy._get_messages_field({"my_col": _openai_messages()}),
            "my_col",
        )

    def test_custom_field_takes_priority_over_stale_messages(self):
        # If the user explicitly configured a custom field, that intent must
        # not be silently overridden by a leftover `messages` column.
        strategy = ProcessingStrategy(
            processor=_make_processor(), field_messages="my_col"
        )
        example = {
            "messages": [{"role": "user", "content": "stale"}],
            "my_col": _openai_messages(),
        }
        self.assertEqual(strategy._get_messages_field(example), "my_col")

    def test_falls_back_to_messages_when_custom_absent(self):
        strategy = ProcessingStrategy(
            processor=_make_processor(), field_messages="my_col"
        )
        self.assertEqual(
            strategy._get_messages_field({"messages": _openai_messages()}),
            "messages",
        )

    def test_returns_none_when_no_match(self):
        strategy = ProcessingStrategy(
            processor=_make_processor(), field_messages="my_col"
        )
        self.assertIsNone(strategy._get_messages_field({"random_col": "x"}))


class TestProcessingStrategyCall(unittest.TestCase):
    """End-to-end ``__call__`` covering canonical keys and custom-field reroute."""

    def test_canonical_messages_openai(self):
        strategy = ProcessingStrategy(processor=_make_processor())
        out = strategy([{"messages": _openai_messages()}])

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["messages"][0]["role"], "user")
        # String content gets wrapped to multimedia content list
        self.assertEqual(
            out[0]["messages"][0]["content"], [{"type": "text", "text": "hello"}]
        )

    def test_canonical_conversations_sharegpt(self):
        strategy = ProcessingStrategy(processor=_make_processor())
        out = strategy([{"conversations": _sharegpt_messages()}])

        # Roles get normalized: human -> user, gpt -> assistant
        self.assertEqual(out[0]["messages"][0]["role"], "user")
        self.assertEqual(out[0]["messages"][1]["role"], "assistant")
        # Original `conversations` key is dropped
        self.assertNotIn("conversations", out[0])

    def test_custom_field_with_openai_schema(self):
        strategy = ProcessingStrategy(
            processor=_make_processor(), field_messages="my_col"
        )
        out = strategy([{"my_col": _openai_messages()}])

        self.assertEqual(out[0]["messages"][0]["role"], "user")
        # The custom column is normalized away
        self.assertNotIn("my_col", out[0])

    def test_custom_field_with_sharegpt_schema(self):
        # Schema auto-detection: a custom column whose first message looks like
        # ShareGPT must be routed through the legacy conversion branch.
        strategy = ProcessingStrategy(
            processor=_make_processor(), field_messages="my_col"
        )
        out = strategy([{"my_col": _sharegpt_messages()}])

        # Roles got normalized via the legacy path (human -> user)
        self.assertEqual(out[0]["messages"][0]["role"], "user")
        self.assertEqual(out[0]["messages"][1]["role"], "assistant")
        self.assertNotIn("my_col", out[0])
        self.assertNotIn("conversations", out[0])

    def test_custom_field_overrides_stale_messages_column(self):
        strategy = ProcessingStrategy(
            processor=_make_processor(), field_messages="my_col"
        )
        out = strategy(
            [
                {
                    "messages": [{"role": "user", "content": "stale"}],
                    "my_col": [{"role": "user", "content": "fresh"}],
                }
            ]
        )

        # Output should reflect `my_col`, not the stale `messages`
        self.assertEqual(
            out[0]["messages"][0]["content"], [{"type": "text", "text": "fresh"}]
        )

    def test_unknown_key_raises_value_error(self):
        strategy = ProcessingStrategy(
            processor=_make_processor(), field_messages="my_col"
        )

        with self.assertRaises(ValueError):
            strategy([{"random_col": _openai_messages()}])


if __name__ == "__main__":
    unittest.main()
