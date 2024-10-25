"""Test module for checking whether the integration of Unsloth with Hugging Face Transformers is working as expected."""
import unittest

from axolotl.monkeypatch.unsloth_ import check_self_attn_is_patchable


class TestUnslothIntegration(unittest.TestCase):
    """Unsloth monkeypatch integration tests."""

    def test_is_self_attn_patchable(self):
        # ensures the current version of transformers has loss code that matches our patching code
        self.assertTrue(
            check_self_attn_is_patchable(),
            "HF transformers self attention code has changed and isn't patchable",
        )
