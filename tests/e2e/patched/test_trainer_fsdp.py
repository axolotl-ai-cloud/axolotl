"""Test module for checking whether the integration of Unsloth with Hugging Face Transformers is working as expected."""
import unittest

from axolotl.monkeypatch.trainer_fsdp_grad_accum import check_training_loop_is_patchable


class TestTrainerFSDPIntegration(unittest.TestCase):
    """Unsloth monkeypatch integration tests."""

    def test_train_loop_patchable(self):
        # ensures the current version of transformers has loss code that matches our patching code
        self.assertTrue(
            check_training_loop_is_patchable(),
            "HF transformers _inner_training_loop has changed and isn't patchable",
        )
