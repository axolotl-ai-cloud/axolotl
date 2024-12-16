""""Test module for checking whether the Hugging Face Transformers is working as expected."""
import unittest

from axolotl.monkeypatch.trainer_grad_accum import (
    check_forward_is_patchable,
    check_training_step_is_patchable,
)


class TestTrainerGAIntegration(unittest.TestCase):
    """llama monkeypatch integration tests."""

    def test_train_step_patchable(self):
        # ensures the current version of transformers has loss code that matches our patching code
        self.assertTrue(
            check_training_step_is_patchable(),
            "HF transformers Trainer.training_step has changed and isn't patchable",
        )

    def test_model_forward_patchable(self):
        # ensures the current version of transformers has loss code that matches our patching code
        self.assertTrue(
            check_forward_is_patchable(),
            "HF transformers LlamaForCausalLM.forward has changed and isn't patchable",
        )
