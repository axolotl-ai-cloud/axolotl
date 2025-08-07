"""Unit tests for trainer loss calc monkeypatch."""

import unittest

from transformers import Trainer

from axolotl.monkeypatch.transformers.trainer_loss_calc import (
    patch_evaluation_loop,
    patch_maybe_log_save_evaluate,
)


class TestTrainerLossCalc(unittest.TestCase):
    """
    Unit test class for trainer loss calc monkeypatch
    """

    def test_patch_evaluation_loop_applies(self):
        """
        Test that patch_evaluation_loop applies successfully
        """
        # Ensure we start with a clean state
        if hasattr(Trainer, "_original_evaluation_loop"):
            delattr(Trainer, "_original_evaluation_loop")

        patch_evaluation_loop()
        self.assertTrue(hasattr(Trainer, "_original_evaluation_loop"))

    def test_patch_maybe_log_save_evaluate_applies(self):
        """
        Test that patch_maybe_log_save_evaluate applies successfully
        """
        # Ensure we start with a clean state
        if hasattr(Trainer, "_original_maybe_log_save_evaluate"):
            delattr(Trainer, "_original_maybe_log_save_evaluate")

        patch_maybe_log_save_evaluate()
        self.assertTrue(hasattr(Trainer, "_original_maybe_log_save_evaluate"))
