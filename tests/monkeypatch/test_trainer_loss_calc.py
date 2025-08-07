"""Unit tests for trainer loss calc monkeypatch."""

import unittest

from axolotl.monkeypatch.transformers.trainer_loss_calc import (
    check_evaluation_loop_is_patchable,
    check_maybe_log_save_evaluate_is_patchable,
)


class TestTrainerLossCalc(unittest.TestCase):
    """
    Unit test class for trainer loss calc monkeypatch
    """

    def test_trainer_loss_calc_is_patchable(self):
        """
        Test that the upstream transformers code is still patchable. This will fail if
        the patched code changes upstream.
        """
        assert check_evaluation_loop_is_patchable()
        assert check_maybe_log_save_evaluate_is_patchable()


if __name__ == "__main__":
    unittest.main()
