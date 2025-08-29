"""
Unit tests for trainer accelerator args monkeypatch
"""

import unittest

from axolotl.monkeypatch.trainer_accelerator_args import (
    check_create_accelerate_code_is_patchable,
)


class TestTrainerAcceleratorArgs(unittest.TestCase):
    """
    Unit test class for trainer accelerator args monkeypatch
    """

    def test_check_create_accelerate_code_is_patchable(self):
        """
        Test that the upstream transformers code is still patchable.
        This will fail if the patched code changes upstream.
        """
        assert check_create_accelerate_code_is_patchable()


if __name__ == "__main__":
    unittest.main()
