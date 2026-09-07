"""
Unit tests for trainer accelerator args monkeypatch
"""

import unittest

from axolotl.monkeypatch.trainer_accelerator_args import (
    PATCHED_TRAINER_CODE,
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

    def test_patched_code_forwards_fp8_recipe(self):
        patched_code = PATCHED_TRAINER_CODE.format(
            fp8_recipe="rowwise", enable_fsdp_float8_all_gather=False
        )

        assert "fp8_recipe='rowwise'" in patched_code
        assert "enable_fsdp_float8_all_gather=False" in patched_code
        assert "self.additional_accelerator_args" in patched_code


if __name__ == "__main__":
    unittest.main()
