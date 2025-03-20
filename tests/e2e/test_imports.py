"""
test module to import various submodules that have historically broken due to dependency issues
"""

import unittest


class TestImports(unittest.TestCase):
    """
    Test class to import various submodules that have historically broken due to dependency issues
    """

    def test_import_causal_trainer(self):
        from axolotl.core.trainer_builder import (  # pylint: disable=unused-import  # noqa: F401
            HFCausalTrainerBuilder,
        )

    def test_import_rl_trainer(self):
        from axolotl.core.trainer_builder import (  # pylint: disable=unused-import  # noqa: F401
            HFRLTrainerBuilder,
        )
