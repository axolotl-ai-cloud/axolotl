"""
test cases for axolotl.utils.import_helper
"""

from axolotl.utils.import_helper import get_cls_from_module_str


def test_get_cls_from_module_str():
    cls = get_cls_from_module_str("axolotl.core.trainers.base.AxolotlTrainer")
    assert cls.__name__ == "AxolotlTrainer"
