"""
test cases for axolotl.utils.import_helper
"""

import pytest

from axolotl.utils.import_helper import get_cls_from_module_str


def test_get_cls_from_module_str():
    cls = get_cls_from_module_str("axolotl.core.trainers.base.AxolotlTrainer")
    assert cls.__name__ == "AxolotlTrainer"


def test_get_cls_from_module_str_empty_string():
    with pytest.raises(ValueError, match="module_str must be a non-empty string"):
        get_cls_from_module_str("")


def test_get_cls_from_module_str_whitespace_only():
    with pytest.raises(ValueError, match="module_str must be a non-empty string"):
        get_cls_from_module_str("   ")


def test_get_cls_from_module_str_invalid_format():
    with pytest.raises(ValueError, match="Invalid module string format"):
        get_cls_from_module_str("single_part")


def test_get_cls_from_module_str_nonexistent_module():
    with pytest.raises(ImportError, match="Failed to import module"):
        get_cls_from_module_str("nonexistent.module.Class")


def test_get_cls_from_module_str_nonexistent_class():
    with pytest.raises(AttributeError, match="Class 'NonExistentClass' not found"):
        get_cls_from_module_str("axolotl.core.trainers.base.NonExistentClass")
