import unittest

import pytest

from axolotl.utils.dict import DictDefault


class DictDefaultTest(unittest.TestCase):
    def test_dict_default(self):
        cfg = DictDefault(
            {
                "key_a": {"key_b": "value_a"},
                "key_c": "value_c",
                "key_d": ["value_d", "value_e"],
            }
        )

        assert (
            cfg.key_a.key_b == "value_a"
        ), "DictDefault should return value for existing nested keys"

        assert (
            cfg.key_c == "value_c"
        ), "DictDefault should return value for existing keys"

        assert (
            cfg.key_d[0] == "value_d"
        ), "DictDefault should return value for existing keys in list"

        assert (
            "value_e" in cfg.key_d
        ), "DictDefault should support in operator for existing keys in list"

    def test_dict_or_operator(self):
        cfg = DictDefault(
            {
                "key_a": {"key_b": "value_a"},
                "key_c": "value_c",
                "key_d": ["value_d", "value_e"],
                "key_f": "value_f",
            }
        )

        cfg = cfg | DictDefault({"key_a": {"key_b": "value_b"}, "key_f": "value_g"})

        assert (
            cfg.key_a.key_b == "value_b"
        ), "DictDefault should support OR operator for existing nested keys"

        assert cfg.key_c == "value_c", "DictDefault should not delete existing key"

        assert cfg.key_d == [
            "value_d",
            "value_e",
        ], "DictDefault should not overwrite existing keys in list"

        assert (
            cfg.key_f == "value_g"
        ), "DictDefault should support OR operator for existing key"

    def test_dict_missingkey(self):
        cfg = DictDefault({})

        assert cfg.random_key is None, "DictDefault should return None for missing keys"

    def test_dict_nested_missingparentkey(self):
        """
        Due to subclassing Dict, DictDefault will error if we try to access a nested key whose parent key does not exist.
        """
        cfg = DictDefault({})

        with pytest.raises(
            AttributeError,
            match=r"'NoneType' object has no attribute 'another_random_key'",
        ):
            cfg.random_key.another_random_key

    def test_dict_shorthand_assignment(self):
        """
        Shorthand assignment is said to not be supported if subclassed. However, their example raises error instead of None.
        This test ensures that it is supported for current implementation.

        Ref: https://github.com/mewwts/addict#default-values
        """

        cfg = DictDefault({"key_a": {"key_b": "value_a"}})

        cfg.key_a.key_b = "value_b"

        assert cfg.key_a.key_b == "value_b", "Shorthand assignment should be supported"
