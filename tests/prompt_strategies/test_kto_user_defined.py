"""
Tests for user-defined KTO dataset transform strategies
"""

import pytest

from axolotl.prompt_strategies.kto import load as load_kto
from axolotl.prompt_strategies.kto.user_defined import default
from axolotl.utils.dict import DictDefault


def make_cfg(type_cfg: dict) -> DictDefault:
    return DictDefault({"datasets": [{"type": type_cfg}]})


class TestKTOUserDefined:
    """
    Test user_defined.default KTO transforms
    """

    def test_documented_config(self):
        # config from docs/rlhf.qmd, the same shape reported in issue #2757
        cfg = make_cfg(
            {
                "field_prompt": "prompt",
                "field_system": "system",
                "field_completion": "completion",
                "field_label": "label",
                "prompt_format": "{prompt}",
                "completion_format": "{completion}",
            }
        )
        transform_fn = default(cfg)
        sample = transform_fn({"prompt": "hello", "completion": "world", "label": True})
        assert sample["prompt"] == "hello"
        assert sample["completion"] == "world"
        assert sample["label"] is True

    def test_defaults_without_formats(self):
        cfg = make_cfg({})
        transform_fn = default(cfg)
        sample = transform_fn(
            {"prompt": "hello", "completion": "world", "label": False}
        )
        assert sample["prompt"] == "hello"
        assert sample["completion"] == "world"
        assert sample["label"] is False

    def test_custom_field_names(self):
        cfg = make_cfg(
            {
                "field_prompt": "question",
                "field_completion": "answer",
                "field_label": "is_good",
            }
        )
        transform_fn = default(cfg)
        sample = transform_fn({"question": "hello", "answer": "world", "is_good": True})
        assert sample["prompt"] == "hello"
        assert sample["completion"] == "world"
        assert sample["label"] is True

    def test_system_in_prompt_format(self):
        cfg = make_cfg(
            {
                "prompt_format": "{system} {prompt}",
            }
        )
        transform_fn = default(cfg)
        sample = transform_fn(
            {
                "system": "be helpful",
                "prompt": "hello",
                "completion": "world",
                "label": True,
            }
        )
        assert sample["prompt"] == "be helpful hello"
        assert sample["completion"] == "world"

    def test_non_dict_type_raises(self):
        cfg = make_cfg({})
        cfg["datasets"][0]["type"] = "user_defined.default"
        with pytest.raises(ValueError, match="must be a dictionary"):
            default(cfg)

    def test_load_kto_user_defined_returns_callable(self):
        # regression: load_kto previously swallowed errors and returned None,
        # which crashed later with "TypeError: None is not a callable object"
        cfg = make_cfg(
            {
                "field_prompt": "prompt",
                "field_completion": "completion",
                "field_label": "label",
                "prompt_format": "{prompt}",
                "completion_format": "{completion}",
            }
        )
        transform_fn = load_kto("user_defined.default", cfg, dataset_idx=0)
        assert callable(transform_fn)
