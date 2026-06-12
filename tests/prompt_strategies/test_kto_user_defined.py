"""
Tests for user-defined KTO dataset transform strategies
"""

import pytest

from axolotl.prompt_strategies.kto import load as load_kto
from axolotl.prompt_strategies.kto.user_defined import default
from axolotl.utils.dict import DictDefault


def make_cfg(type_cfg: dict) -> DictDefault:
    """Build a minimal config with a single user-defined KTO dataset."""
    return DictDefault({"datasets": [{"type": type_cfg}]})


class TestKTOUserDefined:
    """
    Test user_defined.default KTO transforms
    """

    def test_documented_config(self):
        """Transform works with the exact config documented in docs/rlhf.qmd
        (the same shape reported in issue #2757)."""
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
        """Transform falls back to canonical {prompt}/{completion} formats when
        no explicit formats are configured."""
        cfg = make_cfg({})
        transform_fn = default(cfg)
        sample = transform_fn(
            {"prompt": "hello", "completion": "world", "label": False}
        )
        assert sample["prompt"] == "hello"
        assert sample["completion"] == "world"
        assert sample["label"] is False

    def test_custom_field_names(self):
        """Transform reads from custom field_prompt/field_completion/field_label
        columns and writes the canonical prompt/completion/label keys."""
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
        """A {system} placeholder in prompt_format is filled from the system
        field when present in the sample."""
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
        """A non-dict dataset type raises a clear ValueError."""
        cfg = make_cfg({})
        cfg["datasets"][0]["type"] = "user_defined.default"
        with pytest.raises(ValueError, match="must be a dictionary"):
            default(cfg)

    def test_load_kto_user_defined_returns_callable(self):
        """Regression: load_kto previously swallowed errors and returned None,
        which crashed later with "TypeError: None is not a callable object"."""
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
