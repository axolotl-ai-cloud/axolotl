"""
Tests for user-defined DPO dataset transform strategies
"""

import pytest

from axolotl.prompt_strategies.dpo import load as load_dpo
from axolotl.prompt_strategies.dpo.user_defined import default
from axolotl.utils.dict import DictDefault


def make_cfg(type_cfg: dict) -> DictDefault:
    """Build a minimal config with a single user-defined DPO dataset."""
    return DictDefault({"datasets": [{"type": type_cfg}]})


class TestDPOUserDefined:
    """
    Test user_defined.default DPO transforms
    """

    def test_explicit_config(self):
        """Transform works with explicit prompt/chosen/rejected formats."""
        cfg = make_cfg(
            {
                "field_prompt": "prompt",
                "field_system": "system",
                "field_chosen": "chosen",
                "field_rejected": "rejected",
                "prompt_format": "{prompt}",
                "chosen_format": "{chosen}",
                "rejected_format": "{rejected}",
            }
        )
        transform_fn = default(cfg)
        sample = transform_fn({"prompt": "hello", "chosen": "good", "rejected": "bad"})
        assert sample["prompt"] == "hello"
        assert sample["chosen"] == "good"
        assert sample["rejected"] == "bad"

    def test_defaults_without_formats(self):
        """Transform falls back to canonical {prompt}/{chosen}/{rejected} formats
        when no explicit formats are configured."""
        cfg = make_cfg({})
        transform_fn = default(cfg)
        sample = transform_fn({"prompt": "hello", "chosen": "good", "rejected": "bad"})
        assert sample["prompt"] == "hello"
        assert sample["chosen"] == "good"
        assert sample["rejected"] == "bad"

    def test_custom_field_names(self):
        """Transform reads from custom field_prompt/field_chosen/field_rejected
        columns and writes the canonical prompt/chosen/rejected keys.

        Regression test for issue #2645: the default format strings used the
        custom field name as the placeholder (e.g. "{my_chosen}") while .format()
        was called with the canonical keyword (chosen=), raising KeyError for any
        non-default field name."""
        cfg = make_cfg(
            {
                "field_prompt": "question",
                "field_chosen": "my_chosen",
                "field_rejected": "my_rejected",
            }
        )
        transform_fn = default(cfg)
        sample = transform_fn(
            {"question": "hello", "my_chosen": "good", "my_rejected": "bad"}
        )
        assert sample["prompt"] == "hello"
        assert sample["chosen"] == "good"
        assert sample["rejected"] == "bad"

    def test_system_in_prompt_format(self):
        """A {system} placeholder in prompt_format is filled from the system
        field when present in the sample."""
        cfg = make_cfg({"prompt_format": "{system} {prompt}"})
        transform_fn = default(cfg)
        sample = transform_fn(
            {
                "system": "be helpful",
                "prompt": "hello",
                "chosen": "good",
                "rejected": "bad",
            }
        )
        assert sample["prompt"] == "be helpful hello"
        assert sample["chosen"] == "good"
        assert sample["rejected"] == "bad"

    def test_system_in_prompt_format_missing_system_field(self):
        """A {system} placeholder does not raise KeyError when the system field
        is absent from a sample (common in mixed datasets where only some rows
        carry a system prompt): the missing value renders as an empty string."""
        cfg = make_cfg({"prompt_format": "{system} {prompt}"})
        transform_fn = default(cfg)
        sample = transform_fn({"prompt": "hello", "chosen": "good", "rejected": "bad"})
        assert sample["prompt"] == " hello"
        assert sample["chosen"] == "good"
        assert sample["rejected"] == "bad"

    def test_system_in_prompt_format_none_system_field(self):
        """A None system value (how datasets represent a missing optional column)
        renders as an empty string rather than the literal text 'None'."""
        cfg = make_cfg({"prompt_format": "{system} {prompt}"})
        transform_fn = default(cfg)
        sample = transform_fn(
            {"system": None, "prompt": "hello", "chosen": "good", "rejected": "bad"}
        )
        assert sample["prompt"] == " hello"

    def test_custom_chosen_rejected_formats(self):
        """Explicit chosen_format/rejected_format are honored alongside custom
        field names."""
        cfg = make_cfg(
            {
                "field_chosen": "good",
                "field_rejected": "bad",
                "chosen_format": "Answer: {chosen}",
                "rejected_format": "Answer: {rejected}",
            }
        )
        transform_fn = default(cfg)
        sample = transform_fn({"prompt": "q", "good": "yes", "bad": "no"})
        assert sample["prompt"] == "q"
        assert sample["chosen"] == "Answer: yes"
        assert sample["rejected"] == "Answer: no"

    def test_non_dict_type_raises(self):
        """A non-dict dataset type raises a clear ValueError."""
        cfg = make_cfg({})
        cfg["datasets"][0]["type"] = "user_defined.default"
        with pytest.raises(ValueError, match="must be a dictionary"):
            default(cfg)

    def test_load_dpo_user_defined_returns_callable(self):
        """The loader path resolves user_defined.default to a callable transform
        for a config with custom field names."""
        cfg = make_cfg(
            {
                "field_prompt": "question",
                "field_chosen": "my_chosen",
                "field_rejected": "my_rejected",
            }
        )
        transform_fn = load_dpo("user_defined.default", cfg, dataset_idx=0)
        assert callable(transform_fn)
