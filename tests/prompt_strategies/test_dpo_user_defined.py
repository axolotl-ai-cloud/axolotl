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
        """Falls back to canonical {prompt}/{chosen}/{rejected} when no formats given."""
        cfg = make_cfg({})
        transform_fn = default(cfg)
        sample = transform_fn({"prompt": "hello", "chosen": "good", "rejected": "bad"})
        assert sample["prompt"] == "hello"
        assert sample["chosen"] == "good"
        assert sample["rejected"] == "bad"

    def test_custom_field_names(self):
        """Custom field_* columns are read and written to the canonical keys."""
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
        """{system} is filled from the system field when present."""
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

    def test_system_in_prompt_format_custom_field(self):
        """{system} reads from a custom field_system column."""
        cfg = make_cfg({"field_system": "sys", "prompt_format": "{system} {prompt}"})
        transform_fn = default(cfg)
        sample = transform_fn(
            {
                "sys": "be helpful",
                "prompt": "hello",
                "chosen": "good",
                "rejected": "bad",
            }
        )
        assert sample["prompt"] == "be helpful hello"

    @pytest.mark.parametrize(
        "sample",
        [
            {"prompt": "hello", "chosen": "good", "rejected": "bad"},
            {"system": None, "prompt": "hello", "chosen": "good", "rejected": "bad"},
        ],
        ids=["missing", "none"],
    )
    def test_system_in_prompt_format_empty_system(self, sample):
        """A missing or None system field renders as "" instead of raising or 'None'."""
        cfg = make_cfg({"prompt_format": "{system} {prompt}"})
        transform_fn = default(cfg)
        assert transform_fn(sample)["prompt"] == " hello"

    def test_custom_chosen_rejected_formats(self):
        """Explicit chosen_format/rejected_format are honored with custom field names."""
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
        """The loader resolves user_defined.default to a callable transform."""
        cfg = make_cfg(
            {
                "field_prompt": "question",
                "field_chosen": "my_chosen",
                "field_rejected": "my_rejected",
            }
        )
        transform_fn = load_dpo("user_defined.default", cfg, dataset_idx=0)
        assert callable(transform_fn)
