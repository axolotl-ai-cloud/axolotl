"""Tests for nested config option handling via CLI dot-notation."""

from dataclasses import dataclass, field
from typing import Optional

import click
from click.testing import CliRunner
from pydantic import BaseModel, Field

from axolotl.cli.utils.args import (
    add_options_from_config,
    add_options_from_dataclass,
    filter_none_kwargs,
)


class InnerConfig(BaseModel):
    """A nested config model for testing."""

    beta: float | None = Field(
        default=None,
        description="Beta parameter.",
    )
    host: str | None = Field(
        default=None,
        description="Server host.",
    )
    use_feature: bool = Field(
        default=False,
        description="Whether to use the feature.",
    )


class OuterConfig(BaseModel):
    """A top-level config model for testing."""

    learning_rate: float | None = Field(
        default=None,
        description="Learning rate.",
    )
    inner: InnerConfig | None = Field(
        default=None,
        description="Inner config.",
    )
    name: str | None = Field(
        default=None,
        description="Model name.",
    )


@dataclass
class SampleCliArgs:
    """A dataclass whose fields are documented the way the CLI args are."""

    verbose: bool = field(
        default=False,
        metadata={"help": "Log verbosely."},
    )
    tensor_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Number of workers."},
    )
    undocumented: Optional[str] = field(default=None)


class TestAddOptionsFromDataclass:
    """Test that add_options_from_dataclass surfaces field help text."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_field_help_metadata_is_shown_in_help(self):
        """Fields documented with `metadata={"help": ...}` should describe their option."""

        @click.command()
        @add_options_from_dataclass(SampleCliArgs)
        def cmd(**kwargs):
            pass

        result = self.runner.invoke(cmd, ["--help"])

        assert result.exit_code == 0, result.output

        # Click wraps long help text, so compare on normalized whitespace.
        output = " ".join(result.output.split())
        assert "--tensor-parallel-size INTEGER Number of workers." in output
        assert "Log verbosely." in output

    def test_undocumented_field_has_no_help(self):
        """Fields without help metadata should still register as bare options."""

        @click.command()
        @add_options_from_dataclass(SampleCliArgs)
        def cmd(**kwargs):
            pass

        result = self.runner.invoke(cmd, ["--help"])

        assert result.exit_code == 0, result.output
        assert " ".join(result.output.split()).count("--undocumented TEXT") == 1


class TestAddOptionsFromConfigNested:
    """Test that add_options_from_config handles nested BaseModel fields."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_nested_dot_notation_options_are_registered(self):
        """Nested model fields should create --parent.child CLI options."""

        @click.command()
        @add_options_from_config(OuterConfig)
        @filter_none_kwargs
        def cmd(**kwargs):
            for k, v in sorted(kwargs.items()):
                click.echo(f"{k}={v}")

        result = self.runner.invoke(cmd, ["--inner.beta=0.5", "--inner.host=localhost"])
        assert result.exit_code == 0, result.output
        assert "inner__beta=0.5" in result.output
        assert "inner__host=localhost" in result.output

    def test_nested_bool_option(self):
        """Nested bool fields should support --parent.field/--no-parent.field."""

        @click.command()
        @add_options_from_config(OuterConfig)
        @filter_none_kwargs
        def cmd(**kwargs):
            for k, v in sorted(kwargs.items()):
                click.echo(f"{k}={v}")

        result = self.runner.invoke(cmd, ["--inner.use-feature"])
        assert result.exit_code == 0, result.output
        assert "inner__use_feature=True" in result.output

    def test_flat_and_nested_options_together(self):
        """Flat and nested options should work together."""

        @click.command()
        @add_options_from_config(OuterConfig)
        @filter_none_kwargs
        def cmd(**kwargs):
            for k, v in sorted(kwargs.items()):
                click.echo(f"{k}={v}")

        result = self.runner.invoke(
            cmd, ["--learning-rate=0.001", "--inner.beta=0.1", "--name=test"]
        )
        assert result.exit_code == 0, result.output
        assert "learning_rate=0.001" in result.output
        assert "inner__beta=0.1" in result.output
        assert "name=test" in result.output

    def test_no_nested_options_passed(self):
        """When no nested options are passed, they should not appear in kwargs."""

        @click.command()
        @add_options_from_config(OuterConfig)
        @filter_none_kwargs
        def cmd(**kwargs):
            click.echo(f"keys={sorted(kwargs.keys())}")

        result = self.runner.invoke(cmd, ["--learning-rate=0.01"])
        assert result.exit_code == 0, result.output
        assert "inner__" not in result.output


class TestLoadCfgNestedKwargs:
    """Test that load_cfg correctly applies nested (double-underscore) kwargs."""

    @staticmethod
    def _apply_nested_kwargs(cfg, kwargs):
        """Helper that mirrors the nested kwargs handling from load_cfg,
        including type coercion for string CLI values."""
        from axolotl.cli.config import _coerce_value

        nested_kwargs: dict = {}
        flat_kwargs: dict = {}
        for key, value in kwargs.items():
            if "__" in key:
                parent, child = key.split("__", 1)
                nested_kwargs.setdefault(parent, {})[child] = value
            else:
                flat_kwargs[key] = value

        cfg_keys = cfg.keys()
        for key, value in flat_kwargs.items():
            if key in cfg_keys:
                cfg[key] = _coerce_value(value, cfg.get(key))

        for parent, children in nested_kwargs.items():
            if cfg[parent] is None:
                cfg[parent] = {}
            if not isinstance(cfg[parent], dict):
                cfg[parent] = {}
            for child_key, child_value in children.items():
                existing = cfg[parent].get(child_key)
                cfg[parent][child_key] = _coerce_value(child_value, existing)

        return cfg

    def test_nested_kwargs_applied_to_cfg(self, tmp_path):
        """Double-underscore kwargs should set nested config values."""
        from axolotl.utils.dict import DictDefault

        cfg = DictDefault({"trl": {"beta": 0.1}, "learning_rate": 0.01})
        # CLI passes strings, so simulate that
        kwargs = {
            "trl__beta": "0.5",
            "trl__host": "192.168.1.1",
            "learning_rate": "0.02",
        }

        cfg = self._apply_nested_kwargs(cfg, kwargs)

        assert cfg["learning_rate"] == 0.02
        assert isinstance(cfg["learning_rate"], float)
        assert cfg["trl"]["beta"] == 0.5
        assert isinstance(cfg["trl"]["beta"], float)
        assert cfg["trl"]["host"] == "192.168.1.1"

    def test_nested_kwargs_creates_parent_if_none(self):
        """If the parent key is None, nested kwargs should create the dict."""
        from axolotl.utils.dict import DictDefault

        cfg = DictDefault({"trl": None, "learning_rate": 0.01})
        cfg = self._apply_nested_kwargs(cfg, {"trl__beta": "0.5"})

        # No existing value, YAML-style inference: "0.5" -> 0.5
        assert cfg["trl"]["beta"] == 0.5
        assert isinstance(cfg["trl"]["beta"], float)

    def test_nested_kwargs_overwrites_string_parent(self):
        """If the parent key is a string, it should be replaced with a dict."""
        from axolotl.utils.dict import DictDefault

        cfg = DictDefault({"trl": "some_string", "learning_rate": 0.01})
        cfg = self._apply_nested_kwargs(cfg, {"trl__beta": "0.5"})

        assert cfg["trl"]["beta"] == 0.5


class TestCoerceValue:
    """Test YAML-style type coercion for CLI string values."""

    def test_coerce_with_existing_float(self):
        from axolotl.cli.config import _coerce_value

        assert _coerce_value("0.5", 0.1) == 0.5
        assert isinstance(_coerce_value("0.5", 0.1), float)

    def test_coerce_with_existing_int(self):
        from axolotl.cli.config import _coerce_value

        assert _coerce_value("42", 10) == 42
        assert isinstance(_coerce_value("42", 10), int)

    def test_coerce_with_existing_bool(self):
        from axolotl.cli.config import _coerce_value

        assert _coerce_value("true", False) is True
        assert _coerce_value("false", True) is False
        assert _coerce_value("1", False) is True
        assert _coerce_value("0", True) is False

    def test_coerce_yaml_inference_no_existing(self):
        """Without an existing value, use YAML-style inference."""
        from axolotl.cli.config import _coerce_value

        assert _coerce_value("true", None) is True
        assert _coerce_value("false", None) is False
        assert _coerce_value("42", None) == 42
        assert isinstance(_coerce_value("42", None), int)
        assert _coerce_value("3.14", None) == 3.14
        assert isinstance(_coerce_value("3.14", None), float)
        assert _coerce_value("null", None) is None
        assert _coerce_value("hello", None) == "hello"

    def test_coerce_non_string_passthrough(self):
        """Non-string values should pass through unchanged."""
        from axolotl.cli.config import _coerce_value

        assert _coerce_value(0.5, 0.1) == 0.5
        assert _coerce_value(True, False) is True
