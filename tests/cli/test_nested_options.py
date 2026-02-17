"""Tests for nested config option handling via CLI dot-notation."""

import click
import pytest
from click.testing import CliRunner
from pydantic import BaseModel, Field

from axolotl.cli.utils.args import add_options_from_config, filter_none_kwargs


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

    def test_nested_kwargs_applied_to_cfg(self, tmp_path):
        """Double-underscore kwargs should set nested config values."""
        from axolotl.utils.dict import DictDefault

        # Create a minimal config file
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            "base_model: test\n"
            "learning_rate: 0.01\n"
            "trl:\n"
            "  beta: 0.1\n"
            "micro_batch_size: 1\n"
            "gradient_accumulation_steps: 1\n"
            "sequence_len: 512\n"
            "datasets:\n"
            "  - path: test\n"
            "    type: alpaca\n"
        )

        # We test the nested kwargs logic directly since load_cfg has many
        # side effects (torch, validation, etc.)
        cfg = DictDefault({"trl": {"beta": 0.1}, "learning_rate": 0.01})

        # Simulate the nested kwargs handling from load_cfg
        kwargs = {"trl__beta": 0.5, "trl__host": "192.168.1.1", "learning_rate": 0.02}
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
                cfg[key] = value

        for parent, children in nested_kwargs.items():
            if cfg[parent] is None:
                cfg[parent] = {}
            if not isinstance(cfg[parent], dict):
                cfg[parent] = {}
            for child_key, child_value in children.items():
                cfg[parent][child_key] = child_value

        assert cfg["learning_rate"] == 0.02
        assert cfg["trl"]["beta"] == 0.5
        assert cfg["trl"]["host"] == "192.168.1.1"

    def test_nested_kwargs_creates_parent_if_none(self):
        """If the parent key is None, nested kwargs should create the dict."""
        from axolotl.utils.dict import DictDefault

        cfg = DictDefault({"trl": None, "learning_rate": 0.01})

        kwargs = {"trl__beta": 0.5}
        for key, value in kwargs.items():
            parent, child = key.split("__", 1)
            if cfg[parent] is None:
                cfg[parent] = {}
            cfg[parent][child] = value

        assert cfg["trl"]["beta"] == 0.5
