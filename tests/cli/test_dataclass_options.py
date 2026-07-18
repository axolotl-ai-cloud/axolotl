"""Tests for dataclass-based CLI option handling."""

from dataclasses import dataclass, field
from typing import Optional

import click
from click.testing import CliRunner

from axolotl.cli.utils.args import add_options_from_dataclass


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
