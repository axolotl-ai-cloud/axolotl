"""Test suite for Axolotl CLI / configuration options"""

import inspect
import json
import re
import unittest
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Tuple

from click.testing import CliRunner, Result

from axolotl.cli import options
from axolotl.utils.config import load_config


class CLITest(unittest.TestCase):
    """Axolotl CLI test suite"""

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        self.default_config_path = (
            Path(__file__).parent / "fixtures/default_config.yaml"
        )

        self.default_config = load_config(self.default_config_path)

        self.runner = CliRunner(
            env={"AXOLOTL_CONFIG": str(self.default_config_path), "LOG_LEVEL": "DEBUG"}
        )
        self.main_module = import_module("axolotl.__main__")

    def test_option_quality_and_style(self):
        """Checks for convention violations that can easily occur when adding new
        configuration options (largely copy/paste problems)
        """
        # Load option defaults
        # default_config = load_config(self.default_config_path)

        # Get a list of callable members from axolotl.cli.options, these are our decorators
        all_members = inspect.getmembers(options)
        callable_options = [
            member[1]
            for member in all_members
            if callable(member[1]) and member[0].endswith("_option")
        ]

        param_decls_seen = {}
        envvars_seen = {}
        docs_seen = {}

        all_decorator_config_names = []

        for current_callable_option in callable_options:
            current_name = current_callable_option.__name__
            decorator = current_callable_option()

            assert hasattr(
                decorator, "click_params"
            ), f"{current_name} was missing the 'click_params' attribute which means it was not build with option_factory()"

            # Verify that we have documentation
            assert (
                current_callable_option.__doc__ is not None
                and len(current_callable_option.__doc__) >= 10
            ), f"'{current_name}' documentation must have a length of at least 10"

            # Verify unique help/documentation strings
            for doc_key, doc_value in docs_seen.items():
                assert (
                    current_callable_option.__doc__ != doc_value
                ), f"'{current_name}' has a duplicate documentation string with '{doc_key}'"

            docs_seen[current_name] = current_callable_option.__doc__

            # Get references to the option metadata so we can test
            decorator_args = decorator.click_params.args
            decorator_kwargs = decorator.click_params.kwargs

            assert (
                decorator_args is not None and len(decorator_args) > 0
            ), f"'{current_name}' is missing param_decls"

            # Parse parameter declarations and check to see if at least one is in our defaults config
            possible_config_names = []
            for param_decl in decorator_args:
                matcher = re.match(r"^-{0,2}([a-zA-Z_][a-zA-Z0-9_]*)", param_decl)
                assert matcher and matcher.group(
                    1
                ), f"Unable to parse option '{param_decl}' from '{current_name}'"
                possible_config_names.append(matcher.group(1))

            # Maintain a list of all config names, we will need it at the end to do a reverse check to verify there are no configuration values that do not
            # have a matching options decorator
            all_decorator_config_names.extend(possible_config_names)

            assert any(
                config_name in self.default_config
                for config_name in possible_config_names
            ), f"No parameters parsed from '{current_name}' ({possible_config_names}) were found in the default config file '{self.default_config_path}'"

            # Verify that no options have the same name
            assert (
                decorator_args[0] not in param_decls_seen
            ), f"'{current_name}' has a duplicate CLI option flag with '{param_decls_seen[decorator_args[0]]}'"
            param_decls_seen[decorator_args[0]] = current_name

            # Verify that the environment variable is present
            assert (
                decorator_kwargs.envvar is not None
            ), f"'{current_name}' envvar attribute is missing"

            # Verify that environment variables have the correct prefix
            assert decorator_kwargs.envvar.startswith(
                "AXOLOTL_"
            ), f"'{current_name}' envvar '{decorator_kwargs.envvar}' is missing AXOLOTL_ prefix"

            # Verify all unique environment variables
            assert (
                decorator_kwargs.envvar not in envvars_seen
            ), f"'{current_name}' has a duplicate environment variable name ({decorator_kwargs.envvar}) with '{envvars_seen[decorator_kwargs.envvar]}'"
            envvars_seen[decorator_kwargs.envvar] = current_name

        # Verify that every default configuration option has a decorator in options.py
        for default_config_key in self.default_config.keys():
            if default_config_key not in all_decorator_config_names:
                fail_message = f"'{default_config_key}' does not have a matching decorator in options.py"
                print(f"WARNING: {fail_message}")

                # Once training is integrated into the Click CLI this needs to be a fail
                # self.fail(fail_message)

    def _run_parse_json(self, args: List[str], **kwargs) -> Tuple[Result, Dict]:
        """Helper method to invoke the Click CLI"""
        result = self.runner.invoke(self.main_module.cli, args, **kwargs)

        assert (
            result.exit_code == 0
        ), f"Axolotol CLI test exited with status '{result.exit_code}', args: '{args}'"

        # Get the last output line, expected to be a json string ... this is similar to
        # how tools like Airflow parse command output
        last_line = [line for line in result.output.split("\n") if line != ""][-1]

        # Parse output into a Python dictionary for further assertions
        try:
            output_dict = json.loads(last_line)

        # pylint: disable=bare-except
        except:  # noqa: E722
            self.fail(
                f"Unable to parse CLI command output to json. args: '{args}' Raw output: '{result.output}'"
            )

        return (result, output_dict)

    def test_apply_defaults(self):
        """Validates that the CLI parser with no external options will return an unmodified version
        of the Axolotl configuration object"""

        # Run CLI and print all options
        _, output_dict = self._run_parse_json(args=["system", "config", "--no-pretty"])

        def find_first_difference(str1, str2):
            min_length = min(len(str1), len(str2))

            for i in range(min_length):
                if str1[i] != str2[i]:
                    return i

            # If we reach this point, all characters up to min_length are the same
            # Return the length of the shorter string
            return min_length

        # I'm sure there is a better way to do this but to do a deep compare I'll dump both
        # expected and actual to strings
        try:
            output_str = json.dumps(output_dict, sort_keys=True)

        # pylint: disable=bare-except
        except:  # noqa: E722
            self.fail(f"Unable to decode json output. Raw CLI output:\n{output_dict}")

        defaults_str = json.dumps(self.default_config, sort_keys=True)

        # It is worthwhile to output a helpful error message here so nobody gets lost
        if output_str != defaults_str:
            diff_center = 30
            diff_index = find_first_difference(output_str, defaults_str)
            output_diff = output_str[
                diff_index - diff_center : diff_index + diff_center
            ]
            defaults_diff = defaults_str[
                diff_index - diff_center : diff_index + diff_center
            ]

            fail_message = f"Differences detected when comparing defaults from '{self.default_config_path}' and CLI output. Diff:\n  CLI: {output_diff}\n  CFG: {defaults_diff}"
            self.fail(fail_message)

    def test_list_overrides(self):
        """Validate override logic with multiple list options"""
        _, output_dict = self._run_parse_json(args=["system", "config", "--no-pretty"])

        # Run assertions on the default value
        assert len(output_dict["lora_target_modules"]) == 2
        assert "q_proj" in output_dict["lora_target_modules"]
        assert "v_proj" in output_dict["lora_target_modules"]

        _, output_dict = self._run_parse_json(
            args=[
                "system",
                "config",
                "--no-pretty",
                "--lora_target_module=k_proj",
                "--lora_target_module=o_proj",
                "--lora_target_module=gate_proj",
            ]
        )

        # Run assertions on the CLI param overide values
        assert len(output_dict["lora_target_modules"]) == 3
        assert "k_proj" in output_dict["lora_target_modules"]
        assert "o_proj" in output_dict["lora_target_modules"]
        assert "gate_proj" in output_dict["lora_target_modules"]

        _, output_dict = self._run_parse_json(
            args=[
                "system",
                "config",
                "--no-pretty",
            ],
            env={"AXOLOTL_LORA_TARGET_MODULES": "k_proj o_proj gate_proj"},
        )

        # Run assertions on the environment override values
        assert len(output_dict["lora_target_modules"]) == 3
        assert "k_proj" in output_dict["lora_target_modules"]
        assert "o_proj" in output_dict["lora_target_modules"]
        assert "gate_proj" in output_dict["lora_target_modules"]

        _, output_dict = self._run_parse_json(
            args=[
                "system",
                "config",
                "--no-pretty",
                "--lora_target_module=down_proj",
            ],
            env={"AXOLOTL_LORA_TARGET_MODULES": "k_proj o_proj gate_proj"},
        )

        # Run assertions on the environment + CLI override values (CLI should win)
        assert len(output_dict["lora_target_modules"]) == 1
        assert "down_proj" in output_dict["lora_target_modules"]

        # Run case where we want to override the defaults to nothing / empty list
        # Scenario 1, single CLI override
        _, output_dict = self._run_parse_json(
            args=[
                "system",
                "config",
                "--no-pretty",
                "--lora_target_module=",
            ]
        )

        assert len(output_dict["lora_target_modules"]) == 0

        # Scenario 2, multiple CLI override
        _, output_dict = self._run_parse_json(
            args=[
                "system",
                "config",
                "--no-pretty",
                "--lora_target_module=",
                "--lora_target_module=",
            ]
        )

        assert len(output_dict["lora_target_modules"]) == 0

        # Scenario 3, multiple CLI override
        _, output_dict = self._run_parse_json(
            args=[
                "system",
                "config",
                "--no-pretty",
                "--lora_target_module=",
                "--lora_target_module=vpro_j",
            ]
        )

        assert len(output_dict["lora_target_modules"]) == 1
        assert "vpro_j" in output_dict["lora_target_modules"]
