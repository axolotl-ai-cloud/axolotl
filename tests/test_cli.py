import json
from pathlib import Path
from typing import Any, Dict, List
import unittest
import inspect
import re

import pytest


import axolotl.cli.options as options
from axolotl.utils.config import load_config


class CLITest(unittest.TestCase):
    def test_option_quality_and_style(self):
        # Load option defaults
        default_config_path = Path(__file__).parent / "fixtures/default_config.yaml"
        default_config = load_config(Path(__file__).parent / "fixtures/default_config.yaml")

        # Get a list of callable members from axolotl.cli.options, these are our decorators
        all_members = inspect.getmembers(options)
        callable_options = [
            member[1] for member in all_members if callable(member[1]) and member[0].endswith("_option")
        ]

        param_decls_seen = {}
        envvars_seen = {}

        for current_callable_option in callable_options:
            current_name = current_callable_option.__name__
            decorator = current_callable_option()

            assert hasattr(
                decorator, "click_params"
            ), f"{current_name} was missing the 'click_params' attribute which means it was not build with option_factory()"

            # Verify that we have documentation
            assert (
                current_callable_option.__doc__ is not None and len(current_callable_option.__doc__) >= 10
            ), f"{current_name} documentation must have a length of at least 10"

            # Get references to the option metadata so we can test
            decorator_args = decorator.click_params.args
            decorator_kwargs = decorator.click_params.kwargs

            assert decorator_args is not None and len(decorator_args) > 0, f"{current_name} is missing param_decls"

            # Parse parameter declarations and check to see if at least one is in our defaults config
            possible_config_names = []
            for param_decl in decorator_args:
                matcher = re.match(r"^-{0,2}([a-zA-Z_][a-zA-Z0-9_]*)", param_decl)
                assert matcher and matcher.group(1), f"Unable to parse option '{param_decl}' from {current_name}"
                possible_config_names.append(matcher.group(1))

            assert any(
                [config_name in default_config.keys() for config_name in possible_config_names]
            ), f"No parameters parsed from {current_name} ({possible_config_names}) were found in the default config file {default_config_path}"

            # Verify that no options have the same name
            assert (
                decorator_args[0] not in param_decls_seen
            ), f"{current_name} has a duplicate CLI option flag with {param_decls_seen[decorator_args[0]]}"
            param_decls_seen[decorator_args[0]] = current_name

            # Verify that the environment variable is present
            assert decorator_kwargs.envvar is not None, f"{current_name} envvar attribute is missing"

            # Verify that environment variables have the correct prefix
            assert decorator_kwargs.envvar.startswith(
                "AXOLOTL_"
            ), f"{current_name} envvar '{decorator_kwargs.envvar}' is missing AXOLOTL_ prefix"

            # Verify all unique environment variables
            assert (
                decorator_kwargs.envvar not in envvars_seen
            ), f"{current_name} has a duplicate environment variable name ({decorator_kwargs.envvar}) with {envvars_seen[decorator_kwargs.envvar]}"
            envvars_seen[decorator_kwargs.envvar] = current_name

    def test_option_overrides(self):
        # TODO: writeme
        pass
