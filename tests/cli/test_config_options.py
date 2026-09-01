"""Tests for generated CLI config option metadata."""

from axolotl.cli.config_options import AXOLOTL_CONFIG_CLI_OPTIONS
from axolotl.cli.generate_config_options import generate_options


def test_fp8_config_recipe_cli_option_is_generated():
    checked_in_option = next(
        option
        for option in AXOLOTL_CONFIG_CLI_OPTIONS
        if option[0] == ("--fp8-config.recipe",)
    )
    generated_option = next(
        option for option in generate_options() if option[0] == ("--fp8-config.recipe",)
    )

    assert checked_in_option == generated_option
    assert checked_in_option[1] == "fp8_config__recipe"
    assert checked_in_option[2] is None
    assert "tensorwise" in checked_in_option[3]
    assert "rowwise_with_gw_hp" in checked_in_option[3]
