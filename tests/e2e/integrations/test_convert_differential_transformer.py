"""End-to-end tests for differential transformer conversion."""
# pylint: disable=redefined-outer-name

from pathlib import Path
from typing import Optional

import pytest
import yaml
from pytest import approx

from axolotl.cli import load_cfg
from axolotl.cli.evaluate import do_evaluate
from axolotl.cli.integrations.convert_differential_transformer import (
    convert_differential_transformer,
)
from axolotl.common.cli import ConvertDiffTransformerCliArgs, EvaluateCliArgs


@pytest.fixture()
def base_config():
    """Basic config for testing."""
    return {
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "plugins": [
            "axolotl.integrations.differential_transformer.DifferentialTransformerPlugin",
        ],
        "datasets": [
            {
                "path": "axolotl-ai-co/alpaca_100_test",
                "type": "alpaca",
            },
        ],
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "val_set_size": 0.1,
        "micro_batch_size": 1,
        "sequence_len": 2048,
        "special_tokens": {
            "pad_token": "<|endoftext|>",
        },
    }


def test_conversion_cli_basic(tmp_path: Path, base_config):
    output_dir = tmp_path / "converted"
    base_config["output_dir"] = str(output_dir)

    config_path = tmp_path / "config.yml"
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.dump(base_config, file)

    # Load config the same way do_cli does
    cfg = load_cfg(str(config_path))

    # Create CLI args
    cli_args = ConvertDiffTransformerCliArgs()

    # Call convert_differential_transformer directly
    _, debug_info = convert_differential_transformer(cfg, cli_args, str(config_path))

    assert not debug_info
    assert (output_dir / "model.safetensors").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "axolotl_config.yml").exists()


def test_conversion_cli_debug(tmp_path: Path, base_config):
    output_dir = tmp_path / "converted"
    base_config["output_dir"] = str(output_dir)

    config_path = tmp_path / "config.yml"
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.dump(base_config, file)

    # Load config the same way do_cli does
    cfg = load_cfg(str(config_path))

    # Create CLI args
    cli_args = ConvertDiffTransformerCliArgs(debug=True)

    # Call convert_differential_transformer directly
    _, debug_info = convert_differential_transformer(cfg, cli_args, str(config_path))

    assert not debug_info["generations_match"]
    assert not debug_info["match_expected"]
    assert (output_dir / "model.safetensors").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "axolotl_config.yml").exists()


def test_conversion_cli_reproduce(tmp_path: Path, base_config):
    output_dir = tmp_path / "converted"
    base_config["output_dir"] = str(output_dir)

    config_path = tmp_path / "config.yml"
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.dump(base_config, file)

    cfg = load_cfg(str(config_path))
    cli_args = ConvertDiffTransformerCliArgs(
        debug=True, zero_init=True, sublayer_norm=False
    )
    _, debug_info = convert_differential_transformer(cfg, cli_args, str(config_path))

    assert debug_info["generations_match"] is True
    assert (output_dir / "model.safetensors").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "axolotl_config.yml").exists()


@pytest.mark.parametrize(
    "attention", ["eager_attention", "sdp_attention", "flash_attention"]
)
def test_conversion_cli_repoduce_attentions(
    tmp_path: Path, base_config, attention: Optional[str]
):
    output_dir = tmp_path / "converted"
    base_config["output_dir"] = str(output_dir)
    base_config[attention] = True

    config_path = tmp_path / "config.yml"
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.dump(base_config, file)

    cfg = load_cfg(str(config_path))
    cli_args = ConvertDiffTransformerCliArgs(
        debug=True, zero_init=True, sublayer_norm=False
    )
    _, debug_info = convert_differential_transformer(cfg, cli_args, str(config_path))

    assert debug_info["generations_match"] is True
    assert (output_dir / "model.safetensors").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "axolotl_config.yml").exists()


def test_conversion_and_eval_cli(tmp_path: Path, base_config):
    output_dir = tmp_path / "converted"
    base_config["output_dir"] = str(output_dir)

    config_path = tmp_path / "config.yml"
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.dump(base_config, file)

    cfg = load_cfg(str(config_path))
    cli_args = ConvertDiffTransformerCliArgs(
        debug=True, zero_init=True, sublayer_norm=False
    )
    _, debug_info = convert_differential_transformer(cfg, cli_args, str(config_path))

    assert debug_info["generations_match"] is True
    assert (output_dir / "model.safetensors").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "axolotl_config.yml").exists()

    eval_cfg = load_cfg(str(output_dir))
    eval_cli_args = EvaluateCliArgs()
    all_metrics = do_evaluate(eval_cfg, eval_cli_args)

    assert list(all_metrics.keys()) == [
        "train_loss",
        "train_model_preparation_time",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "eval_loss",
        "eval_model_preparation_time",
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
    ]
    assert all_metrics["train_loss"] == approx(1.7307, rel=1e-4)
    assert all_metrics["eval_loss"] == approx(1.8387, rel=1e-4)
