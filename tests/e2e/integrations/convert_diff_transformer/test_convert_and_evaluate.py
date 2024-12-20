"""End-to-end tests for differential transformer conversion and evaluation."""
# pylint: disable=duplicate-code

from pathlib import Path

import yaml
from pytest import approx

from axolotl.cli import load_cfg
from axolotl.cli.evaluate import do_evaluate
from axolotl.cli.integrations.convert_diff_transformer import convert_diff_transformer
from axolotl.common.cli import ConvertDiffTransformerCliArgs, EvaluateCliArgs


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
    _, debug_info = convert_diff_transformer(cfg, cli_args, str(config_path))

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
