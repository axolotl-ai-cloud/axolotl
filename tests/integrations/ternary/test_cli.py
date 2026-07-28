"""CPU-only tests for the `axolotl ternary` CLI and the shipped example configs."""

import json
import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from axolotl.cli.main import cli
from axolotl.cli.plugins import BUILTIN_COMMANDS
from axolotl.integrations.ternary.ptq.damage import (
    DEFAULT_SEQUENCE_LEN,
    DEFAULT_SEQUENCES,
)
from axolotl.utils.dict import DictDefault

EXAMPLES_DIR = Path(__file__).parents[3] / "examples" / "ternary"
EXAMPLE_CONFIGS = sorted(EXAMPLES_DIR.glob("*.yaml"))

MINIMAL_CONFIG = """
base_model: HuggingFaceTB/SmolLM2-135M
plugins:
  - axolotl.integrations.ternary.TernaryPlugin
ternary:
  export:
    formats: [master_bf16, hf_bitnet]
datasets:
  - path: mhenrichsen/alpaca_2k_test
    type: alpaca
sequence_len: 2048
max_steps: 1
micro_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1e-4
save_only_model: true
special_tokens:
  pad_token: <|endoftext|>
"""


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text(MINIMAL_CONFIG)
    return path


@pytest.fixture
def stub_export(monkeypatch, tmp_path):
    """Replaces `run_export` so invocation is tested without loading a checkpoint."""
    calls = []

    def _run_export(cfg, formats=None, master_dir=None, output_dir=None):
        calls.append(
            {
                "cfg": cfg,
                "formats": formats,
                "master_dir": master_dir,
                "output_dir": output_dir,
            }
        )
        return {"hf_bitnet": tmp_path / "hf_bitnet"}

    monkeypatch.setattr(
        "axolotl.integrations.ternary.export.run_export", _run_export, raising=True
    )
    return calls


@pytest.fixture
def stub_damage_map(monkeypatch):
    """Replaces `damage_map` so the CLI wiring is tested without loading a model."""
    from axolotl.integrations.ternary.ptq.damage import DamageReport, DamageRow

    calls = []

    def _damage_map(cfg, num_sequences, sequence_len, dataset):
        calls.append(
            {
                "cfg": cfg,
                "num_sequences": num_sequences,
                "sequence_len": sequence_len,
                "dataset": dataset,
            }
        )
        return DamageReport(
            baseline_perplexity=9.5,
            rows=[DamageRow("weights", 12.5, 3.0)],
            sequences=num_sequences,
            sequence_len=sequence_len,
        )

    monkeypatch.setattr(
        "axolotl.integrations.ternary.ptq.damage.damage_map", _damage_map, raising=True
    )
    return calls


@pytest.fixture
def stub_load_cfg(monkeypatch):
    """Replaces `load_cfg` with a plain YAML read (no validation, no GPU probing)."""

    def _load_cfg(config, **kwargs):
        with open(config, encoding="utf-8") as file:
            return DictDefault(yaml.safe_load(file))

    monkeypatch.setattr("axolotl.cli.config.load_cfg", _load_cfg, raising=True)


def test_ternary_listed_in_help(runner):
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "ternary" in result.output


def test_help_does_not_import_the_command_module(runner, monkeypatch):
    # delitem restores the modules afterwards: leaving them purged makes the plugin's
    # lazy imports build a second set of classes, silently breaking isinstance elsewhere
    for module in list(sys.modules):
        if module.startswith("axolotl.integrations.ternary"):
            monkeypatch.delitem(sys.modules, module)

    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "axolotl.integrations.ternary.cli" not in sys.modules
    assert "convert and export ternary (1.58-bit) models" in result.output


def test_registry_entry_matches_the_group_docstring():
    from axolotl.integrations.ternary.cli import ternary

    spec = BUILTIN_COMMANDS["ternary"]

    assert spec.target == "axolotl.integrations.ternary.cli:ternary"
    assert spec.short_help == ternary.__doc__.strip()


def test_group_help_lists_the_commands(runner):
    result = runner.invoke(cli, ["ternary", "--help"])

    assert result.exit_code == 0
    assert "export" in result.output
    assert "damage-map" in result.output


def test_export_help_lists_options(runner):
    result = runner.invoke(cli, ["ternary", "export", "--help"])

    assert result.exit_code == 0
    assert "--format" in result.output
    assert "--output-dir" in result.output


def test_export_reaches_run_export(runner, config_path, stub_load_cfg, stub_export):
    result = runner.invoke(cli, ["ternary", "export", str(config_path)])

    assert result.exit_code == 0, result.output
    assert len(stub_export) == 1
    assert stub_export[0]["formats"] is None
    assert stub_export[0]["output_dir"] is None
    assert stub_export[0]["cfg"].base_model == "HuggingFaceTB/SmolLM2-135M"
    assert "hf_bitnet" in result.output


def test_export_passes_formats_and_output_dir(
    runner, config_path, tmp_path, stub_load_cfg, stub_export
):
    result = runner.invoke(
        cli,
        [
            "ternary",
            "export",
            str(config_path),
            "--format",
            "gguf_tq2_0",
            "--format",
            "i2_s",
            "--output-dir",
            str(tmp_path / "exports"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert stub_export[0]["formats"] == ["gguf_tq2_0", "i2_s"]
    assert stub_export[0]["output_dir"] == str(tmp_path / "exports")


def test_export_rejects_unknown_format(runner, config_path, stub_load_cfg, stub_export):
    result = runner.invoke(
        cli, ["ternary", "export", str(config_path), "--format", "awq"]
    )

    assert result.exit_code != 0
    assert not stub_export


def test_export_requires_the_plugin(runner, tmp_path, stub_load_cfg, stub_export):
    path = tmp_path / "no-plugin.yml"
    path.write_text(
        MINIMAL_CONFIG.replace("  - axolotl.integrations.ternary.TernaryPlugin\n", "")
    )

    result = runner.invoke(cli, ["ternary", "export", str(path)])

    assert result.exit_code != 0
    assert "does not enable the ternary plugin" in result.output
    assert not stub_export


def test_export_requires_an_existing_config(runner, tmp_path, stub_export):
    result = runner.invoke(cli, ["ternary", "export", str(tmp_path / "missing.yml")])

    assert result.exit_code != 0
    assert not stub_export


def test_damage_map_help_lists_options(runner):
    result = runner.invoke(cli, ["ternary", "damage-map", "--help"])

    assert result.exit_code == 0
    assert "--num-sequences" in result.output
    assert "--sequence-len" in result.output
    assert "--dataset" in result.output
    assert "--json" in result.output


def test_damage_map_prints_a_table(runner, config_path, stub_load_cfg, stub_damage_map):
    result = runner.invoke(cli, ["ternary", "damage-map", str(config_path)])

    assert result.exit_code == 0, result.output
    assert len(stub_damage_map) == 1
    assert stub_damage_map[0]["num_sequences"] == DEFAULT_SEQUENCES
    assert stub_damage_map[0]["sequence_len"] == DEFAULT_SEQUENCE_LEN
    assert stub_damage_map[0]["dataset"] is None
    assert stub_damage_map[0]["cfg"].base_model == "HuggingFaceTB/SmolLM2-135M"
    assert "baseline" in result.output
    assert "+3.0000" in result.output


def test_damage_map_passes_its_options(
    runner, config_path, stub_load_cfg, stub_damage_map
):
    result = runner.invoke(
        cli,
        [
            "ternary",
            "damage-map",
            str(config_path),
            "--num-sequences",
            "8",
            "--sequence-len",
            "256",
            "--dataset",
            "wikitext",
        ],
    )

    assert result.exit_code == 0, result.output
    assert stub_damage_map[0]["num_sequences"] == 8
    assert stub_damage_map[0]["sequence_len"] == 256
    assert stub_damage_map[0]["dataset"] == "wikitext"


def test_damage_map_emits_json(runner, config_path, stub_load_cfg, stub_damage_map):
    result = runner.invoke(cli, ["ternary", "damage-map", str(config_path), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["baseline_perplexity"] == 9.5
    assert payload["rows"][0]["scope"] == "weights"


def test_damage_map_requires_the_plugin(
    runner, tmp_path, stub_load_cfg, stub_damage_map
):
    path = tmp_path / "no-plugin.yml"
    path.write_text(
        MINIMAL_CONFIG.replace("  - axolotl.integrations.ternary.TernaryPlugin\n", "")
    )

    result = runner.invoke(cli, ["ternary", "damage-map", str(path)])

    assert result.exit_code != 0
    assert "does not enable the ternary plugin" in result.output
    assert not stub_damage_map


def test_damage_map_rejects_a_zero_sequence_count(
    runner, config_path, stub_load_cfg, stub_damage_map
):
    result = runner.invoke(
        cli, ["ternary", "damage-map", str(config_path), "--num-sequences", "0"]
    )

    assert result.exit_code != 0
    assert not stub_damage_map


def test_examples_exist():
    assert {path.name for path in EXAMPLE_CONFIGS} == {
        "llama-3.2-1b-qat.yaml",
        "llama-3.2-1b-ptq-init.yaml",
        "qwen3-4b-distill.yaml",
    }
    assert (EXAMPLES_DIR / "README.md").is_file()


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_example_config_validates(path):
    from axolotl.utils.config import prepare_plugins, validate_config

    cfg = DictDefault(yaml.safe_load(path.read_text(encoding="utf-8")))
    prepare_plugins(cfg)
    validated = validate_config(cfg)

    assert "axolotl.integrations.ternary.TernaryPlugin" in validated.plugins
    assert validated.ternary is not None
    assert validated.adapter is None
    assert validated.save_only_model is True


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_example_config_export_formats_are_known(path):
    from axolotl.integrations.ternary.args import resolve_ternary_config

    cfg = DictDefault(yaml.safe_load(path.read_text(encoding="utf-8")))
    ternary_cfg = resolve_ternary_config(cfg)

    assert ternary_cfg.export.formats
    assert ternary_cfg.export.formats[0] == "master_bf16"


def test_ptq_init_example_configures_the_calibrated_fit():
    from axolotl.integrations.ternary.args import resolve_ternary_config

    path = EXAMPLES_DIR / "llama-3.2-1b-ptq-init.yaml"
    ternary_cfg = resolve_ternary_config(
        DictDefault(yaml.safe_load(path.read_text(encoding="utf-8")))
    )

    assert ternary_cfg.init == "ternary_fit_calibrated"
    assert ternary_cfg.init_calibration.num_sequences == 128
    assert ternary_cfg.init_calibration.sequence_len == 2048
    assert ternary_cfg.init_calibration.dataset is None
    assert ternary_cfg.distill.schedule == "anchored"
    assert ternary_cfg.distill.anchor_start == 0.9
    assert "mask_sign" in ternary_cfg.export.formats
