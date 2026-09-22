"""Cloud provider discovery, dispatch, and built-in compatibility."""

import subprocess
import sys
from importlib.metadata import EntryPoint
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from axolotl.cli import cloud
from axolotl.cli.cloud import registry
from axolotl.cli.cloud.base import Cloud, CloudLauncher


class RecordingCloud(CloudLauncher):
    """Minimal train-only provider."""

    def train(self, config_yaml, **kwargs):
        self.call = (config_yaml, kwargs)


@pytest.fixture
def external_provider(monkeypatch):
    point = EntryPoint(
        name="external",
        value="tests.cli.test_cloud_providers:RecordingCloud",
        group=registry.ENTRY_POINT_GROUP,
    )
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [point])
    return point


def test_external_provider_accepts_plain_config(external_provider):
    config = {"provider": "external", "backend": {"endpoint": "localhost"}}
    provider = registry.load_cloud_provider(config)
    assert isinstance(provider, RecordingCloud)
    assert provider.config == config
    assert provider.config is not config


def test_builtin_names_are_reserved(monkeypatch, external_provider):
    monkeypatch.setattr(
        registry, "BUILTIN_PROVIDERS", {"external": external_provider.value}
    )
    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda **kwargs: pytest.fail("Built-ins should not need plugin discovery"),
    )
    assert isinstance(
        registry.load_cloud_provider({"provider": "external"}), RecordingCloud
    )


def test_duplicate_provider_is_rejected(monkeypatch, external_provider):
    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda **kwargs: [external_provider, external_provider],
    )
    with pytest.raises(ValueError, match="Multiple cloud providers"):
        registry.load_cloud_provider({"provider": "external"})


def test_unknown_provider_has_install_hint(external_provider):
    with pytest.raises(ValueError, match="Install a package"):
        registry.load_cloud_provider({"provider": "missing"})


@pytest.mark.parametrize("name", [True, False, 0, 12, [], ["modal"], {"name": "modal"}])
def test_invalid_provider_name(name):
    with pytest.raises(ValueError, match="must be a string"):
        registry.load_cloud_provider({"provider": name})


def test_invalid_provider_type(monkeypatch):
    monkeypatch.setattr(registry, "BUILTIN_PROVIDERS", {"invalid": "builtins:dict"})
    with pytest.raises(TypeError, match="must be a CloudLauncher subclass"):
        registry.load_cloud_provider({"provider": "invalid"})


def test_unselected_provider_is_not_imported(monkeypatch, external_provider):
    broken = EntryPoint(
        name="broken", value="missing_provider:Cloud", group=registry.ENTRY_POINT_GROUP
    )
    monkeypatch.setattr(
        registry, "entry_points", lambda **kwargs: [broken, external_provider]
    )
    assert isinstance(
        registry.load_cloud_provider({"provider": "external"}), RecordingCloud
    )


def test_selected_provider_import_error_is_preserved(monkeypatch):
    monkeypatch.setattr(
        registry, "BUILTIN_PROVIDERS", {"broken": "missing_provider:Cloud"}
    )
    with pytest.raises(ModuleNotFoundError, match="missing_provider"):
        registry.load_cloud_provider({"provider": "broken"})


@pytest.mark.parametrize("operation", ["preprocess", "lm_eval"])
def test_train_only_provider_optional_operations(external_provider, operation):
    provider = registry.load_cloud_provider({"provider": "external"})
    with pytest.raises(NotImplementedError, match="does not support"):
        getattr(provider, operation)("base_model: example")


@pytest.mark.parametrize("contents", ["", "[]", "modal", "false"])
def test_cloud_config_requires_mapping(tmp_path, contents):
    config = tmp_path / "cloud.yaml"
    config.write_text(contents)
    with pytest.raises(ValueError, match="YAML mapping"):
        cloud.load_cloud_cfg(config)


@pytest.mark.parametrize("operation", ["train", "preprocess", "lm_eval"])
def test_all_commands_dispatch_selected_provider(tmp_path, monkeypatch, operation):
    cloud_path = tmp_path / "cloud.yaml"
    cloud_path.write_text("provider: external\nbackend:\n  endpoint: localhost\n")
    train_path = tmp_path / "train.yaml"
    original = "plugins:\n  - example.RemoteTrainerPlugin\nremote:\n  transport: http\n"
    train_path.write_text(original)
    provider = MagicMock(spec=CloudLauncher)

    def load(config):
        assert type(config) is dict
        assert type(config["backend"]) is dict
        assert config["provider"] == "external"
        return provider

    monkeypatch.setattr(cloud, "load_cloud_provider", load)
    getattr(cloud, f"do_cli_{operation}")(cloud_path, train_path)
    args, _ = getattr(provider, operation).call_args
    assert args == (original,)


def test_train_forwards_launcher_overrides_and_mounts(tmp_path, monkeypatch):
    cloud_path = tmp_path / "cloud.yaml"
    cloud_path.write_text("provider: external\n")
    train_path = tmp_path / "train.yaml"
    train_path.write_text("base_model: example\n")
    provider = RecordingCloud({})
    monkeypatch.setattr(cloud, "load_cloud_provider", lambda config: provider)
    cloud.do_cli_train(
        cloud_path,
        train_path,
        launcher="torchrun",
        launcher_args=["--nproc_per_node", "2"],
        cwd=str(tmp_path),
        max_steps=5,
    )
    assert provider.call == (
        train_path.read_text(),
        {
            "launcher": "torchrun",
            "launcher_args": ["--nproc_per_node", "2"],
            "local_dirs": {"/workspace/mounts": str(tmp_path)},
            "max_steps": 5,
        },
    )


def test_default_provider_and_legacy_imports(monkeypatch):
    from axolotl.cli.cloud.baseten import BasetenCloud as LegacyBaseten
    from axolotl.cli.cloud.modal_ import ModalCloud as LegacyModal
    from axolotl.integrations.baseten.cloud import BasetenCloud
    from axolotl.integrations.modal import cloud as modal_cloud

    monkeypatch.setattr(modal_cloud.modal, "App", MagicMock())
    assert LegacyModal is modal_cloud.ModalCloud
    assert LegacyBaseten is BasetenCloud
    assert isinstance(registry.load_cloud_provider({}), LegacyModal)
    assert isinstance(
        registry.load_cloud_provider({"provider": "baseten"}), LegacyBaseten
    )


def test_baseten_packages_templates_and_training_plugins(monkeypatch):
    from axolotl.integrations.baseten import cloud as baseten_cloud

    original = "plugins:\n  - example.RemoteTrainerPlugin\n"
    calls = []

    def submit(command, cwd, check):
        directory = Path(cwd)
        assert command == ["truss", "train", "push", "train_sft.py"]
        assert (directory / "train.yaml").read_text() == original
        assert (directory / "run.sh").is_file()
        assert (directory / "train_sft.py").is_file()
        config = yaml.safe_load((directory / "cloud.yaml").read_text())
        assert config["launcher"] == "torchrun"
        assert config["launcher_args"] == ["--nproc_per_node", "2"]
        calls.append(command)

    monkeypatch.setattr(baseten_cloud.subprocess, "run", submit)
    provider = registry.load_cloud_provider({"provider": "baseten"})
    provider.train(
        original, launcher="torchrun", launcher_args=["--nproc_per_node", "2"]
    )
    assert len(calls) == 1


def test_discovery_does_not_import_training_or_modal():
    script = """
import sys
from axolotl.cli.cloud import load_cloud_provider
load_cloud_provider({"provider": "baseten"})
assert not {"torch", "transformers", "peft", "modal"}.intersection(sys.modules)
assert "axolotl.integrations.base" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("operation", ["train", "preprocess", "lm_eval"])
def test_modal_dispatches_moved_remote_functions(monkeypatch, operation):
    from axolotl.integrations.modal import cloud as modal_cloud

    app = MagicMock()
    provider = modal_cloud.ModalCloud({"gpu": "h100"}, app=app)
    decorator = MagicMock()
    monkeypatch.setattr(provider, "get_train_env", lambda *args: decorator)
    monkeypatch.setattr(provider, "get_preprocess_env", lambda: decorator)
    monkeypatch.setattr(modal_cloud.modal, "enable_output", MagicMock())
    original = "plugins:\n  - example.RemoteTrainerPlugin\n"
    getattr(provider, operation)(original)
    decorator.assert_called_once_with(getattr(modal_cloud, f"_{operation}"))
    assert decorator.return_value.remote.call_args.args == (original,)
    app.run.assert_called_once_with(detach=True)


def test_provider_failure_propagates(tmp_path, monkeypatch):
    cloud_path = tmp_path / "cloud.yaml"
    cloud_path.write_text("provider: external\n")
    train_path = tmp_path / "train.yaml"
    train_path.write_text("base_model: example\n")
    provider = MagicMock(spec=CloudLauncher)
    provider.train.side_effect = RuntimeError("submission failed")
    monkeypatch.setattr(cloud, "load_cloud_provider", lambda config: provider)
    with pytest.raises(RuntimeError, match="submission failed"):
        cloud.do_cli_train(cloud_path, train_path)


def test_legacy_cloud_subclass_loads(monkeypatch):
    class LegacyCloud(Cloud):
        def train(self, config_yaml, **kwargs):
            pass

    assert Cloud is CloudLauncher
    monkeypatch.setattr(EntryPoint, "load", lambda self: LegacyCloud)
    provider = registry.load_cloud_provider({"provider": "modal"})
    assert isinstance(provider, CloudLauncher)
    assert provider.config == {"provider": "modal"}
