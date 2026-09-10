"""Nebius launch and remote-bootstrap tests without a cloud account or GPU."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from axolotl.cli.cloud import (
    do_cli_lm_eval,
    do_cli_preprocess,
    do_cli_train,
    load_cloud_provider,
)
from axolotl.cli.cloud.nebius import NebiusCloud, runner
from axolotl.utils.dict import DictDefault


@pytest.fixture
def cloud_config():
    return {
        "provider": "nebius",
        "image": "axolotlai/axolotl:test-image",
        "platform": "gpu-h100-sxm",
        "preset": "1gpu-16vcpu-200gb",
    }


@pytest.fixture
def training_config():
    return {"base_model": "example/model", "output_dir": "lora output", "max_steps": 30}


@pytest.fixture
def capture_launch(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "axolotl.cli.cloud.nebius.shutil.which", lambda _: "/bin/nebius"
    )

    def run(command, cwd, check):
        captured["command"] = command
        captured["context"] = {p.name: p.read_text() for p in Path(cwd).iterdir()}
        captured["cwd"] = cwd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("axolotl.cli.cloud.nebius.subprocess.run", run)
    return captured


def test_submit_isolated_context_and_forward_options(
    cloud_config, training_config, capture_launch
):
    cloud_config.update(
        {
            "profile": "training",
            "parent_id": "project-example",
            "subnet_id": "subnet-example",
            "disk_size": "300Gi",
            "timeout": 3600,
            "output": "storagebucket-example",
            "volumes": [
                {"source": "storagebucket-input", "mount": "/data", "mode": "ro"}
            ],
            "env": {"RUN_LABEL": "value with spaces; $(not-a-command)"},
            "env_secret": {"HF_TOKEN": "secret-id/HF_TOKEN"},
        }
    )
    NebiusCloud(cloud_config).train(
        yaml.safe_dump(training_config),
        learning_rate=0.0001,
        launcher_args=["--num_processes", "1"],
    )
    command = capture_launch["command"]
    assert command[:5] == ["/bin/nebius", "ai", "job", "run", "run.py"]
    assert "RUN_LABEL=value with spaces; $(not-a-command)" in command
    assert "HF_TOKEN=secret-id/HF_TOKEN" in command
    assert "storagebucket-input:/data:ro" in command
    assert command[command.index("--timeout") + 1] == "3600s"
    context = capture_launch["context"]
    assert set(context) == {
        "train.yaml",
        "launch.json",
        "run.py",
        "nebius_completion.py",
    }
    assert "secret-id" not in "\n".join(context.values())
    assert "RUN_LABEL" not in "\n".join(context.values())
    assert yaml.safe_load(context["train.yaml"])["learning_rate"] == 0.0001
    assert json.loads(context["launch.json"])["launcher_args"] == [
        "--num_processes",
        "1",
    ]
    assert not Path(capture_launch["cwd"]).exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("timeout", True),
        ("timeout", 10),
        ("timeout", 604801),
        ("timeout", "1h"),
        ("image", ""),
        ("preset", None),
        ("show_context", "yes"),
        ("gpu", "h100"),
        ("env", ["HF_TOKEN"]),
        ("env", {"NEBIUS_OUTPUT_DIR": "/tmp/lost"}),
        ("env_secret", {"HF_TOKEN": ""}),
        ("volumes", [{"source": "bucket", "mount": "/outputs"}]),
        ("volumes", [{"source": "bucket", "mount": "//outputs"}]),
        ("volumes", [{"source": "bucket", "mount": "/"}]),
        ("volumes", [{"source": "bucket", "mount": "relative"}]),
        ("volumes", [{"source": "bucket", "mount": "/data/../outputs"}]),
    ],
)
def test_invalid_cloud_config_fails_before_submit(cloud_config, key, value):
    cloud_config[key] = value
    with pytest.raises(ValueError):
        NebiusCloud(cloud_config)


def test_overlapping_mounts_rejected(cloud_config):
    cloud_config["volumes"] = [
        {"source": "bucket", "mount": "/data"},
        {"source": "other", "mount": "/data/models"},
    ]
    with pytest.raises(ValueError, match="overlap"):
        NebiusCloud(cloud_config)


def test_secret_and_plain_variable_cannot_overlap(cloud_config):
    cloud_config.update(env={"TOKEN": "plain"}, env_secret={"TOKEN": "secret"})
    with pytest.raises(ValueError, match="both"):
        NebiusCloud(cloud_config)


@pytest.mark.parametrize(
    "changes",
    [
        {"output_dir": "/workspace/ephemeral"},
        {"output_dir": "../escape"},
        {"auto_resume_from_checkpoints": True},
        {"resume_from_checkpoint": True},
        {"resume_from_checkpoint": "/outputs/old"},
        {"use_ray": True},
    ],
)
def test_training_validation_before_cli(
    cloud_config, training_config, changes, monkeypatch
):
    training_config.update(changes)
    called = Mock()
    monkeypatch.setattr("axolotl.cli.cloud.nebius.subprocess.run", called)
    with pytest.raises(ValueError):
        NebiusCloud(cloud_config).train(yaml.safe_dump(training_config))
    called.assert_not_called()


def test_missing_cli_is_actionable(cloud_config, training_config, monkeypatch):
    monkeypatch.setattr("axolotl.cli.cloud.nebius.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="Install the Nebius CLI"):
        NebiusCloud(cloud_config).train(yaml.safe_dump(training_config))


@pytest.mark.parametrize("returncode", [1, 2, 3, 5, 6])
def test_failure_is_not_retried_or_reported_as_success(
    cloud_config, training_config, monkeypatch, returncode
):
    cloud_config["env"] = {"PRIVATE_VALUE": "do-not-log-this"}
    monkeypatch.setattr(
        "axolotl.cli.cloud.nebius.shutil.which", lambda _: "/bin/nebius"
    )
    called = Mock(return_value=SimpleNamespace(returncode=returncode))
    monkeypatch.setattr("axolotl.cli.cloud.nebius.subprocess.run", called)
    with pytest.raises(RuntimeError) as exc:
        NebiusCloud(cloud_config).train(yaml.safe_dump(training_config))
    assert f"code {returncode}" in str(exc.value)
    assert "do-not-log-this" not in str(exc.value)
    called.assert_called_once()


def test_interrupt_keeps_remote_job_and_cleans_local_context(
    cloud_config, training_config, monkeypatch
):
    monkeypatch.setattr(
        "axolotl.cli.cloud.nebius.shutil.which", lambda _: "/bin/nebius"
    )
    directories = []

    def interrupt(command, cwd, check):
        directories.append(cwd)
        raise KeyboardInterrupt

    monkeypatch.setattr("axolotl.cli.cloud.nebius.subprocess.run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        NebiusCloud(cloud_config).train(yaml.safe_dump(training_config))
    assert len(directories) == 1
    assert not Path(directories[0]).exists()


def test_nebius_train_dispatch(cloud_config, training_config, tmp_path, capture_launch):
    cloud = tmp_path / "cloud.yml"
    config = tmp_path / "train.yml"
    cloud.write_text(yaml.safe_dump(cloud_config))
    config.write_text(yaml.safe_dump(training_config))
    do_cli_train(cloud, config, launcher="python", max_steps=5)
    assert json.loads(capture_launch["context"]["launch.json"])["launcher"] == "python"
    assert yaml.safe_load(capture_launch["context"]["train.yaml"])["max_steps"] == 5


@pytest.mark.parametrize("action", [do_cli_preprocess, do_cli_lm_eval])
def test_unsupported_nebius_command_does_not_launch_modal(
    cloud_config, tmp_path, action
):
    cloud = tmp_path / "cloud.yml"
    config = tmp_path / "train.yml"
    cloud.write_text(yaml.safe_dump(cloud_config))
    config.write_text("base_model: test/model")
    with pytest.raises(NotImplementedError):
        action(cloud, config)


@pytest.mark.parametrize("provider", [None, "modal", "baseten"])
def test_existing_provider_dispatch(monkeypatch, provider):
    modal_cls, baseten_cls = Mock(), Mock()
    monkeypatch.setitem(
        sys.modules, "axolotl.cli.cloud.modal_", SimpleNamespace(ModalCloud=modal_cls)
    )
    monkeypatch.setitem(
        sys.modules,
        "axolotl.cli.cloud.baseten",
        SimpleNamespace(BasetenCloud=baseten_cls),
    )
    cfg = DictDefault({"provider": provider})
    assert load_cloud_provider(cfg) is (
        baseten_cls.return_value if provider == "baseten" else modal_cls.return_value
    )


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        load_cloud_provider(DictDefault(provider="unknown"))


@pytest.fixture
def remote_context(tmp_path, training_config):
    root = tmp_path / "context"
    root.mkdir()
    (root / "train.yaml").write_text(yaml.safe_dump(training_config))
    (root / "launch.json").write_text(
        json.dumps({"launcher": "python", "launcher_args": [], "mounts": []})
    )
    output = tmp_path / "mounted"
    output.mkdir()
    return root, output


def test_output_resolved_to_managed_mount(remote_context):
    root, output = remote_context
    resolved, path = runner.prepare_config(root, output, [])
    assert path == output / "lora output"
    assert yaml.safe_load(resolved.read_text())["output_dir"] == str(path)
    with pytest.raises(ValueError, match="not empty"):
        (path / "old-checkpoint").touch()
        runner.prepare_config(root, output, [])


def test_partial_checkpoint_rejected_before_training(remote_context):
    root, output = remote_context
    checkpoint = root / "previous" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    cfg = yaml.safe_load((root / "train.yaml").read_text())
    cfg["resume_from_checkpoint"] = str(checkpoint)
    (root / "train.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="Trainer/optimizer"):
        runner.prepare_config(root, output, [str(checkpoint.parent)])


def test_complete_checkpoint_passed_explicitly(remote_context):
    root, output = remote_context
    checkpoint = root / "previous" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    for file in (
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "adapter_model.safetensors",
        "rng_state.pth",
    ):
        (checkpoint / file).write_text(
            '{"global_step": 10}' if file == "trainer_state.json" else "fixture"
        )
    cfg = yaml.safe_load((root / "train.yaml").read_text())
    cfg["resume_from_checkpoint"] = str(checkpoint)
    (root / "train.yaml").write_text(yaml.safe_dump(cfg))
    resolved, _ = runner.prepare_config(root, output, [str(checkpoint.parent)])
    assert yaml.safe_load(resolved.read_text())["resume_from_checkpoint"] == str(
        checkpoint
    )


@pytest.mark.parametrize("exit_code,complete", [(0, True), (7, True), (0, False)])
def test_remote_process_exit_and_completion_marker(
    remote_context, tmp_path, exit_code, complete
):
    root, output = remote_context
    (root / "run.py").write_text(Path(runner.__file__).read_text())
    binary = tmp_path / "bin"
    binary.mkdir()
    fake = binary / "axolotl"
    fake.write_text(f"""#!{sys.executable}
import os, json, sys
from pathlib import Path
if {complete!r}:
    Path(os.environ["AXOLOTL_NEBIUS_COMPLETION_FILE"]).write_text(json.dumps({{"global_step": 30, "max_steps": 30}}))
sys.exit({exit_code})
""")
    fake.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "NEBIUS_OUTPUT_DIR": str(output),
    }
    result = subprocess.run(
        [sys.executable, str(root / "run.py")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    success = exit_code == 0 and complete
    assert (result.returncode == 0) == success, result.stderr
    assert (output / "lora output" / "nebius-result.json").exists() == success
    if not complete and exit_code == 0:
        assert "without normal completion" in result.stderr


def test_real_nebius_context_preview(cloud_config, training_config, monkeypatch):
    """Opt-in local CLI packaging; --show-context never submits a Job."""
    if not os.environ.get("TEST_NEBIUS_CONTEXT"):
        pytest.skip(
            "Set TEST_NEBIUS_CONTEXT=1 to exercise the installed CLI without submission"
        )
    cloud_config["show_context"] = True
    NebiusCloud(cloud_config).train(yaml.safe_dump(training_config))


def test_null_output_uses_managed_bucket(cloud_config, training_config, capture_launch):
    cloud_config["output"] = None
    NebiusCloud(cloud_config).train(yaml.safe_dump(training_config))
    command = capture_launch["command"]
    assert command[command.index("--output") + 1] == "auto"


@pytest.mark.parametrize("is_primary", [True, False])
def test_completion_callback_writes_only_on_primary(monkeypatch, tmp_path, is_primary):
    import importlib.util

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(TrainerCallback=object)
    )
    monkeypatch.setitem(
        sys.modules, "axolotl.integrations.base", SimpleNamespace(BasePlugin=object)
    )
    spec = importlib.util.spec_from_file_location(
        "test_nebius_completion", Path(runner.__file__).with_name("completion.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "complete.json"
    monkeypatch.setenv("AXOLOTL_NEBIUS_COMPLETION_FILE", str(target))
    callback = module.NebiusCompletionPlugin().add_callbacks_pre_trainer(None, None)[0]
    callback.on_train_end(
        None,
        SimpleNamespace(is_world_process_zero=is_primary, global_step=30, max_steps=30),
        None,
    )
    assert target.exists() == is_primary
    if is_primary:
        assert json.loads(target.read_text()) == {"global_step": 30, "max_steps": 30}
