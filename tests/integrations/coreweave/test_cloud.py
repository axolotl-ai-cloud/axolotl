"""SUNK submission, serialization, and process topology."""

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from axolotl.cli.cloud.registry import load_cloud_provider
from axolotl.integrations.coreweave.cloud import CoreWeaveCloud, container_image
from axolotl.integrations.coreweave.cloud_runner import launch_command


def provider(tmp_path, **kwargs):
    return CoreWeaveCloud(
        {
            "provider": "coreweave",
            "image": "/mnt/shared/axolotl.sqsh",
            "partition": "h100",
            "shared_dir": str(tmp_path),
            **kwargs,
        }
    )


def test_local_submission_keeps_plugins_overrides_and_outputs(tmp_path, monkeypatch):
    submit = MagicMock(
        return_value=subprocess.CompletedProcess([], 0, stdout="123;cluster\n")
    )
    monkeypatch.setattr(subprocess, "run", submit)
    result = provider(tmp_path).train(
        "base_model: test\nplugins: [axolotl.integrations.coreweave.CoreWeavePlugin]",
        learning_rate=0.001,
    )
    directory = Path(result["directory"])
    config = yaml.safe_load((directory / "train.yaml").read_text())
    assert config["learning_rate"] == 0.001
    assert config["plugins"] == ["axolotl.integrations.coreweave.CoreWeavePlugin"]
    assert config["output_dir"] == str(directory / "outputs")
    assert result["job_id"] == "123"
    command = submit.call_args.args[0]
    assert command[0:2] == ["sbatch", "--parsable"]
    assert "--gpus-per-node=8" in command
    assert directory.stat().st_mode & 0o777 == 0o700


def test_remote_staging_uses_stdin_archive(tmp_path, monkeypatch):
    submit = MagicMock(return_value=subprocess.CompletedProcess([], 0, stdout="456\n"))
    monkeypatch.setattr(subprocess, "run", submit)
    result = provider(tmp_path, ssh_host="user@sunk").train(
        "base_model: test", launcher="torchrun"
    )
    assert result["job_id"] == "456"
    assert len(submit.call_args_list) == 4
    tar_call = submit.call_args_list[2]
    assert tar_call.args[0][:4] == ["ssh", "-o", "BatchMode=yes", "user@sunk"]
    with tarfile.open(fileobj=io.BytesIO(tar_call.kwargs["input"])) as archive:
        assert set(archive.getnames()) == {
            "train.yaml",
            "launch.json",
            "runner.py",
            "job.sh",
        }
        assert json.load(archive.extractfile("launch.json"))["launcher"] == "torchrun"


def test_dry_run_has_no_side_effects_and_script_is_valid(tmp_path, monkeypatch):
    run = subprocess.run
    submit = MagicMock(side_effect=AssertionError("No remote calls"))
    monkeypatch.setattr(subprocess, "run", submit)
    result = provider(
        tmp_path, dry_run=True, env={"TEST_VALUE": "quote'; $(false)"}
    ).train("base_model: test")
    assert list(tmp_path.iterdir()) == []
    script = result["files"]["job.sh"].decode()
    run(["bash", "-n"], input=script, text=True, check=True)
    assert "--kill-on-bad-exit=1" in script


@pytest.mark.parametrize("launcher", ["accelerate", "torchrun"])
def test_multi_node_topology(launcher):
    command = launch_command(
        {"launcher": launcher, "launcher_args": [], "gpus_per_node": 8},
        {
            "SLURM_NNODES": "2",
            "SLURM_PROCID": "1",
            "MASTER_ADDR": "node0",
            "MASTER_PORT": "29500",
        },
    )
    assert command[:5] == ["axolotl", "train", "train.yaml", "--launcher", launcher]
    if launcher == "accelerate":
        assert command[command.index("--num_processes") + 1] == "16"
        assert command[command.index("--machine_rank") + 1] == "1"
    else:
        assert command[command.index("--nproc_per_node") + 1] == "8"
        assert command[command.index("--node_rank") + 1] == "1"


@pytest.mark.parametrize(
    "args", [["--nnodes=2"], ["--num-processes", "3"], ["--standalone"]]
)
def test_topology_override_rejected(tmp_path, args):
    with pytest.raises(ValueError, match="topology"):
        provider(tmp_path).train("base_model: test", launcher_args=args)
    assert list(tmp_path.iterdir()) == []


def test_submission_failure_never_retries(tmp_path, monkeypatch):
    submit = MagicMock(side_effect=subprocess.CalledProcessError(1, ["sbatch"]))
    monkeypatch.setattr(subprocess, "run", submit)
    with pytest.raises(RuntimeError, match="before resubmitting"):
        provider(tmp_path).train("base_model: test")
    submit.assert_called_once()
    assert list(tmp_path.glob("*/train.yaml"))


def test_registry_loads_without_training_dependencies(tmp_path):
    script = """
import sys
from axolotl.cli.cloud.registry import load_cloud_provider
p = load_cloud_provider({'provider':'coreweave', 'image':'example/image:tag', 'shared_dir':'/mnt/shared', 'partition':'h100'})
assert type(p).__name__ == 'CoreWeaveCloud'
assert not {'torch', 'transformers', 'peft', 'boto3'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_config_discovery(tmp_path):
    assert isinstance(
        load_cloud_provider(
            {
                "provider": "coreweave",
                "image": "test/image:tag",
                "shared_dir": str(tmp_path),
                "partition": "h100",
            }
        ),
        CoreWeaveCloud,
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"output_dir": "/tmp/ephemeral"}, {"use_ray": True}, {"output_dir": "../outside"}],
)
def test_invalid_training_paths_and_backend(tmp_path, kwargs):
    with pytest.raises(ValueError):
        provider(tmp_path).train("base_model: test", **kwargs)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "image, expected",
    [
        ("ghcr.io/org/image:tag", "docker://ghcr.io#org/image:tag"),
        ("registry.example:5000/image:tag", "docker://registry.example:5000#image:tag"),
        ("/shared/model.sqsh", "/shared/model.sqsh"),
        ("axolotlai/axolotl:tag", "axolotlai/axolotl:tag"),
    ],
)
def test_enroot_registry_references(image, expected):
    assert container_image(image) == expected


def test_container_environment_overrides_image_defaults(tmp_path):
    import os
    import shlex

    env_file = tmp_path / "training.env"
    env_file.write_text("HF_HOME=/from-file\nNCCL_SOCKET_IFNAME=eth0\n")
    result = provider(
        tmp_path, dry_run=True, env_file=str(env_file), env={"HF_HOME": "/explicit"}
    ).train("base_model: test")
    script = result["files"]["job.sh"].decode()
    command = shlex.split(script.split("exec ", 1)[1])
    assert "--container-env=HF_HOME,MASTER_ADDR,MASTER_PORT" in command
    assert f"{env_file}:{env_file}:ro" in next(
        arg for arg in command if arg.startswith("--container-mounts=")
    )
    bootstrap = command[-1].rsplit("exec python3", 1)[0] + "env"
    result = subprocess.run(
        ["bash", "-c", bootstrap],
        check=True,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "HF_HOME": "/image-default",
            "NCCL_SOCKET_IFNAME": "wrong",
            "MASTER_ADDR": "node0",
            "MASTER_PORT": "29500",
        },
    )
    values = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    assert values["HF_HOME"] == "/explicit"
    assert values["NCCL_SOCKET_IFNAME"] == "eth0"
    assert values["MASTER_ADDR"] == "node0"
