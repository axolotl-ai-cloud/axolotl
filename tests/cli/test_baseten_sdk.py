"""Native Baseten SDK contract tests; install the baseten extra to run."""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("truss_train")

TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "src/axolotl/integrations/baseten/template/train_sft.py"
)


def load_project(config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cloud.yaml").write_text(yaml.safe_dump(config))
    return runpy.run_path(str(TEMPLATE))["first_project_with_job"]


@pytest.mark.parametrize("gpu", ["h100", "h200", "a100"])
def test_job_definition_honors_config(tmp_path, monkeypatch, gpu):
    project = load_project(
        {
            "image": "ghcr.io/team/axolotl:fork",
            "gpu": gpu,
            "gpu_count": 4,
            "node_count": 2,
            "project_name": "custom-training",
            "secrets": ["HF_TOKEN"],
            "launcher": "torchrun",
            "launcher_args": ["--nproc_per_node", "4"],
        },
        tmp_path,
        monkeypatch,
    )
    assert project.name == "custom-training"
    assert project.job.image.base_image == "ghcr.io/team/axolotl:fork"
    assert project.job.image.docker_auth is None
    assert project.job.compute.accelerator.accelerator.value == gpu.upper()
    assert project.job.compute.accelerator.count == 4
    assert project.job.compute.node_count == 2
    env = project.job.runtime.environment_variables
    assert env["HF_TOKEN"].name == "HF_TOKEN"
    assert env["AXOLOTL_LAUNCHER"] == "torchrun"
    assert env["AXOLOTL_LAUNCHER_ARGS"] == "-- --nproc_per_node 4"


@pytest.mark.parametrize(
    "registry,method,field,settings",
    [
        (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com",
            "AWS_IAM",
            "aws_iam_docker_auth",
            {
                "access_key_secret_ref": {"name": "ecr-access-key"},
                "secret_access_key_secret_ref": {"name": "ecr-secret-key"},
            },
        ),
        (
            "us-central1-docker.pkg.dev",
            "GCP_SERVICE_ACCOUNT_JSON",
            "gcp_service_account_json_docker_auth",
            {"service_account_json_secret_ref": {"name": "gcp-pull"}},
        ),
        (
            "ghcr.io",
            "REGISTRY_SECRET",
            "registry_secret_docker_auth",
            {"secret_ref": {"name": "ghcr-pull"}},
        ),
        (
            "example.azurecr.io",
            "REGISTRY_SECRET",
            "registry_secret_docker_auth",
            {"secret_ref": {"name": "acr-pull"}},
        ),
    ],
)
def test_native_registry_auth(tmp_path, monkeypatch, registry, method, field, settings):
    auth = {"auth_method": method, "registry": registry, field: settings}
    project = load_project(
        {
            "image": registry + "/team/axolotl:fork",
            "docker_auth": auth,
        },
        tmp_path,
        monkeypatch,
    )
    assert (
        project.job.image.docker_auth.model_dump(mode="json", exclude_none=True) == auth
    )


def test_training_push_cli_contract():
    executable = Path(sys.executable).with_name("truss")
    result = subprocess.run(
        [str(executable), "train", "push", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "CONFIG" in result.stdout
    assert "--remote" in result.stdout
