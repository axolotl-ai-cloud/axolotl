"""Prebuilt images and local image builds for cloud launchers."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from pydantic import ValidationError

from axolotl.cli.cloud import images, load_cloud_cfg
from axolotl.utils.schemas.cloud import CloudImageBuild, CloudImageConfig


@pytest.fixture
def modal_sdk():
    return pytest.importorskip("modal")


@pytest.fixture
def build_context(tmp_path):
    context = tmp_path / "local fork"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM example/base:latest\nCOPY . /src\n")
    (context / "plugin.py").write_text("CUSTOM_PLUGIN = True\n")
    return context


@pytest.mark.parametrize(
    "image", ["registry.example/axolotl:fork", "repo/image@sha256:abc"]
)
def test_prebuilt_image(image):
    config = CloudImageConfig.model_validate({"image": image})
    assert config.image == image
    assert config.image_build is None


@pytest.mark.parametrize("image", ["", "has spaces", "-option", True])
def test_invalid_image(image):
    with pytest.raises(ValidationError):
        CloudImageConfig.model_validate({"image": image})


def test_mutually_exclusive_sources(build_context):
    with pytest.raises(ValidationError, match="either image or image_build"):
        CloudImageConfig.model_validate(
            {"image": "repo/image:tag", "image_build": {"context": build_context}}
        )


def test_build_requires_local_dockerfile(tmp_path):
    with pytest.raises(ValidationError, match="Dockerfile does not exist"):
        CloudImageBuild(context=tmp_path)


def test_dockerfile_must_be_inside_context(build_context):
    with pytest.raises(ValidationError, match="inside the build context"):
        CloudImageBuild(context=build_context, dockerfile="../Dockerfile")


def test_unknown_build_option_rejected(build_context):
    with pytest.raises(ValidationError, match="Extra inputs"):
        CloudImageBuild.model_validate({"context": build_context, "typo": True})


def test_build_context_is_relative_to_cloud_yaml(build_context, tmp_path, monkeypatch):
    config_path = tmp_path / "cloud.yaml"
    config_path.write_text("image_build:\n  context: local fork\n")
    monkeypatch.chdir(build_context)
    config = CloudImageConfig.from_config(
        load_cloud_cfg(config_path).to_dict(), config_dir=config_path.parent
    )
    assert config.image_build.context == build_context
    assert config.image_build.dockerfile == build_context / "Dockerfile"


def test_build_and_push_uses_explicit_context(build_context, monkeypatch):
    run = MagicMock()
    monkeypatch.setattr(images.subprocess, "run", run)
    build = CloudImageBuild(
        context=build_context,
        tag="registry.example/fork:test",
        build_args={"AXOLOTL_EXTRAS": "deepspeed, custom"},
    )
    assert images.build_and_push_image(build) == build.tag
    assert run.call_args_list[0].args[0] == [
        "docker",
        "build",
        "--platform",
        "linux/amd64",
        "--file",
        str(build_context / "Dockerfile"),
        "--tag",
        build.tag,
        "--build-arg",
        "AXOLOTL_EXTRAS=deepspeed, custom",
        str(build_context),
    ]
    assert run.call_args_list[1].args[0] == ["docker", "push", build.tag]
    assert all(call.kwargs == {"check": True} for call in run.call_args_list)


@pytest.mark.parametrize("fail_at", ["build", "push"])
def test_image_failure_prevents_baseten_submission(build_context, monkeypatch, fail_at):
    from axolotl.integrations.baseten.cloud import BasetenCloud

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[1] == fail_at:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(images.subprocess, "run", run)
    provider = BasetenCloud(
        {"image_build": {"context": build_context, "tag": "registry.example/fork:test"}}
    )
    with pytest.raises(subprocess.CalledProcessError):
        provider.train("base_model: example")
    assert all(command[0] == "docker" for command in calls)
    assert len(calls) == (1 if fail_at == "build" else 2)


def test_baseten_requires_build_tag(build_context):
    from axolotl.integrations.baseten.cloud import BasetenCloud

    with pytest.raises(ValueError, match="requires a registry tag"):
        BasetenCloud({"image_build": {"context": build_context}})


@pytest.mark.parametrize("build_local", [False, True])
def test_baseten_uses_selected_image(build_context, monkeypatch, build_local):
    from axolotl.integrations.baseten import cloud as baseten_cloud

    image = "registry.example/fork:test"
    config = (
        {"image_build": {"context": build_context, "tag": image}}
        if build_local
        else {"image": image}
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "truss":
            root = Path(kwargs["cwd"])
            resolved = yaml.safe_load((root / "cloud.yaml").read_text())
            assert resolved["image"] == image
            assert "image_build" not in resolved
            assert kwargs["check"] is True

    monkeypatch.setattr(baseten_cloud.subprocess, "run", run)
    baseten_cloud.BasetenCloud(config).train("plugins: [custom.Plugin]\n")
    assert [command[:2] for command in calls] == (
        [["docker", "build"], ["docker", "push"], ["truss", "train"]]
        if build_local
        else [["truss", "train"]]
    )


@pytest.mark.usefixtures("modal_sdk")
def test_modal_builds_context_without_local_docker(build_context, monkeypatch):
    from axolotl.integrations.modal import cloud as modal_cloud

    builder = MagicMock()
    monkeypatch.setattr(modal_cloud.modal.Image, "from_dockerfile", builder)
    monkeypatch.setattr(
        modal_cloud.subprocess,
        "check_output",
        lambda *a, **k: pytest.fail("Local Docker must not be called"),
    )
    provider = modal_cloud.ModalCloud(
        {"image_build": {"context": build_context, "build_args": {"EXTRA": "custom"}}},
        app=MagicMock(),
    )
    provider.get_image()
    builder.assert_called_once_with(
        build_context / "Dockerfile",
        context_dir=build_context,
        build_args={"EXTRA": "custom"},
    )
    assert builder.return_value.env.called


@pytest.mark.usefixtures("modal_sdk")
def test_modal_prebuilt_image_without_local_docker(monkeypatch):
    from axolotl.integrations.modal import cloud as modal_cloud

    registry = MagicMock()
    monkeypatch.setattr(modal_cloud.modal.Image, "from_registry", registry)
    monkeypatch.setattr(
        modal_cloud.subprocess,
        "check_output",
        lambda *a, **k: pytest.fail("Local Docker must not be called"),
    )
    modal_cloud.ModalCloud(
        {"image": "registry.example/fork:test"}, app=MagicMock()
    ).get_image()
    registry.assert_called_once_with("registry.example/fork:test")


@pytest.mark.parametrize("key", ["branch", "docker_tag", "dockerfile_commands"])
@pytest.mark.usefixtures("modal_sdk")
def test_modal_build_rejects_legacy_customization(build_context, key):
    from axolotl.integrations.modal.cloud import ModalCloud

    with pytest.raises(ValueError, match="cannot be combined"):
        ModalCloud(
            {"image_build": {"context": build_context}, key: "custom"}, app=MagicMock()
        )


@pytest.mark.parametrize(
    "reference",
    [
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/team/axolotl:fork",
        "us-central1-docker.pkg.dev/project/repository/axolotl:fork",
        "example.azurecr.io/axolotl:fork",
        "ghcr.io/team/axolotl:fork",
    ],
)
def test_registry_build_preserves_full_reference(build_context, monkeypatch, reference):
    run = MagicMock()
    monkeypatch.setattr(images.subprocess, "run", run)
    assert (
        images.build_and_push_image(
            CloudImageBuild(context=build_context, tag=reference)
        )
        == reference
    )
    assert run.call_args.args[0] == ["docker", "push", reference]


@pytest.mark.parametrize(
    "provider,method",
    [
        ("registry", "from_registry"),
        ("aws_ecr", "from_aws_ecr"),
        ("gcp_artifact_registry", "from_gcp_artifact_registry"),
    ],
)
@pytest.mark.usefixtures("modal_sdk")
def test_modal_registry_pull_uses_named_secret(monkeypatch, provider, method):
    from axolotl.integrations.modal import cloud as modal_cloud

    loader, secret = MagicMock(), MagicMock()
    monkeypatch.setattr(modal_cloud.modal.Image, method, loader)
    monkeypatch.setattr(modal_cloud.modal.Secret, "from_name", secret)
    image = "registry.example/team/axolotl:fork"
    modal_cloud.ModalCloud(
        {
            "image": image,
            "image_registry": {"provider": provider, "secret": "pull-credentials"},
        },
        app=MagicMock(),
    ).get_image()
    secret.assert_called_once_with("pull-credentials")
    loader.assert_called_once_with(image, secret=secret.return_value)


@pytest.mark.usefixtures("modal_sdk")
def test_modal_tagged_build_uses_registry_path(build_context, monkeypatch):
    from axolotl.integrations.modal import cloud as modal_cloud

    tag = "123456789012.dkr.ecr.us-east-1.amazonaws.com/axolotl:fork"
    builder = MagicMock(return_value=tag)
    loader = MagicMock()
    monkeypatch.setattr(modal_cloud, "build_and_push_image", builder)
    monkeypatch.setattr(modal_cloud.modal.Image, "from_aws_ecr", loader)
    monkeypatch.setattr(modal_cloud.modal.Secret, "from_name", MagicMock())
    modal_cloud.ModalCloud(
        {
            "image_build": {"context": build_context, "tag": tag},
            "image_registry": {"provider": "aws_ecr", "secret": "ecr-pull"},
        },
        app=MagicMock(),
    ).get_image()
    assert builder.call_args.args[0].tag == tag
    assert loader.call_args.args[0] == tag


@pytest.mark.usefixtures("modal_sdk")
def test_modal_native_build_rejects_unused_pull_credentials(build_context):
    from axolotl.integrations.modal.args import ModalImageConfig

    with pytest.raises(ValidationError, match="image_registry requires"):
        ModalImageConfig.model_validate(
            {
                "image_build": {"context": build_context},
                "image_registry": {"secret": "ecr-pull"},
            }
        )


def test_baseten_passes_native_registry_auth_to_image(tmp_path, monkeypatch):
    import runpy
    import sys
    from types import ModuleType

    from axolotl.integrations.baseten import cloud as baseten_cloud

    auth = {
        "auth_method": "AWS_OIDC",
        "registry": "123456789012.dkr.ecr.us-east-1.amazonaws.com",
        "aws_oidc_docker_auth": {
            "role_arn": "arn:aws:iam::123456789012:role/pull",
            "region": "us-east-1",
        },
    }
    image = auth["registry"] + "/axolotl:fork"
    definitions = MagicMock()
    truss_base = ModuleType("truss.base")
    truss_base.truss_config = MagicMock()
    truss_train = ModuleType("truss_train")
    truss_train.definitions = definitions
    monkeypatch.setitem(sys.modules, "truss", ModuleType("truss"))
    monkeypatch.setitem(sys.modules, "truss.base", truss_base)
    monkeypatch.setitem(sys.modules, "truss_train", truss_train)

    def submit(command, cwd, check):
        monkeypatch.chdir(cwd)
        runpy.run_path(str(Path(cwd) / "train_sft.py"))

    monkeypatch.setattr(baseten_cloud.subprocess, "run", submit)
    baseten_cloud.BasetenCloud({"image": image, "docker_auth": auth}).train(
        "base_model: example"
    )
    definitions.DockerAuth.model_validate.assert_called_once_with(auth, strict=False)
    definitions.Image.assert_called_once_with(
        base_image=image, docker_auth=definitions.DockerAuth.model_validate.return_value
    )
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("provider_name", ["modal", "baseten"])
def test_provider_resolves_its_own_image_paths(
    build_context, tmp_path, monkeypatch, provider_name
):
    from axolotl.cli.cloud.registry import load_cloud_provider

    if provider_name == "modal":
        pytest.importorskip("modal")
    config = {
        "provider": provider_name,
        "image_build": {"context": build_context.name, "tag": "example/image:fork"},
    }
    monkeypatch.chdir(build_context)
    provider = load_cloud_provider(config, config_dir=tmp_path)
    assert provider.image_config.image_build.context == build_context
    assert config["image_build"]["context"] == build_context.name
