"""Image preparation shared by registry-based cloud launchers."""

import subprocess  # nosec B404

from axolotl.utils.schemas.cloud import CloudImageBuild


def build_and_push_image(build: CloudImageBuild) -> str:
    """Build the configured local context and publish it for a remote image pull."""
    if not build.tag:
        raise ValueError(
            "image_build.tag is required to build and push a registry image"
        )
    command = [
        "docker",
        "build",
        "--platform",
        build.platform,
        "--file",
        str(build.dockerfile),
        "--tag",
        build.tag,
    ]
    for name, value in build.build_args.items():
        command.extend(["--build-arg", f"{name}={value}"])
    command.append(str(build.context))
    subprocess.run(command, check=True)  # nosec B603 B607
    subprocess.run(["docker", "push", build.tag], check=True)  # nosec B603 B607
    return build.tag
