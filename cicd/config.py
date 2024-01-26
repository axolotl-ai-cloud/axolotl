import os
import pathlib

import modal
from modal import Image, Stub

cicd_path = pathlib.Path(__file__).parent.resolve()

DOCKER_ENV = {
    "PYTORCH_VERSION": os.environ.get("PYTORCH_VERSION"),
    "BASE_TAG": os.environ.get("BASE_TAG"),
    "CUDA": os.environ.get("CUDA"),
    "GITHUB_REF": os.environ.get("GITHUB_REF", "refs/head/main"),
}

cicd_image = Image.from_dockerfile(
    cicd_path / "../docker/Dockerfile-modal", gpu="A10G"
).env(DOCKER_ENV)

stub = Stub("Axolotl CI/CD", secrets=[])


N_GPUS = int(os.environ.get("N_GPUS", 2))
GPU_CONFIG = modal.gpu.A10G(count=N_GPUS)


def run_cmd(cmd: str, run_folder: str):
    import subprocess

    # Propagate errors from subprocess.
    if exit_code := subprocess.call(cmd.split(), cwd=run_folder):
        exit(exit_code)


@stub.function(
    image=cicd_image,
    gpu=GPU_CONFIG,
    timeout=60 * 30,
)
def cicd_pytest():
    CMD = "pytest /workspace/axolotl/tests/e2e/patched/"
    run_cmd(CMD, "/workspace/axolotl")


@stub.local_entrypoint()
def main():
    cicd_pytest.remote()
