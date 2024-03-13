"""
 modal application to run axolotl gpu tests in Modal
 """
import os
import pathlib
import tempfile

import jinja2
import modal
from jinja2 import select_autoescape
from modal import Image, Stub

cicd_path = pathlib.Path(__file__).parent.resolve()

template_loader = jinja2.FileSystemLoader(searchpath=cicd_path)
template_env = jinja2.Environment(
    loader=template_loader, autoescape=select_autoescape()
)
df_template = template_env.get_template("Dockerfile.jinja")

df_args = {
    "AXOLOTL_EXTRAS": os.environ.get("AXOLOTL_EXTRAS", ""),
    "AXOLOTL_ARGS": os.environ.get("AXOLOTL_ARGS", ""),
    "PYTORCH_VERSION": os.environ.get("PYTORCH_VERSION", "2.0.1"),
    "BASE_TAG": os.environ.get("BASE_TAG", "main-base-py3.10-cu118-2.0.1"),
    "CUDA": os.environ.get("CUDA", "118"),
    "GITHUB_REF": os.environ.get("GITHUB_REF", "refs/heads/main"),
    "GITHUB_SHA": os.environ.get("GITHUB_SHA", ""),
}

dockerfile_contents = df_template.render(**df_args)

temp_dir = tempfile.mkdtemp()
with open(pathlib.Path(temp_dir) / "Dockerfile", "w", encoding="utf-8") as f:
    f.write(dockerfile_contents)

cicd_image = (
    Image.from_dockerfile(
        pathlib.Path(temp_dir) / "Dockerfile",
        force_build=True,
        gpu="A10G",
    )
    .env(df_args)
    .pip_install("fastapi==0.110.0", "pydantic==2.6.3")
)

stub = Stub("Axolotl CI/CD", secrets=[])


N_GPUS = int(os.environ.get("N_GPUS", 1))
GPU_CONFIG = modal.gpu.A10G(count=N_GPUS)


def run_cmd(cmd: str, run_folder: str):
    import subprocess  # nosec

    # Propagate errors from subprocess.
    if exit_code := subprocess.call(cmd.split(), cwd=run_folder):  # nosec
        exit(exit_code)  # pylint: disable=consider-using-sys-exit


@stub.function(
    image=cicd_image,
    gpu=GPU_CONFIG,
    timeout=45 * 60,
    cpu=8.0,
    memory=131072,
)
def cicd_pytest():
    run_cmd("./cicd/cicd.sh", "/workspace/axolotl")


@stub.local_entrypoint()
def main():
    cicd_pytest.remote()
