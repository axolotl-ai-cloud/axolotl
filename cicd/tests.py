"""Modal app to run axolotl GPU tests"""

# pylint: disable=duplicate-code

import os
import pathlib
import tempfile

import jinja2
import modal
from jinja2 import select_autoescape
from modal import App, Image

cicd_path = pathlib.Path(__file__).parent.resolve()

template_loader = jinja2.FileSystemLoader(searchpath=cicd_path)
template_env = jinja2.Environment(
    loader=template_loader, autoescape=select_autoescape()
)
df_template = template_env.get_template("Dockerfile.jinja")

df_args = {
    "AXOLOTL_EXTRAS": os.environ.get("AXOLOTL_EXTRAS", ""),
    "AXOLOTL_ARGS": os.environ.get("AXOLOTL_ARGS", ""),
    "PYTORCH_VERSION": os.environ.get("PYTORCH_VERSION", "2.4.1"),
    "BASE_TAG": os.environ.get("BASE_TAG", "main-base-py3.11-cu121-2.4.1"),
    "CUDA": os.environ.get("CUDA", "121"),
    "GITHUB_REF": os.environ.get("GITHUB_REF", "refs/heads/main"),
    "GITHUB_SHA": os.environ.get("GITHUB_SHA", ""),
    "NIGHTLY_BUILD": os.environ.get("NIGHTLY_BUILD", ""),
    "HF_HOME": "/workspace/data/huggingface-cache/hub",
}

dockerfile_contents = df_template.render(**df_args)

temp_dir = tempfile.mkdtemp()
with open(pathlib.Path(temp_dir) / "Dockerfile", "w", encoding="utf-8") as f:
    f.write(dockerfile_contents)

cicd_image = Image.from_dockerfile(
    pathlib.Path(temp_dir) / "Dockerfile",
    context_mount=None,
    force_build=True,
    gpu="A10G",
).env(df_args)

app = App("Axolotl CI/CD", secrets=[])

hf_cache_volume = modal.Volume.from_name(
    "axolotl-ci-hf-hub-cache", create_if_missing=True
)
VOLUME_CONFIG = {
    "/workspace/data/huggingface-cache/hub": hf_cache_volume,
}

N_GPUS = int(os.environ.get("N_GPUS", 1))
GPU_CONFIG = modal.gpu.L40S(count=N_GPUS)


def run_cmd(cmd: str, run_folder: str):
    import subprocess  # nosec

    # Propagate errors from subprocess.
    if exit_code := subprocess.call(cmd.split(), cwd=run_folder):  # nosec
        exit(exit_code)  # pylint: disable=consider-using-sys-exit


@app.function(
    image=cicd_image,
    gpu=GPU_CONFIG,
    timeout=60 * 60,
    cpu=8.0,
    memory=131072,
    volumes=VOLUME_CONFIG,
)
def cicd_pytest():
    run_cmd("./cicd/cicd.sh", "/workspace/axolotl")


@app.local_entrypoint()
def main():
    cicd_pytest.remote()
