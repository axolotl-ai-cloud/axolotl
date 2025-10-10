"""Modal app to run axolotl GPU tests"""

import os
import pathlib
import tempfile

import jinja2
import modal
import modal.experimental
from jinja2 import select_autoescape
from modal import App

cicd_path = pathlib.Path(__file__).parent.resolve()

template_loader = jinja2.FileSystemLoader(searchpath=cicd_path)
template_env = jinja2.Environment(
    loader=template_loader, autoescape=select_autoescape()
)
dockerfile = os.environ.get("E2E_DOCKERFILE", "Dockerfile.jinja")
df_template = template_env.get_template(dockerfile)

df_args = {
    "AXOLOTL_EXTRAS": os.environ.get("AXOLOTL_EXTRAS", ""),
    "AXOLOTL_ARGS": os.environ.get("AXOLOTL_ARGS", ""),
    "PYTORCH_VERSION": os.environ.get("PYTORCH_VERSION", "2.6.0"),
    "BASE_TAG": os.environ.get("BASE_TAG", "main-base-py3.11-cu126-2.6.0"),
    "CUDA": os.environ.get("CUDA", "126"),
    "GITHUB_REF": os.environ.get("GITHUB_REF", "refs/heads/main"),
    "GITHUB_SHA": os.environ.get("GITHUB_SHA", ""),
    "NIGHTLY_BUILD": os.environ.get("NIGHTLY_BUILD", ""),
    "CODECOV_TOKEN": os.environ.get("CODECOV_TOKEN", ""),
    "HF_HOME": "/workspace/data/huggingface-cache/hub",
    "PYTHONUNBUFFERED": os.environ.get("PYTHONUNBUFFERED", "1"),
    "DEEPSPEED_LOG_LEVEL": os.environ.get("DEEPSPEED_LOG_LEVEL", "WARNING"),
}

dockerfile_contents = df_template.render(**df_args)

temp_dir = tempfile.mkdtemp()
with open(pathlib.Path(temp_dir) / "Dockerfile", "w", encoding="utf-8") as f:
    f.write(dockerfile_contents)

cicd_image = modal.experimental.raw_dockerfile_image(
    pathlib.Path(temp_dir) / "Dockerfile",
    # context_mount=None,
    force_build=True,
    # gpu="A10G",
).env(df_args)

app = App("Axolotl CI/CD", secrets=[])

hf_cache_volume = modal.Volume.from_name(
    "axolotl-ci-hf-hub-cache", create_if_missing=True
)
VOLUME_CONFIG = {
    "/workspace/data/huggingface-cache/hub": hf_cache_volume,
}

N_GPUS = int(os.environ.get("N_GPUS", 1))
GPU_TYPE = os.environ.get("GPU_TYPE", "L40S")
GPU_CONFIG = f"{GPU_TYPE}:{N_GPUS}"


def run_cmd(cmd: str, run_folder: str):
    import subprocess  # nosec

    sp_env = os.environ.copy()
    sp_env["AXOLOTL_DATASET_NUM_PROC"] = "8"

    # Propagate errors from subprocess.
    try:
        exit_code = subprocess.call(cmd.split(), cwd=run_folder, env=sp_env)  # nosec
        if exit_code:
            print(f"Command '{cmd}' failed with exit code {exit_code}")
            return exit_code
    except Exception as e:  # pylint: disable=broad-except
        print(f"Command '{cmd}' failed with exception {e}")
