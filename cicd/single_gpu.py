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
dockerfile = os.environ.get("E2E_DOCKERFILE", "Dockerfile-uv.jinja")
df_template = template_env.get_template(dockerfile)

df_args = {
    "AXOLOTL_EXTRAS": os.environ.get("AXOLOTL_EXTRAS", ""),
    "AXOLOTL_ARGS": os.environ.get("AXOLOTL_ARGS", ""),
    "PYTORCH_VERSION": os.environ.get("PYTORCH_VERSION", "2.13.0"),
    "BASE_TAG": os.environ.get("BASE_TAG", "main-base-py3.12-cu130-2.13.0"),
    "DEPS_HASH": os.environ.get("DEPS_HASH", "dev"),
    "CUDA": os.environ.get("CUDA", "130"),
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


REFRESH_MARKER = ".ci_source_refreshed"


def _git(run_folder: str, *args: str, check: bool = True, capture: bool = False):
    import subprocess  # nosec

    return subprocess.run(  # nosec
        ["git", *args],
        cwd=run_folder,
        check=check,
        capture_output=capture,
        text=True,
    )


def checkout_source(run_folder: str, ref: str, sha: str) -> None:
    _git(run_folder, "fetch", "--depth=1", "origin", f"+{ref}")
    ref_head = _git(run_folder, "rev-parse", "FETCH_HEAD", capture=True).stdout.strip()
    target = sha
    if (
        _git(run_folder, "cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode
        != 0
    ):
        if (
            _git(
                run_folder, "fetch", "--depth=1", "origin", sha, check=False
            ).returncode
            != 0
        ):
            # refs/pull/N/merge is regenerated when the base moves; the old merge SHA is then unreachable
            if not ref.startswith("refs/pull/"):
                raise RuntimeError(f"{sha} not reachable from {ref}")
            print(f"WARNING: {sha} not fetchable, testing tip of {ref}", flush=True)
            target = ref_head
    _git(run_folder, "checkout", "-f", target)
    _git(run_folder, "log", "-1", "--oneline")


def refresh_source(run_folder: str) -> None:
    import subprocess  # nosec

    sha = os.environ.get("GITHUB_SHA", "")
    if not sha:
        return
    marker = pathlib.Path(run_folder) / REFRESH_MARKER
    if marker.exists() and marker.read_text().strip() == sha:
        return
    checkout_source(run_folder, os.environ.get("GITHUB_REF", "refs/heads/main"), sha)
    subprocess.run(  # nosec
        ["uv", "pip", "install", "--no-build-isolation", "--no-deps", "-e", "."],
        cwd=run_folder,
        check=True,
    )
    marker.write_text(sha)


def run_cmd(cmd: str, run_folder: str):
    import subprocess  # nosec

    refresh_source(run_folder)

    sp_env = os.environ.copy()
    sp_env["AXOLOTL_DATASET_NUM_PROC"] = "8"

    # Propagate errors from subprocess.
    exit_code = subprocess.call(cmd.split(), cwd=run_folder, env=sp_env)  # nosec
    if exit_code:
        raise RuntimeError(f"Command '{cmd}' failed with exit code {exit_code}")
