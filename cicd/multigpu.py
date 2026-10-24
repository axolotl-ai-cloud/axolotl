"""
modal application to run axolotl gpu tests in Modal
"""

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
    "IMAGE_REF": os.environ.get("IMAGE_REF")
    or os.environ.get("GITHUB_REF", "refs/heads/main"),
    "GITHUB_SHA": os.environ.get("GITHUB_SHA", ""),
    "NIGHTLY_BUILD": os.environ.get("NIGHTLY_BUILD", ""),
    "MULTIGPU_TEST_SUITE": os.environ.get("MULTIGPU_TEST_SUITE", "multigpu"),
    "NF4_TEST_STACK": os.environ.get("NF4_TEST_STACK", "pinned"),
    "CODECOV_TOKEN": os.environ.get("CODECOV_TOKEN", ""),
    "HF_HOME": "/workspace/data/huggingface-cache/hub",
    "PYTHONUNBUFFERED": os.environ.get("PYTHONUNBUFFERED", "1"),
    "DEEPSPEED_LOG_LEVEL": os.environ.get("DEEPSPEED_LOG_LEVEL", "WARNING"),
    "E2E_SELECTED_TESTS": os.environ.get("E2E_SELECTED_TESTS", ""),
}

dockerfile_contents = df_template.render(**df_args)

temp_dir = tempfile.mkdtemp()
with open(pathlib.Path(temp_dir) / "Dockerfile", "w", encoding="utf-8") as f:
    f.write(dockerfile_contents)

cicd_image = (
    Image.from_dockerfile(
        pathlib.Path(temp_dir) / "Dockerfile",
        gpu="A10G",
    )
    # cwd leads sys.path; from the image's checkout it shadows the cicd/ Modal mounts at /root
    .workdir("/root")
    .env(df_args)
)

app = App("Axolotl CI/CD", secrets=[])

hf_cache_volume = modal.Volume.from_name(
    "axolotl-ci-hf-hub-cache", create_if_missing=True
)
DEEP_EP_WHEELS = "/workspace/data/deep-ep-wheels"
deep_ep_volume = modal.Volume.from_name(
    "axolotl-ci-deep-ep-wheels", create_if_missing=True
)
VOLUME_CONFIG = {
    "/workspace/data/huggingface-cache/hub": hf_cache_volume,
    DEEP_EP_WHEELS: deep_ep_volume,
}

# Same unpatched recipe as docker/Dockerfile-cloud-blackwell-uv; only these suites run deep_ep tests.
DEEP_EP_REF = "v1.2.1"
DEEP_EP_ARCHES = {"9.0", "10.0"}
DEEP_EP_SUITES = {"multigpu", "multigpu_selected"}
DEEP_EP_ABI = """
import importlib.metadata as md, sys, torch
nvshmem = md.version(f"nvidia-nvshmem-cu{torch.version.cuda.split('.')[0]}")
print(f"torch{torch.__version__}-cp{sys.version_info[0]}{sys.version_info[1]}-nvshmem{nvshmem}")
"""

N_GPUS = int(os.environ.get("N_GPUS", 2))
GPU_TYPE = os.environ.get("GPU_TYPE", "H100")
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


def install_deep_ep(wheel_cache: str) -> bool:
    """Install DeepEP from a wheel cached per torch/python/nvshmem ABI, building it on a miss.

    Returns whether a wheel was built. The image itself is rebuilt with every DEPS_HASH change,
    so the build lives outside it.
    """
    import shutil
    import subprocess  # nosec
    import sys
    import tempfile

    has_deep_ep = (
        "import importlib.util as u, sys; sys.exit(not u.find_spec('deep_ep'))"
    )
    if subprocess.run([sys.executable, "-c", has_deep_ep]).returncode == 0:  # nosec
        print("DeepEP: shipped by the image, skipping", flush=True)
        return False
    arch = subprocess.run(  # nosec
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()[0]
    if arch not in DEEP_EP_ARCHES:
        print(f"DeepEP: no unpatched build for sm_{arch}, skipping", flush=True)
        return False
    abi = subprocess.run(  # nosec
        [sys.executable, "-c", DEEP_EP_ABI], check=True, capture_output=True, text=True
    ).stdout.strip()
    wheel_dir = pathlib.Path(wheel_cache) / f"{DEEP_EP_REF}-sm{arch}-{abi}"
    wheels = sorted(wheel_dir.glob("deep_ep-*.whl"))
    built = not wheels
    if built:
        print(f"DeepEP: building {wheel_dir.name}", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            src = f"{tmp}/DeepEP"
            subprocess.run(  # nosec
                [
                    "git",
                    "clone",
                    "--depth=1",
                    f"--branch={DEEP_EP_REF}",
                    "https://github.com/deepseek-ai/DeepEP.git",
                    src,
                ],
                check=True,
            )
            # CUDA 13.0 moved cuda/std under include/cccl
            cccl = "-I/usr/local/cuda/include/cccl"
            env = {
                **os.environ,
                "TORCH_CUDA_ARCH_LIST": arch,
                "MAX_JOBS": "16",
                "CFLAGS": cccl,
                "CXXFLAGS": cccl,
                "NVCC_PREPEND_FLAGS": cccl,
            }
            subprocess.run(  # nosec
                [sys.executable, "setup.py", "bdist_wheel", "-d", f"{tmp}/dist"],
                cwd=src,
                env=env,
                check=True,
            )
            wheel_dir.mkdir(parents=True, exist_ok=True)
            for whl in pathlib.Path(tmp, "dist").glob("deep_ep-*.whl"):
                partial = wheel_dir / f".{whl.name}.partial"
                shutil.copyfile(whl, partial)
                partial.rename(wheel_dir / whl.name)
        wheels = sorted(wheel_dir.glob("deep_ep-*.whl"))
    subprocess.run(  # nosec
        ["uv", "pip", "install", "--no-deps", *map(str, wheels)], check=True
    )
    subprocess.run(  # nosec
        [sys.executable, "-c", "import deep_ep; print('DeepEP:', deep_ep.Buffer)"],
        check=True,
    )
    return built


def run_cmd(cmd: str, run_folder: str):
    import subprocess  # nosec

    refresh_source(run_folder)

    # Propagate errors from subprocess.
    if exit_code := subprocess.call(cmd.split(), cwd=run_folder):  # nosec
        exit(exit_code)


@app.function(
    image=cicd_image,
    gpu=GPU_CONFIG,
    timeout=120 * 60,
    cpu=16.0,
    memory=(
        32768
        if os.environ.get("MULTIGPU_TEST_SUITE") in {"nf4", "nvfp4"}
        else 131072 * N_GPUS
    ),
    volumes=VOLUME_CONFIG,
)
def cicd_pytest():
    suite = os.environ.get("MULTIGPU_TEST_SUITE", "multigpu")
    scripts = {
        "multigpu": "./cicd/multigpu.sh",
        "nf4": "bash ./cicd/nf4.sh",
        "nvfp4": "bash ./cicd/nvfp4.sh",
        "multigpu_selected": "bash ./cicd/multigpu_selected.sh",
        "gpu_slow_kernels": "bash ./cicd/gpu_slow.sh",
        "gpu_slow_distributed": "bash ./cicd/gpu_slow.sh",
        "gpu_slow_four_rank": "bash ./cicd/gpu_slow.sh",
    }
    if suite in DEEP_EP_SUITES and install_deep_ep(DEEP_EP_WHEELS):
        deep_ep_volume.commit()
    run_cmd(scripts[suite], "/workspace/axolotl")


@app.local_entrypoint()
def main():
    cicd_pytest.remote()
