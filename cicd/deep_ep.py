"""DeepEP for the CI Modal lanes: a wheel built once per source, recipe and ABI, cached on a volume.

The CI image is rebuilt whenever DEPS_HASH changes, so the build lives outside it.
"""

import hashlib
import inspect
import os
import pathlib

import modal

DEEP_EP_WHEELS = "/workspace/data/deep-ep-wheels"
deep_ep_volume = modal.Volume.from_name(
    "axolotl-ci-deep-ep-wheels", create_if_missing=True
)

# Same unpatched recipe as docker/Dockerfile-cloud-blackwell-uv, pinned to the commit so a moved
# tag cannot be served from a stale cache entry.
DEEP_EP_COMMIT = "9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee"  # v1.2.1
DEEP_EP_ARCHES = {"9.0", "10.0"}
DEEP_EP_ABI = """
import importlib.metadata as md, sys, torch
nvshmem = md.version(f"nvidia-nvshmem-cu{torch.version.cuda.split('.')[0]}")
print(f"torch{torch.__version__}-cp{sys.version_info[0]}{sys.version_info[1]}-nvshmem{nvshmem}")
"""


def _build_wheel(arch: str, src: str, dist: str) -> None:
    import subprocess  # nosec
    import sys

    os.makedirs(src)
    for args in (
        ["init", "-q"],
        [
            "fetch",
            "-q",
            "--depth=1",
            "https://github.com/deepseek-ai/DeepEP.git",
            DEEP_EP_COMMIT,
        ],
        ["checkout", "-q", "FETCH_HEAD"],
    ):
        subprocess.run(["git", *args], cwd=src, check=True)  # nosec
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
        [sys.executable, "setup.py", "bdist_wheel", "-d", dist],
        cwd=src,
        env=env,
        check=True,
    )


def deep_ep_cache_key(arch: str, abi: str) -> str:
    """Cache directory for a wheel: DeepEP commit, build recipe, GPU arch and torch ABI."""
    recipe = hashlib.sha256(inspect.getsource(_build_wheel).encode()).hexdigest()[:10]
    return f"{DEEP_EP_COMMIT[:12]}-recipe{recipe}-sm{arch}-{abi}"


def install_deep_ep(wheel_cache: str) -> bool:
    """Install DeepEP from the cached wheel for this source/recipe/ABI, building it on a miss.

    Returns whether a wheel was built.
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

    def install(wheels):
        subprocess.run(  # nosec
            ["uv", "pip", "install", "--no-deps", *map(str, wheels)], check=True
        )
        subprocess.run(  # nosec
            [sys.executable, "-c", "import deep_ep; print('DeepEP:', deep_ep.Buffer)"],
            check=True,
        )

    wheel_dir = pathlib.Path(wheel_cache) / deep_ep_cache_key(arch, abi)
    wheels = sorted(wheel_dir.glob("deep_ep-*.whl"))
    if wheels:
        install(wheels)
        return False
    print(f"DeepEP: building {wheel_dir.name}", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        _build_wheel(arch, f"{tmp}/DeepEP", f"{tmp}/dist")
        built = sorted(pathlib.Path(tmp, "dist").glob("deep_ep-*.whl"))
        # validate before caching: the volume can commit in the background, and a cached wheel
        # is never rebuilt
        install(built)
        wheel_dir.mkdir(parents=True, exist_ok=True)
        for whl in built:
            partial = wheel_dir / f".{whl.name}.partial"
            shutil.copyfile(whl, partial)
            partial.rename(wheel_dir / whl.name)
    return True
