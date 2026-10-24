"""DeepEP for the CI Modal lanes: a wheel built once per ABI and cached on a volume.

The CI image is rebuilt whenever DEPS_HASH changes, so the build lives outside it.
"""

import os
import pathlib

import modal

DEEP_EP_WHEELS = "/workspace/data/deep-ep-wheels"
deep_ep_volume = modal.Volume.from_name(
    "axolotl-ci-deep-ep-wheels", create_if_missing=True
)

# Same unpatched recipe as docker/Dockerfile-cloud-blackwell-uv.
DEEP_EP_REF = "v1.2.1"
DEEP_EP_ARCHES = {"9.0", "10.0"}
DEEP_EP_ABI = """
import importlib.metadata as md, sys, torch
nvshmem = md.version(f"nvidia-nvshmem-cu{torch.version.cuda.split('.')[0]}")
print(f"torch{torch.__version__}-cp{sys.version_info[0]}{sys.version_info[1]}-nvshmem{nvshmem}")
"""


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
