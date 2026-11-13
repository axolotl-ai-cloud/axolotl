"""Modal app to run CUDA-heavy single-GPU tests."""

import os

from .modal_testmon import run_local, run_remote
from .single_gpu import GPU_CONFIG, VOLUME_CONFIG, app, cicd_image, refresh_source


@app.function(
    image=cicd_image,
    gpu=GPU_CONFIG,
    timeout=90 * 60,
    cpu=8.0,
    memory=131072,
    volumes=VOLUME_CONFIG,
)
def cicd_cuda_kernels(state: dict[str, bytes], full: bool, select: bool):
    refresh_source("/workspace/axolotl")
    os.environ["AXOLOTL_DATASET_NUM_PROC"] = "8"
    return run_remote("./cicd/cicd_cuda_kernels.sh", state, full, select, False)


@app.local_entrypoint()
def main():
    run_local(cicd_cuda_kernels)
