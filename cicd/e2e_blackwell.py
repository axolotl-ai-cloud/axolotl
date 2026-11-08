"""Modal app to run single-GPU gpu_blackwell tests (set GPU_TYPE=B200)."""

from .deep_ep import DEEP_EP_WHEELS, deep_ep_volume, install_deep_ep
from .modal_testmon import run_local, run_remote
from .single_gpu import GPU_CONFIG, VOLUME_CONFIG, app, cicd_image, refresh_source


@app.function(
    image=cicd_image,
    gpu=GPU_CONFIG,
    timeout=60 * 60,
    cpu=8.0,
    memory=131072,
    volumes={**VOLUME_CONFIG, DEEP_EP_WHEELS: deep_ep_volume},
)
def cicd_blackwell(state: dict[str, bytes], full: bool, select: bool):
    refresh_source("/workspace/axolotl")
    if install_deep_ep(DEEP_EP_WHEELS):
        deep_ep_volume.commit()
    return run_remote("./cicd/cicd_blackwell.sh", state, full, select, False)


@app.local_entrypoint()
def main():
    run_local(cicd_blackwell)
