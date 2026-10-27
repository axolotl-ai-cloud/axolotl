"""Modal app to run single-GPU gpu_blackwell tests (set GPU_TYPE=B200)."""

from .deep_ep import DEEP_EP_WHEELS, deep_ep_volume, install_deep_ep
from .single_gpu import GPU_CONFIG, VOLUME_CONFIG, app, cicd_image, run_cmd


@app.function(
    image=cicd_image,
    gpu=GPU_CONFIG,
    timeout=60 * 60,
    cpu=8.0,
    memory=131072,
    volumes={**VOLUME_CONFIG, DEEP_EP_WHEELS: deep_ep_volume},
)
def cicd_blackwell():
    if install_deep_ep(DEEP_EP_WHEELS):
        deep_ep_volume.commit()
    run_cmd("./cicd/cicd_blackwell.sh", "/workspace/axolotl")


@app.local_entrypoint()
def main():
    cicd_blackwell.remote()
