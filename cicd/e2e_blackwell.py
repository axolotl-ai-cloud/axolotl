"""Modal app to run single-GPU gpu_blackwell tests (set GPU_TYPE=B200)."""

from .single_gpu import GPU_CONFIG, VOLUME_CONFIG, app, cicd_image, run_cmd


@app.function(
    image=cicd_image,
    gpu=GPU_CONFIG,
    timeout=60 * 60,
    cpu=8.0,
    memory=131072,
    volumes=VOLUME_CONFIG,
)
def cicd_blackwell():
    run_cmd("./cicd/cicd_blackwell.sh", "/workspace/axolotl")


@app.local_entrypoint()
def main():
    cicd_blackwell.remote()
