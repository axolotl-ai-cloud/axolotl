"""Modal app to run CUDA-heavy single-GPU tests."""

from .single_gpu import GPU_CONFIG, VOLUME_CONFIG, app, cicd_image, run_cmd


@app.function(
    image=cicd_image,
    gpu=GPU_CONFIG,
    timeout=90 * 60,
    cpu=8.0,
    memory=131072,
    volumes=VOLUME_CONFIG,
)
def cicd_cuda_kernels():
    run_cmd("./cicd/cicd_cuda_kernels.sh", "/workspace/axolotl")


@app.local_entrypoint()
def main():
    cicd_cuda_kernels.remote()
