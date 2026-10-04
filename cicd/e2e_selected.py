"""Modal app to run the selector-picked single-GPU e2e test files."""

from .single_gpu import GPU_CONFIG, VOLUME_CONFIG, app, cicd_image, run_cmd


@app.function(
    image=cicd_image,
    gpu=GPU_CONFIG,
    timeout=120 * 60,
    cpu=8.0,
    memory=131072,
    volumes=VOLUME_CONFIG,
)
def cicd_pytest_selected():
    run_cmd("bash ./cicd/cicd_selected.sh", "/workspace/axolotl")


@app.local_entrypoint()
def main():
    cicd_pytest_selected.remote()
