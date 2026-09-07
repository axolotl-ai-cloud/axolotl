"""Modal app to run axolotl GPU tests"""

from .single_gpu import GPU_CONFIG, VOLUME_CONFIG, app, cicd_image, run_cmd


@app.function(
    image=cicd_image,
    gpu=GPU_CONFIG,
    timeout=120 * 60,  # 90 min
    cpu=8.0,
    memory=131072,
    volumes=VOLUME_CONFIG,
)
def cicd_pytest():
    run_cmd("./cicd/cicd.sh", "/workspace/axolotl")


@app.local_entrypoint()
def main():
    cicd_pytest.remote()
