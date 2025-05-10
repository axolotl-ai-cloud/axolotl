"""Modal app to run axolotl GPU cleanup"""

from .single_gpu import VOLUME_CONFIG, app, cicd_image, run_cmd


@app.function(
    image=cicd_image,
    timeout=60 * 60,
    cpu=8.0,
    memory=131072,
    volumes=VOLUME_CONFIG,
)
def cleanup():
    run_cmd("./cicd/cleanup.sh", "/workspace/axolotl")


@app.local_entrypoint()
def main():
    cleanup.remote()
