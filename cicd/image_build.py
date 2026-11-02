"""Modal app that builds the CI image and starts it once, without running tests"""

from .single_gpu import app, cicd_image, run_cmd


@app.function(image=cicd_image, timeout=15 * 60, cpu=2.0, memory=16384)
def image_smoke():
    run_cmd("axolotl --help", "/workspace/axolotl")


@app.local_entrypoint()
def main():
    image_smoke.remote()
