"""Modal app to run the B200 (sm_100)-only test suite."""

from .single_gpu import VOLUME_CONFIG, app, cicd_image, run_cmd


@app.function(
    image=cicd_image,
    # Hardcoded on purpose: every test in this suite hard-gates on sm_100, so
    # any other GPU type would silently skip everything and report green.
    gpu="B200:1",
    timeout=60 * 60,
    cpu=8.0,
    memory=131072,
    volumes=VOLUME_CONFIG,
)
def cicd_b200():
    run_cmd("./cicd/cicd_b200.sh", "/workspace/axolotl")


@app.local_entrypoint()
def main():
    cicd_b200.remote()
