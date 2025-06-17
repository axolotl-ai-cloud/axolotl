"""Modal app to run axolotl GPU tests"""

import pathlib

from .single_gpu import GPU_CONFIG, VOLUME_CONFIG, app, cicd_image, run_cmd


@app.function(
    image=cicd_image,
    gpu=GPU_CONFIG,
    timeout=90 * 60,  # 90 min
    cpu=8.0,
    memory=131072,
    volumes=VOLUME_CONFIG,
)
def cicd_pytest():

    run_cmd("./cicd/cicd.sh", "/workspace/axolotl")

    # Read the coverage file if it exists
    coverage_file = pathlib.Path("/workspace/axolotl/e2e-coverage.xml")
    if coverage_file.exists():
        return coverage_file.read_text(encoding="utf-8")
    return None


@app.local_entrypoint()
def main():
    coverage = cicd_pytest.remote()

    # Save the coverage file to the local filesystem if it was generated
    if coverage:
        with open("e2e-coverage.xml", "w", encoding="utf-8") as f:
            f.write(coverage)
