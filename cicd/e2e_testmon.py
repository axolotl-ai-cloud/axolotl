"""Modal app to run the single-GPU e2e suite under pytest-testmon."""

import os

import modal

from .single_gpu import GPU_CONFIG, VOLUME_CONFIG, app, cicd_image, run_cmd

TESTMON_MOUNT = "/workspace/data/testmon"
testmon_volume = modal.Volume.from_name("axolotl-ci-testmon", create_if_missing=True)


@app.function(
    image=cicd_image,
    gpu=GPU_CONFIG,
    timeout=120 * 60,
    cpu=8.0,
    memory=131072,
    volumes={**VOLUME_CONFIG, TESTMON_MOUNT: testmon_volume},
)
def cicd_pytest_testmon():
    run_cmd("bash ./cicd/cicd_testmon.sh", "/workspace/axolotl")
    if os.environ.get("GITHUB_REF") == "refs/heads/main":
        testmon_volume.commit()


@app.local_entrypoint()
def main():
    cicd_pytest_testmon.remote()
