"""
Runpod serverless entrypoint handler
"""

import os

import runpod
import yaml
from huggingface_hub._login import login
from train import train
from utils import get_output_dir

BASE_VOLUME = os.environ.get("BASE_VOLUME", "/runpod-volume")
if not os.path.exists(BASE_VOLUME):
    os.makedirs(BASE_VOLUME)

logger = runpod.RunPodLogger()


async def handler(job):
    runpod_job_id = job["id"]
    inputs = job["input"]
    run_id = inputs.get("run_id", "default_run_id")
    args = inputs.get("args", {})

    # Set output directory
    output_dir = os.path.join(BASE_VOLUME, get_output_dir(run_id))
    args["output_dir"] = output_dir

    # First save args to a temporary config file
    config_path = "/workspace/test_config.yaml"

    # Add run_name and job_id to args before saving
    args["run_name"] = run_id
    args["runpod_job_id"] = runpod_job_id

    yaml_data = yaml.dump(args, default_flow_style=False)
    with open(config_path, "w", encoding="utf-8") as file:
        file.write(yaml_data)

    # Handle credentials
    credentials = inputs.get("credentials", {})

    # Workers are reused across jobs, so endpoint-level credentials have to survive
    # the per-job cleanup below while job-supplied ones must not leak into the next job.
    endpoint_credentials = {
        key: os.environ.get(key) for key in ("WANDB_API_KEY", "HF_TOKEN")
    }

    # Empty values are treated as absent so they don't clobber endpoint-level credentials.
    if credentials.get("wandb_api_key"):
        os.environ["WANDB_API_KEY"] = credentials["wandb_api_key"]
    if credentials.get("hf_token"):
        os.environ["HF_TOKEN"] = credentials["hf_token"]

    if os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])
    else:
        logger.info("No HF_TOKEN provided. Skipping login.")

    logger.info("Starting Training.")
    async for result in train(config_path):  # Pass the config path instead of args
        logger.info(result)
    logger.info("Training Complete.")

    # Cleanup
    for key, value in endpoint_credentials.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
