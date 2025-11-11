"""
Baseten Training Script for Axolotl
"""

# pylint: skip-file
import yaml
from truss.base import truss_config

# Import necessary classes from the Baseten Training SDK
from truss_train import definitions

cloud_config = yaml.safe_load(open("cloud.yaml", "r"))
gpu = cloud_config.get("gpu", "h100")
gpu_count = int(cloud_config.get("gpu_count", 1))
node_count = int(cloud_config.get("node_count", 1))
project_name = cloud_config.get("project_name", "axolotl-project") or "axolotl-project"
secrets = cloud_config.get("secrets", [])
launcher = cloud_config.get("launcher", "accelerate")
launcher_args = cloud_config.get("launcher_args", [])
script_name = "run.sh"

launcher_args_str = ""
if launcher_args:
    launcher_args_str = "-- " + " ".join(launcher_args)

# 1. Define a base image for your training job
# must use torch 2.7.0 for vllm
BASE_IMAGE = "axolotlai/axolotl:main-py3.11-cu126-2.7.1"

# 2. Define the Runtime Environment for the Training Job
# This includes start commands and environment variables.a
# Secrets from the baseten workspace like API keys are referenced using
# `SecretReference`.

env_vars = {
    "AXOLOTL_LAUNCHER": launcher,
    "AXOLOTL_LAUNCHER_ARGS": launcher_args_str,
}
for secret_name in secrets:
    env_vars[secret_name] = definitions.SecretReference(name=secret_name)

training_runtime = definitions.Runtime(
    start_commands=[  # Example: list of commands to run your training script
        f"/bin/sh -c 'chmod +x ./{script_name} && ./{script_name}'"
    ],
    environment_variables=env_vars,
)

# 3. Define the Compute Resources for the Training Job
training_compute = definitions.Compute(
    node_count=node_count,
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,
        count=gpu_count,
    ),
)

# 4. Define the Training Job
# This brings together the image, compute, and runtime configurations.
my_training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)


# This config will be pushed using the Truss CLI.
# The association of the job to the project happens at the time of push.
first_project_with_job = definitions.TrainingProject(
    name=project_name, job=my_training_job
)
