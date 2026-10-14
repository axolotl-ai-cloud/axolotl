"""Run one Axolotl launcher per node inside the training container (SUNK or CKS)."""

import json
import os
import subprocess  # nosec B404
from pathlib import Path


def launch_command(launch, environment):
    """Derive process topology from the allocation, not from local hardware."""
    launcher = launch["launcher"]
    nodes = int(environment.get("AXOLOTL_NNODES") or environment["SLURM_NNODES"])
    rank = int(environment.get("AXOLOTL_NODE_RANK") or environment["SLURM_PROCID"])
    gpus = launch["gpus_per_node"]
    command = ["axolotl", "train", "train.yaml", "--launcher", launcher]
    args = list(launch["launcher_args"])
    if launcher == "accelerate":
        args += ["--num_machines", str(nodes), "--num_processes", str(nodes * gpus)]
        if nodes * gpus > 1 and not any(
            arg.split("=", 1)[0].lstrip("-").replace("-", "_")
            in {"use_deepspeed", "use_fsdp", "config_file"}
            for arg in args
        ):
            args += ["--multi_gpu"]
        if nodes > 1:
            args += [
                "--machine_rank",
                str(rank),
                "--main_process_ip",
                environment["MASTER_ADDR"],
                "--main_process_port",
                environment["MASTER_PORT"],
            ]
    elif launcher == "torchrun":
        args += [
            "--nnodes",
            str(nodes),
            "--nproc_per_node",
            str(gpus),
            "--node_rank",
            str(rank),
            "--master_addr",
            environment["MASTER_ADDR"],
            "--master_port",
            environment["MASTER_PORT"],
        ]
    elif launcher != "python" or nodes != 1 or gpus != 1:
        raise ValueError("The python launcher requires one node and one GPU")
    if args:
        command += ["--", *args]
    return command


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    launch = json.loads((root / "launch.json").read_text(encoding="utf-8"))
    command = launch_command(launch, os.environ)
    cuda_env = Path("/workspace/axolotl/scripts/cuda13_env.sh")
    if cuda_env.is_file():
        command = [
            "/bin/bash",
            "-c",
            'source /workspace/axolotl/scripts/cuda13_env.sh && exec "$@"',
            "axolotl-sunk",
            *command,
        ]
    subprocess.run(command, check=True)  # nosec B603


if __name__ == "__main__":
    main()
