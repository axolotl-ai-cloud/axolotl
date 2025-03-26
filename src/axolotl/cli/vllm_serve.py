"""
CLI to start the vllm server for online RL
"""

import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from typing import Any, Union

import requests

from axolotl.cli.args import VllmServeCliArgs
from axolotl.cli.config import load_cfg


def do_vllm_serve(
    config: Union[Path, str],
    cli_args: VllmServeCliArgs,
):
    """
    Starts the VLLM server for serving LLM models used for online RL

    Args
        :param cfg: Parsed doct of the YAML config
        :param cli_args: additional command-line arguments

    Returns:
        process_id: the process id of the started VLLM server
    """
    cfg = load_cfg(config)
    model = cfg.base_model
    tensor_parallel_size = cli_args.tensor_parallel_size
    host = cli_args.host
    port = cli_args.port

    gpu_memory_utilization = (
        cfg.trl.vllm_gpu_memory_utilization or cli_args.gpu_memory_utilization
    )
    dtype = cfg.trl.vllm_dtype or cli_args.dtype
    max_model_len = cfg.trl.vllm_max_model_len or cli_args.max_model_len
    enable_prefix_caching = (
        cfg.trl.enable_prefix_caching or cli_args.enable_prefix_caching
    )
    wait = cli_args.wait

    start_vllm_kwargs: dict[str, Any] = {}
    if wait is not None:
        start_vllm_kwargs["wait"] = wait

    process_id: int = start_vllm(
        model,
        tensor_parallel_size=tensor_parallel_size,
        host=host,
        port=port,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        max_model_len=max_model_len,
        enable_prefix_caching=enable_prefix_caching,
        **start_vllm_kwargs,
    )

    return process_id


def start_vllm(
    model: str, env: dict | None = None, wait: int | None = None, quiet=False, **kwargs
) -> int:
    cmd = [sys.executable, "-m", "trl.scripts.vllm_serve", "--model", model]

    if tensor_parallel_size := kwargs.get("tensor_parallel_size"):
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    if host := kwargs.get("host"):
        cmd.extend(["--host", host])
    if port := kwargs.get("port"):
        cmd.extend(["--port", str(port)])
    if gpu_memory_utilization := kwargs.get("gpu_memory_utilization"):
        cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
    if dtype := kwargs.get("dtype"):
        cmd.extend(["--dtype", dtype])
    if max_model_len := kwargs.get("max_model_len"):
        cmd.extend(["--max-model-len", str(max_model_len)])
    if kwargs.get("enable_prefix_caching"):
        cmd.extend(["--enable-prefix-caching"])

    # print out the command to be executed
    print(" ".join(cmd))

    # start `trl vllm-serve` command in the background and capture the process id
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        cmd,
        env=env,
        stdout=subprocess.DEVNULL if quiet else subprocess.PIPE,
        stderr=subprocess.DEVNULL if quiet else subprocess.PIPE,
    )  # nosec B603

    # print out the process id so the user can easily kill it later
    print(f"VLLM server process started (PID: {process.pid})")

    # wait until the http server is ready, even if it 404s, but timeout after 60 seconds
    started = False
    if wait and host and port:
        for _ in range(int(wait)):
            try:
                response = requests.get(f"http://{host}:{port}", timeout=1)
                if int(response.status_code) in [200, 404]:
                    started = True
                    break
            except requests.exceptions.RequestException:
                pass

            time.sleep(1)

    if wait and not started:
        print(
            f"VLLM server process did not start within {wait} seconds. Please check your server logs."
        )
        process.kill()
        raise RuntimeError(f"VLLM server process did not start within {wait} seconds.")

    # return the process id
    return process.pid
