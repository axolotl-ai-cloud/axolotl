"""
CLI to start the vllm server for online RL
"""

from pathlib import Path
from typing import Union

from trl.scripts.vllm_serve import ScriptArguments
from trl.scripts.vllm_serve import main as vllm_serve_main

from axolotl.cli.config import load_cfg


def do_vllm_serve(
    config: Union[Path, str],
    cli_args: dict,
):
    """
    Starts the VLLM server for serving LLM models used for online RL

    Args
        :param cfg: Parsed doct of the YAML config
        :param cli_args: dict of additional command-line arguments of type VllmServeCliArgs

    Returns:
        process_id: the process id of the started VLLM server
    """
    cfg = load_cfg(config)
    model = cfg.base_model

    tensor_parallel_size = (
        cli_args.get("tensor_parallel_size") or cfg.vllm.tensor_parallel_size
    )
    host = cli_args.get("host") or cfg.vllm.host
    port = cli_args.get("port") or cfg.vllm.port
    gpu_memory_utilization = (
        cli_args.get("gpu_memory_utilization") or cfg.vllm.gpu_memory_utilization
    )
    dtype = cli_args.get("dtype") or cfg.vllm.dtype
    max_model_len = cli_args.get("max_model_len") or cfg.vllm.max_model_len
    enable_prefix_caching = (
        cli_args.get("enable_prefix_caching") or cfg.vllm.enable_prefix_caching
    )

    vllm_script_args = ScriptArguments(
        model,
        tensor_parallel_size=tensor_parallel_size,
        host=host,
        port=port,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        max_model_len=max_model_len,
        enable_prefix_caching=enable_prefix_caching,
    )
    vllm_serve_main(vllm_script_args)
