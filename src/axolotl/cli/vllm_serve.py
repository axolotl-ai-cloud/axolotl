"""
CLI to start the vllm server for online RL
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import trl
from trl.scripts.vllm_serve import ScriptArguments

from axolotl.cli.config import load_cfg


@dataclass
class AxolotlScriptArguments(ScriptArguments):
    """
    Additional arguments for the VLLM server
    """

    reasoning_parser: str = field(default="", kw_only=True)
    enable_reasoning: bool | None = field(default=None, kw_only=True)


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
    patch_vllm_worker()
    cfg = load_cfg(config)
    model = cfg.base_model

    serve_module = cli_args.get("serve_module", "trl.scripts.vllm_serve")
    vllm_serve_main = getattr(__import__(serve_module, fromlist=["main"]), "main")

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
    reasoning_parser = (
        cli_args.get("reasoning_parser") or cfg.vllm.reasoning_parser or ""
    )
    enable_reasoning = (
        cli_args.get("enable_reasoning") or cfg.vllm.enable_reasoning or False
    )

    # pylint: disable=unexpected-keyword-arg
    vllm_script_args = AxolotlScriptArguments(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        host=host,
        port=port,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        max_model_len=max_model_len,
        enable_prefix_caching=enable_prefix_caching,
        reasoning_parser=reasoning_parser,
        enable_reasoning=enable_reasoning,
    )
    vllm_serve_main(vllm_script_args)


def patch_vllm_worker():
    from multiprocessing.connection import Connection

    from vllm import LLM

    def llm_worker(
        script_args: AxolotlScriptArguments,
        data_parallel_rank: int,
        master_port: int,
        connection: Connection,
    ) -> None:
        # Set required environment variables for DP to work with vLLM
        os.environ["VLLM_DP_RANK"] = str(data_parallel_rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(data_parallel_rank)
        os.environ["VLLM_DP_SIZE"] = str(script_args.data_parallel_size)
        os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

        llm = LLM(
            model=script_args.model,
            revision=script_args.revision,
            tensor_parallel_size=script_args.tensor_parallel_size,
            gpu_memory_utilization=script_args.gpu_memory_utilization,
            enforce_eager=script_args.enforce_eager,
            dtype=script_args.dtype,
            # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
            # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
            # This is particularly useful here because we generate completions from the same prompts.
            enable_prefix_caching=script_args.enable_prefix_caching,
            kv_cache_dtype=script_args.kv_cache_dtype,
            max_model_len=script_args.max_model_len,
            worker_extension_cls="trl.scripts.vllm_serve.WeightSyncWorkerExtension",
            enable_reasoning=script_args.enable_reasoning,
            reasoning_parser=script_args.reasoning_parser,
        )

        # Send ready signal to parent process
        connection.send({"status": "ready"})

        while True:
            # Wait for commands from the parent process
            try:
                command = connection.recv()
            except KeyboardInterrupt:
                llm.collective_rpc(method="close_communicator")
                break

            # Handle commands
            if command["type"] in ["call", "fire_and_forget"]:
                method_name = command["method"]
                args, kwargs = command.get("args", ()), command.get("kwargs", {})
                method = getattr(llm, method_name)
                result = method(*args, **kwargs)
                if command["type"] == "call":
                    connection.send(result)
            elif command["type"] == "shutdown":
                break

    trl.scripts.vllm_serve.llm_worker = llm_worker
