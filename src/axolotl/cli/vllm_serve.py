"""
CLI to start the vllm server for online RL
"""

import json
import os
import sys
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Union

from packaging.version import Version

from axolotl.cli.config import load_cfg
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Superseded by vLLM's native server; ignored with a warning.
LEGACY_SERVE_MODULES = ("axolotl.scripts.vllm_serve_lora", "trl.scripts.vllm_serve")
LEGACY_WORKER_EXTENSION_PREFIXES = ("axolotl.scripts.", "trl.scripts.vllm_serve.")
SUPPORTED_MAX_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def round_max_lora_rank(rank: int) -> int:
    for supported in SUPPORTED_MAX_LORA_RANKS:
        if rank <= supported:
            return supported
    raise ValueError(
        f"LoRA rank {rank} exceeds the largest rank vLLM supports "
        f"({SUPPORTED_MAX_LORA_RANKS[-1]}); trl.vllm_lora_sync cannot be used"
    )


@dataclass
class VllmServeArguments:
    """
    Arguments for vLLM's native OpenAI server as used by TRL trainers
    """

    model: str
    revision: str | None = None
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    host: str = "0.0.0.0"  # nosec B104
    port: int = 8000
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    max_model_len: int | None = None
    enable_prefix_caching: bool | None = None
    enforce_eager: bool = False
    kv_cache_dtype: str = "auto"
    trust_remote_code: bool = False
    log_level: str = "info"
    vllm_model_impl: str = "vllm"
    reasoning_parser: str = ""
    enable_reasoning: bool | None = None
    enable_lora: bool = False
    max_lora_rank: int = 64
    max_loras: int = 2
    worker_extension_cls: str | None = None


AxolotlScriptArguments = VllmServeArguments


def _vllm_at_least(min_version: str) -> bool:
    try:
        return Version(version("vllm")) >= Version(min_version)
    except PackageNotFoundError:
        return False


def build_command(script_args: VllmServeArguments) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        script_args.model,
        "--host",
        script_args.host,
        "--port",
        str(script_args.port),
        "--tensor-parallel-size",
        str(script_args.tensor_parallel_size),
        "--data-parallel-size",
        str(script_args.data_parallel_size),
        "--gpu-memory-utilization",
        str(script_args.gpu_memory_utilization),
        "--dtype",
        script_args.dtype,
        "--kv-cache-dtype",
        script_args.kv_cache_dtype,
        "--model-impl",
        script_args.vllm_model_impl,
        "--uvicorn-log-level",
        script_args.log_level,
    ]
    if script_args.revision is not None:
        command += ["--revision", script_args.revision]
    if script_args.max_model_len is not None:
        command += ["--max-model-len", str(script_args.max_model_len)]
    if script_args.enable_prefix_caching is not None:
        command += [
            "--enable-prefix-caching"
            if script_args.enable_prefix_caching
            else "--no-enable-prefix-caching"
        ]
    if script_args.enforce_eager:
        command += ["--enforce-eager"]
    if script_args.trust_remote_code:
        command += ["--trust-remote-code"]
    if script_args.reasoning_parser and script_args.enable_reasoning is not False:
        command += ["--reasoning-parser", script_args.reasoning_parser]
    if script_args.enable_lora:
        command += [
            "--enable-lora",
            "--max-lora-rank",
            str(round_max_lora_rank(script_args.max_lora_rank)),
            "--max-loras",
            str(script_args.max_loras),
            # With data parallelism vLLM starts one API server per rank, and a
            # runtime-loaded adapter only registers on the server that loaded it.
            "--api-server-count",
            "1",
        ]
    if script_args.worker_extension_cls:
        command += ["--worker-extension-cls", script_args.worker_extension_cls]
    # Same fixed flags as TRL's `trl vllm-serve`: scale-out adds the render and
    # generate routes used for multimodal prompts, the NCCL transfer engine
    # receives merged weights, and processed logprobs feed importance sampling.
    if _vllm_at_least("0.30.0"):
        command += ["--enable-scale-out"]
    command += [
        "--weight-transfer-config",
        json.dumps({"backend": "nccl"}),
        "--logprobs-mode",
        "processed_logprobs",
        "--max-logprobs",
        "-1",
    ]
    return command


def build_env(
    script_args: VllmServeArguments, base_env: dict[str, str] | None = None
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if script_args.enable_lora:
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
    return env


def serve(script_args: VllmServeArguments):
    command = build_command(script_args)
    env = build_env(script_args)
    os.execve(sys.executable, command, env)  # nosec B606


def _drop_legacy_serve_options(
    serve_module: str | None, worker_extension_cls: str | None
) -> tuple[str | None, str | None]:
    if serve_module in LEGACY_SERVE_MODULES:
        LOG.warning(
            f"vllm.serve_module '{serve_module}' is deprecated and ignored; "
            "using vLLM's native server"
        )
        serve_module = None
    if worker_extension_cls and worker_extension_cls.startswith(
        LEGACY_WORKER_EXTENSION_PREFIXES
    ):
        LOG.warning(
            f"vllm.worker_extension_cls '{worker_extension_cls}' is deprecated and "
            "ignored; vLLM's native weight transfer replaces it"
        )
        worker_extension_cls = None
    return serve_module, worker_extension_cls


def _max_lora_rank(cfg) -> int | None:
    ranks = [cfg.lora_r, *(cfg.lora_rank_pattern or {}).values()]
    return max((r for r in ranks if r), default=None)


def do_vllm_serve(
    config: Union[Path, str],
    cli_args: dict,
):
    """
    Starts the VLLM server for serving LLM models used for online RL

    Args
        :param config: Parsed dict of the YAML config
        :param cli_args: dict of additional command-line arguments of type VllmServeCliArgs

    Returns:
        None; the process is replaced by the native vLLM server
    """
    cfg = load_cfg(config)
    vllm_cfg = cfg.vllm

    def cli_or_cfg(key: str, default=None):
        return cli_args.get(key) or vllm_cfg[key] or default

    def cli_or_cfg_flag(key: str):
        # Checks None so an explicit CLI False can disable a config-enabled option.
        value = cli_args.get(key)
        return vllm_cfg[key] if value is None else value

    serve_module, worker_extension_cls = _drop_legacy_serve_options(
        cli_or_cfg("serve_module"), vllm_cfg.worker_extension_cls
    )
    vllm_script_args = VllmServeArguments(
        model=cfg.base_model,
        revision=cfg.revision_of_model,
        trust_remote_code=bool(cfg.trust_remote_code),
        tensor_parallel_size=cli_or_cfg("tensor_parallel_size", 1),
        data_parallel_size=cli_or_cfg("data_parallel_size", 1),
        host=cli_or_cfg("host"),
        port=cli_or_cfg("port"),
        gpu_memory_utilization=cli_or_cfg("gpu_memory_utilization", 0.9),
        dtype=cli_or_cfg("dtype"),
        max_model_len=cli_or_cfg("max_model_len"),
        enable_prefix_caching=cli_or_cfg_flag("enable_prefix_caching"),
        enforce_eager=bool(vllm_cfg.enforce_eager),
        reasoning_parser=cli_or_cfg("reasoning_parser", ""),
        enable_reasoning=cli_or_cfg_flag("enable_reasoning"),
        enable_lora=bool(getattr(cfg.trl, "vllm_lora_sync", False)),
        max_lora_rank=_max_lora_rank(cfg) or VllmServeArguments.max_lora_rank,
        worker_extension_cls=worker_extension_cls,
    )

    if serve_module is None:
        serve(vllm_script_args)
    else:
        import_module(serve_module).main(vllm_script_args)
