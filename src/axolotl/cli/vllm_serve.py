"""
CLI to start the vllm server for online RL
"""

import json
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from axolotl.cli.config import load_cfg
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# `vllm serve --max-lora-rank` only accepts these values.
VLLM_MAX_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)

REMOVED_SERVE_MODULES = {"axolotl.scripts.vllm_serve_lora"}


@dataclass
class VllmServeArgs:
    """
    Arguments for `vllm serve`. A superset of `trl.scripts.vllm_serve.ScriptArguments`,
    so a custom `serve_module` written against TRL's `main(script_args)` keeps working.
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
    enforce_eager: bool | None = False
    kv_cache_dtype: str = "auto"
    trust_remote_code: bool = False
    log_level: str = "info"
    vllm_model_impl: str = "vllm"
    distributed_executor_backend: str | None = None
    speculative_config: str | None = None
    reasoning_parser: str | None = None
    worker_extension_cls: str | None = None
    enable_lora: bool = False
    max_lora_rank: int = 64
    extra_args: list[str] = field(default_factory=list)


def round_up_lora_rank(rank: int) -> int:
    for allowed in VLLM_MAX_LORA_RANKS:
        if rank <= allowed:
            return allowed
    raise ValueError(
        f"lora_r={rank} exceeds the largest LoRA rank vLLM supports "
        f"({VLLM_MAX_LORA_RANKS[-1]})"
    )


def build_vllm_serve_command(args: VllmServeArgs) -> list[str]:
    """Build the `vllm serve` command line that TRL's VLLMClient expects to talk to."""
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--data-parallel-size",
        str(args.data_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--dtype",
        args.dtype,
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--model-impl",
        args.vllm_model_impl,
        "--uvicorn-log-level",
        args.log_level,
    ]
    if args.revision is not None:
        command += ["--revision", args.revision]
    if args.max_model_len is not None:
        command += ["--max-model-len", str(args.max_model_len)]
    if args.enable_prefix_caching is not None:
        command += [
            "--enable-prefix-caching"
            if args.enable_prefix_caching
            else "--no-enable-prefix-caching"
        ]
    if args.enforce_eager:
        command += ["--enforce-eager"]
    if args.trust_remote_code:
        command += ["--trust-remote-code"]
    if args.distributed_executor_backend is not None:
        command += ["--distributed-executor-backend", args.distributed_executor_backend]
    if args.speculative_config is not None:
        command += ["--speculative-config", args.speculative_config]
    if args.reasoning_parser:
        command += ["--reasoning-parser", args.reasoning_parser]
    if args.worker_extension_cls:
        command += ["--worker-extension-cls", args.worker_extension_cls]
    if args.enable_lora:
        command += [
            "--enable-lora",
            "--max-lora-rank",
            str(round_up_lora_rank(args.max_lora_rank)),
        ]

    # NCCL weight-transfer engine for VLLMClient.update_named_params; processed
    # logprobs so importance-sampling correction sees temperature-scaled values.
    command += [
        "--weight-transfer-config",
        json.dumps({"backend": "nccl"}),
        "--logprobs-mode",
        "processed_logprobs",
        "--max-logprobs",
        "-1",
    ]
    return command + list(args.extra_args)


def build_vllm_serve_env(args: VllmServeArgs) -> dict[str, str]:
    env = os.environ.copy()
    # /init_weight_transfer_engine, /update_weights and /reset_prefix_cache are dev-mode routes.
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if args.enable_lora:
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    return env


def main(args: VllmServeArgs):
    command = build_vllm_serve_command(args)
    env = build_vllm_serve_env(args)
    LOG.info("Starting vLLM: %s", shlex.join(["vllm", *command[3:]]))
    os.execve(sys.executable, command, env)  # nosec B606


def do_vllm_serve(
    config: Union[Path, str],
    cli_args: dict,
):
    """
    Starts the VLLM server for serving LLM models used for online RL

    Args
        :param config: Parsed dict of the YAML config
        :param cli_args: dict of additional command-line arguments of type VllmServeCliArgs
    """
    cfg = load_cfg(config)

    serve_module = cli_args.get("serve_module") or getattr(
        cfg.vllm, "serve_module", None
    )
    if serve_module in REMOVED_SERVE_MODULES:
        LOG.warning(
            "`%s` was removed; `vllm serve` now provides native LoRA loading. "
            "Ignoring `serve_module`.",
            serve_module,
        )
        serve_module = None
    if cli_args.get("enable_reasoning") or cfg.vllm.enable_reasoning:
        LOG.warning(
            "`enable_reasoning` is ignored: vLLM enables reasoning whenever "
            "`reasoning_parser` is set."
        )

    tensor_parallel_size = (
        cli_args.get("tensor_parallel_size") or cfg.vllm.tensor_parallel_size or 1
    )
    data_parallel_size = (
        cli_args.get("data_parallel_size") or cfg.vllm.data_parallel_size or 1
    )
    # Booleans check for None so an explicit CLI False can disable a config-
    # enabled option (`cli or cfg` would let a falsy CLI value fall through).
    cli_prefix = cli_args.get("enable_prefix_caching")
    enable_prefix_caching = (
        cfg.vllm.enable_prefix_caching if cli_prefix is None else cli_prefix
    )
    cli_enforce_eager = cli_args.get("enforce_eager")
    cfg_enforce_eager = getattr(cfg.vllm, "enforce_eager", None)
    raw_enforce_eager = (
        cfg_enforce_eager if cli_enforce_eager is None else cli_enforce_eager
    )

    enable_lora = bool(cfg.trl and getattr(cfg.trl, "vllm_lora_sync", False))

    args = VllmServeArgs(
        model=cfg.base_model,
        revision=cfg.revision_of_model,
        trust_remote_code=bool(cfg.trust_remote_code),
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        host=cli_args.get("host") or cfg.vllm.host,
        port=cli_args.get("port") or cfg.vllm.port,
        gpu_memory_utilization=(
            cli_args.get("gpu_memory_utilization") or cfg.vllm.gpu_memory_utilization
        ),
        dtype=cli_args.get("dtype") or cfg.vllm.dtype,
        max_model_len=cli_args.get("max_model_len") or cfg.vllm.max_model_len,
        enable_prefix_caching=enable_prefix_caching,
        enforce_eager=bool(raw_enforce_eager),
        reasoning_parser=(
            cli_args.get("reasoning_parser") or cfg.vllm.reasoning_parser or None
        ),
        worker_extension_cls=getattr(cfg.vllm, "worker_extension_cls", None),
        enable_lora=enable_lora,
        max_lora_rank=cfg.lora_r or 64,
    )

    if serve_module is not None:
        __import__(serve_module, fromlist=["main"]).main(args)
    else:
        main(args)
