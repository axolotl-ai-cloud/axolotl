"""
CLI to start the vllm server for online RL
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

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
    cfg = load_cfg(config)
    model = cfg.base_model

    # Determine serve module: explicit CLI/config > default (axolotl's LoRA-aware serve).
    # We default to axolotl's serve module instead of TRL's because TRL's sends
    # truncate_prompt_tokens which is unsupported in vLLM 0.17+.
    serve_module = cli_args.get("serve_module") or getattr(
        cfg.vllm, "serve_module", None
    )
    if serve_module is None:
        serve_module = "axolotl.scripts.vllm_serve_lora"
    vllm_serve_main = __import__(serve_module, fromlist=["main"]).main
    tensor_parallel_size = 1
    data_parallel_size = 1

    if cli_args.get("tensor_parallel_size") or cfg.vllm.tensor_parallel_size:
        tensor_parallel_size = (
            cli_args.get("tensor_parallel_size") or cfg.vllm.tensor_parallel_size
        )
    if cli_args.get("data_parallel_size") or cfg.vllm.data_parallel_size:
        data_parallel_size = (
            cli_args.get("data_parallel_size") or cfg.vllm.data_parallel_size
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

    cli_enforce_eager = cli_args.get("enforce_eager")
    cfg_enforce_eager = getattr(cfg.vllm, "enforce_eager", None)
    raw_enforce_eager = (
        cfg_enforce_eager if cli_enforce_eager is None else cli_enforce_eager
    )
    enforce_eager = bool(raw_enforce_eager) if raw_enforce_eager is not None else False
    base_kwargs = dict(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        host=host,
        port=port,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        max_model_len=max_model_len,
        enable_prefix_caching=enable_prefix_caching,
        enforce_eager=enforce_eager,
    )

    # Use LoRAScriptArguments when serving with native LoRA support
    if serve_module == "axolotl.scripts.vllm_serve_lora":
        from axolotl.scripts.vllm_serve_lora import LoRAScriptArguments

        lora_kwargs = {}
        if hasattr(cfg, "lora_r") and cfg.lora_r:
            lora_kwargs["max_lora_rank"] = cfg.lora_r
        # Disable native LoRA in vLLM if not using vllm_lora_sync
        # (merged weight sync via batch_update doesn't need vLLM LoRA mode)
        if not getattr(cfg.trl, "vllm_lora_sync", False):
            lora_kwargs["enable_lora"] = False
        if getattr(cfg.vllm, "worker_extension_cls", None):
            lora_kwargs["worker_extension_cls"] = cfg.vllm.worker_extension_cls
        vllm_script_args = LoRAScriptArguments(**base_kwargs, **lora_kwargs)
    else:
        vllm_script_args = AxolotlScriptArguments(
            **base_kwargs,
            reasoning_parser=reasoning_parser,
            enable_reasoning=enable_reasoning,
        )

    vllm_serve_main(vllm_script_args)
