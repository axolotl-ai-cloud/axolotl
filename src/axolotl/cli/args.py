"""Module for axolotl CLI command arguments."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PreprocessCliArgs:
    """Dataclass with CLI arguments for `axolotl preprocess` command."""

    debug: bool = field(default=False)
    debug_text_only: bool = field(default=False)
    debug_num_examples: int = field(default=1)
    prompter: Optional[str] = field(default=None)
    download: Optional[bool] = field(default=True)
    iterable: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Use IterableDataset for streaming processing of large datasets"
        },
    )


@dataclass
class TrainerCliArgs:
    """Dataclass with CLI arguments for `axolotl train` command."""

    debug: bool = field(default=False)
    debug_text_only: bool = field(default=False)
    debug_num_examples: int = field(default=0)
    merge_lora: bool = field(default=False)
    prompter: Optional[str] = field(default=None)
    shard: bool = field(default=False)
    main_process_port: Optional[int] = field(default=None)
    num_processes: Optional[int] = field(default=None)


@dataclass
class VllmServeCliArgs:
    """Dataclass with CLI arguments for `axolotl vllm-serve` command."""

    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    host: str = field(
        default="0.0.0.0",  # nosec B104
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: Optional[float] = field(
        default=None,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the "
            "hardware support this feature."
        },
    )


@dataclass
class EvaluateCliArgs:
    """Dataclass with CLI arguments for `axolotl evaluate` command."""

    debug: bool = field(default=False)
    debug_text_only: bool = field(default=False)
    debug_num_examples: int = field(default=0)


@dataclass
class InferenceCliArgs:
    """Dataclass with CLI arguments for `axolotl inference` command."""

    prompter: Optional[str] = field(default=None)
