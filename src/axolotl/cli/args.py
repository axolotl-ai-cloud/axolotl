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
        default=False,
        metadata={
            "help": (
                "Deprecated in v0.13.0, will be removed in v0.14.0. For streaming "
                "datasets, use 'axolotl train' and set 'streaming: true' in your YAML "
                "config, or pass --streaming instead in the CLI."
            )
        },
    )


@dataclass
class TrainerCliArgs:
    """Dataclass with CLI arguments for `axolotl train` command."""

    debug: bool = field(default=False)
    debug_text_only: bool = field(default=False)
    debug_num_examples: int = field(default=0)
    prompter: Optional[str] = field(default=None)
    shard: bool = field(default=False)


@dataclass
class VllmServeCliArgs:
    """Dataclass with CLI arguments for `axolotl vllm-serve` command."""

    tensor_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    data_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of data parallel workers to use for vLLM serving. This controls how many model replicas are used for parallel inference."
        },
    )
    host: Optional[str] = field(
        default=None,  # nosec B104
        metadata={"help": "Host address to run the server on."},
    )
    port: Optional[int] = field(
        default=None,
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
    serve_module: Optional[str] = field(
        default=None,
        metadata={
            "help": "Module to serve. If not set, the default module will be used."
        },
    )

    enable_reasoning: Optional[bool] = field(
        default=None,
    )

    reasoning_parser: Optional[str] = field(
        default=None,
    )


@dataclass
class QuantizeCliArgs:
    """Dataclass with CLI arguments for `axolotl quantize` command."""

    base_model: Optional[str] = field(default=None)
    weight_dtype: Optional[str] = field(default=None)
    activation_dtype: Optional[str] = field(default=None)
    quantize_embedding: Optional[bool] = field(default=None)
    group_size: Optional[int] = field(default=None)
    output_dir: Optional[str] = field(default=None)
    hub_model_id: Optional[str] = field(default=None)


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
