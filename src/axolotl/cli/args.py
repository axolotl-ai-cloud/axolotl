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
class EvaluateCliArgs:
    """Dataclass with CLI arguments for `axolotl evaluate` command."""

    debug: bool = field(default=False)
    debug_text_only: bool = field(default=False)
    debug_num_examples: int = field(default=0)


@dataclass
class InferenceCliArgs:
    """Dataclass with CLI arguments for `axolotl inference` command."""

    prompter: Optional[str] = field(default=None)
