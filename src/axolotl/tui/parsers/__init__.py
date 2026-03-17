"""Built-in line parsers — auto-imported to trigger @register_parser decorators."""

from axolotl.tui.parsers.deepspeed import DeepSpeedParser  # noqa: F401
from axolotl.tui.parsers.nccl import NCCLErrorParser  # noqa: F401
from axolotl.tui.parsers.raw_log import RawLogParser  # noqa: F401
from axolotl.tui.parsers.tqdm import TqdmParser  # noqa: F401
from axolotl.tui.parsers.torch_compile import TorchCompileParser  # noqa: F401
