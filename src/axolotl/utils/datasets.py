"""helper functions for datasets"""

import os

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Multimodal rows can carry tens of MB of pixel_values each, so datasets' default
# 1000-row map/writer buffers would hold many GB per worker.
MULTIMODAL_MAP_BUFFER_SIZE = 32


def dataset_map_buffer_kwargs(
    cfg, batched: bool = False, multimodal: bool | None = None
) -> dict:
    """Buffer kwargs (`writer_batch_size`, and `batch_size` if batched) for
    `Dataset.map`/`.filter` calls over rows that may carry large media columns."""
    if multimodal is None:
        multimodal = bool(cfg.processor_type or cfg.is_multimodal)
    default = MULTIMODAL_MAP_BUFFER_SIZE if multimodal else None

    kwargs = {}
    if writer_batch_size := (cfg.dataset_writer_batch_size or default):
        kwargs["writer_batch_size"] = writer_batch_size
    if batched and (batch_size := (cfg.dataset_map_batch_size or default)):
        kwargs["batch_size"] = batch_size
    return kwargs


def get_default_process_count():
    if axolotl_dataset_num_proc := os.environ.get("AXOLOTL_DATASET_NUM_PROC"):
        return int(axolotl_dataset_num_proc)
    if axolotl_dataset_processes := os.environ.get("AXOLOTL_DATASET_PROCESSES"):
        LOG.warning(
            "AXOLOTL_DATASET_PROCESSES and `dataset_processes` are deprecated and will be "
            "removed in a future version. Please use `dataset_num_proc` instead."
        )
        return int(axolotl_dataset_processes)
    if runpod_cpu_count := os.environ.get("RUNPOD_CPU_COUNT"):
        return int(runpod_cpu_count)
    return os.cpu_count()
