"""helper functions for datasets"""

import os
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

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
