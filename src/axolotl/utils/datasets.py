"""helper functions for datasets"""

import os


def get_default_process_count():
    if axolotl_dataset_num_proc := os.environ.get("AXOLOTL_DATASET_NUM_PROC"):
        return int(axolotl_dataset_num_proc)
    if axolotl_dataset_processes := os.environ.get("AXOLOTL_DATASET_PROCESSES"):
        return int(axolotl_dataset_processes)
    if runpod_cpu_count := os.environ.get("RUNPOD_CPU_COUNT"):
        return int(runpod_cpu_count)
    return os.cpu_count()
