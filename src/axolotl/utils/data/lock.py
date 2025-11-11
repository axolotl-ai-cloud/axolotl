"""Logic for loading / preparing a dataset once over all processes."""

import time
from pathlib import Path
from typing import Any, Callable

from filelock import FileLock

from axolotl.common.const import DEFAULT_DATASET_PREPARED_PATH
from axolotl.utils.dict import DictDefault

LOCK_FILE_NAME = "datasets_prep.lock"
READY_FILE_NAME = "datasets_ready.flag"
PROCESS_COUNTER_FILE_NAME = "process_counter.txt"


class FileLockLoader:
    """
    Simple class for abstracting single process data loading / processing. The first
    process that creates a lock file does the work; the remaining procesees simply load
    the preprocessed dataset once the first process is done.
    """

    def __init__(self, cfg: DictDefault):
        self.cfg = cfg
        self.dataset_prepared_path = (
            cfg.dataset_prepared_path or DEFAULT_DATASET_PREPARED_PATH
        )
        self.lock_file_path = Path(self.dataset_prepared_path) / LOCK_FILE_NAME
        self.ready_flag_path = Path(self.dataset_prepared_path) / READY_FILE_NAME
        self.counter_path = Path(self.dataset_prepared_path) / PROCESS_COUNTER_FILE_NAME

    def load(self, load_fn: Callable[[], Any]) -> Any:
        with FileLock(str(self.lock_file_path)):
            self._increment_counter()

            if not self.ready_flag_path.exists():
                result = load_fn()
                self.ready_flag_path.touch()
                return result

            while not self.ready_flag_path.exists():
                time.sleep(1)
            return load_fn()

    def _increment_counter(self):
        """Safely increment the process counter."""
        if self.counter_path.exists():
            counter_content = self.counter_path.read_text().strip()
            count = int(counter_content) if counter_content else 0
        else:
            count = 0
        self.counter_path.write_text(str(count + 1))

    def cleanup(self):
        """Clean up ready flag when last process is done."""
        with FileLock(str(self.lock_file_path)):
            counter_content = self.counter_path.read_text().strip()
            count = int(counter_content) if counter_content else 0
            count -= 1

            if count <= 0:
                # Last process cleans everything up
                self.ready_flag_path.unlink(missing_ok=True)
                self.counter_path.unlink(missing_ok=True)
            else:
                # Still have active processes
                self.counter_path.write_text(str(count))
