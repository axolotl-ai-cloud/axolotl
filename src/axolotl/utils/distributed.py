"""This module contains utility methods related to distributed processing"""

from contextlib import contextmanager
from datetime import datetime, timezone

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import broadcast

accelerate = None  # pylint: disable=invalid-name


def format_ts(timestamp: datetime) -> str:
    """Generates a filesystem-friently ISO8601 timestamp string

    Parameters
    ----------
    timestamp : datetime
        Source timestamp

    Returns
    -------
    str
        Timestamp string
    """

    # Define the format for the timestamp. 'Z' is only appended if the time is in UTC.
    time_format = "%Y%m%dT%H%M%S"
    time_format += "Z" if timestamp.tzinfo == timezone.utc else ""

    return timestamp.strftime(time_format)


def gen_run_id(accelerator: Accelerator) -> str:
    """Generates a multi-process safe run_id based on the current timestamp of the
    main process.

    Returns
    -------
    str
        Datestamp-based run_id
    """

    # Generate a run_id from the current timestamp on the main process then broadcast
    # to all workers. If it isn't done this way there will be millisecond to second
    # variations to the timestamps on each workerNote that this needs to be a PyTorch
    # tensor to work properly, feels like a hack and I'm open to any better ideas.
    run_timestamp = torch.zeros([1]).to(accelerator.device)
    if accelerator.is_main_process:
        run_timestamp = torch.tensor(datetime.now(timezone.utc).timestamp()).to(
            accelerator.device
        )

    broadcast(run_timestamp)
    run_datetime = datetime.fromtimestamp(run_timestamp.item(), tz=timezone.utc)
    run_id = format_ts(run_datetime)

    return run_id


def load_accelerate():
    global accelerate  # pylint: disable=global-statement
    accelerate = Accelerator()


def is_distributed():
    """
    Check if distributed training is initialized.
    """
    global accelerate  # pylint: disable=global-statement
    if not accelerate:
        accelerate = Accelerator()
    return dist.is_available() and dist.is_initialized()


def barrier():
    """
    Acts as a barrier to wait for all processes. This ensures that all processes
    reach the barrier before proceeding further.
    """
    if is_distributed():
        dist.barrier()


def is_main_process():
    """
    Check if the current process is the main process.
    If not in distributed mode, always return True.
    """
    if not is_distributed():
        return True
    return dist.get_rank() == 0


@contextmanager
def zero_first(is_main):
    """
    runs the wrapped context so that rank 0 runs first before other ranks
    """
    if not is_main:  # other ranks wait first
        barrier()
    yield
    if is_main:  # then rank 0 waits after it has run the context
        barrier()
