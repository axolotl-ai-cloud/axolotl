"""Control-plane synchronization and progress reporting for staged NF4 loading."""

import os
import threading
import time
from contextlib import contextmanager
from datetime import timedelta

import torch.distributed as dist

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# only the rank-zero conversion thread writes these, so plain ints need no lock
_progress = {"tensors": 0, "bytes": 0}


def record_progress(nbytes: int) -> None:
    """Count one converted tensor so staging heartbeats can show forward progress."""
    _progress["tensors"] += 1
    _progress["bytes"] += nbytes


@contextmanager
def nf4_loading_group(cfg):
    """Keep long rank-zero staging waits off the default NCCL process group."""
    if not cfg.fsdp_config:
        yield None
        return
    seconds = int(os.environ.get("AXOLOTL_NCCL_TIMEOUT", cfg.ddp_timeout or 1800))
    if seconds <= 0:
        raise ValueError("NF4 loading timeout must be positive")
    if not dist.is_initialized():
        from axolotl.utils.distributed import init_distributed_state

        init_distributed_state()
    group = dist.new_group(backend="gloo", timeout=timedelta(seconds=seconds))
    try:
        dist.barrier(group=group)
        if dist.get_rank() == 0:
            LOG.info("NF4 staging control group ready; timeout=%s seconds", seconds)
        yield group
    finally:
        dist.destroy_process_group(group)


def nf4_loading_device():
    """Reuse Axolotl's distributed state without constructing a bare PartialState."""
    from axolotl.utils.distributed import get_distributed_state, init_distributed_state

    init_distributed_state()
    state = get_distributed_state()
    if state is None:
        raise RuntimeError("Could not initialize the NF4 loading device")
    return state.device


@contextmanager
def nf4_phase(name: str, interval: float = 60, enabled: bool = True):
    """Log phase boundaries and elapsed-time heartbeats on rank zero."""
    if not enabled or (dist.is_initialized() and dist.get_rank() != 0):
        yield
        return
    start = time.monotonic()
    done = threading.Event()
    base = last = (_progress["tensors"], _progress["bytes"])

    def heartbeat():
        nonlocal last
        while not done.wait(interval):
            current = (_progress["tensors"], _progress["bytes"])
            tensors = current[0] - base[0]
            if tensors:
                detail = (
                    f"{tensors} tensors / {(current[1] - base[1]) / 1024**3:.2f} GiB "
                    f"converted, +{current[0] - last[0]} since the last report"
                )
            else:
                detail = "no tensors converted yet"
            last = current
            LOG.info(
                "%s: still running after %.0fs (%s)",
                name,
                time.monotonic() - start,
                detail,
            )

    LOG.info("%s: starting", name)
    thread = threading.Thread(target=heartbeat, name="nf4-progress", daemon=True)
    thread.start()
    success = False
    try:
        yield
        success = True
    finally:
        done.set()
        thread.join()
        LOG.info(
            "%s: %s after %.1fs",
            name,
            "completed" if success else "failed",
            time.monotonic() - start,
        )
