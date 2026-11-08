"""Warm start for forkserver children (DataLoader workers and packing pools)."""

import multiprocessing

# Sample-packing pools always start from the forkserver; this needs only numpy and numba.
PACKING_PRELOAD = ("axolotl.utils.samplers.bin_packing",)

# What SFT DataLoader workers unpickle; without a preload each one re-imports these on
# every start (~30s per DataLoader iteration). Keep peft, bitsandbytes and torchao out:
# importing them initializes CUDA and starts threads in the forkserver.
DATALOADER_PRELOAD = (
    "torch",
    "transformers",
    "datasets",
    "axolotl.utils.collators",
    "axolotl.utils.worker_seed",
    "axolotl.monkeypatch.data.batch_dataset_fetcher",
)

_configured = False


def forkserver_preload_modules() -> list[str]:
    """Modules worth preloading given which children will start from the forkserver."""
    modules = ["__main__", *PACKING_PRELOAD]
    # get_start_method() without allow_none would fix the default for the process
    method = (
        multiprocessing.get_start_method(allow_none=True)
        or multiprocessing.get_all_start_methods()[0]
    )
    if method == "forkserver":
        # DataLoader workers use the default start method (Python 3.14+ on Linux)
        modules.extend(DATALOADER_PRELOAD)
    return modules


def preload_forkserver_modules() -> None:
    """Preload heavy modules into the forkserver; a no-op once it is running."""
    global _configured
    if _configured or "forkserver" not in multiprocessing.get_all_start_methods():
        return
    multiprocessing.set_forkserver_preload(forkserver_preload_modules())
    _configured = True
