"""Warm start for forkserver children (DataLoader workers and packing pools)."""

import multiprocessing

# Without a preload every forkserver child re-imports these on each start, which is
# ~30s per DataLoader iteration on Python 3.14 (its default start method on Linux).
# The bitsandbytes import behind them initializes CUDA in the forkserver, so its
# children cannot use CUDA, the same as fork children of the trainer.
FORKSERVER_PRELOAD = (
    "__main__",
    "torch",
    "transformers",
    "transformers.trainer_utils",
    "datasets",
    "axolotl.utils.collators",
    "axolotl.utils.samplers.multipack",
    "axolotl.monkeypatch.data.batch_dataset_fetcher",
)

_configured = False


def preload_forkserver_modules() -> None:
    """Preload heavy modules into the forkserver; a no-op once it is running."""
    global _configured
    if _configured or "forkserver" not in multiprocessing.get_all_start_methods():
        return
    multiprocessing.set_forkserver_preload(list(FORKSERVER_PRELOAD))
    _configured = True
