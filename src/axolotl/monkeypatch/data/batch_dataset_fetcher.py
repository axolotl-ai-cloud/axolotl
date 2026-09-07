"""Monkey patches for the dataset fetcher to handle batches of packed indexes."""

import torch
from torch.utils.data._utils.fetch import _BaseDatasetFetcher
from torch.utils.data._utils.worker import _worker_loop

_ORIGINAL_MAP_DATASET_FETCHER = None
_ORIGINAL_WORKER_LOOP = None
_IS_PATCHED = False


class _MapDatasetFetcher(_BaseDatasetFetcher):
    """
    Custom dataset fetcher that handles nested batch structures from
    MultipackBatchSampler.
    """

    def fetch(self, possibly_batched_index):
        if isinstance(possibly_batched_index[0], list):
            # Handle nested structure from MultipackBatchSampler
            data = [None for i in possibly_batched_index]
            for i, possibly_batched_index_ in enumerate(possibly_batched_index):
                if self.auto_collation:
                    if (
                        hasattr(self.dataset, "__getitems__")
                        and self.dataset.__getitems__
                    ):
                        data[i] = self.dataset.__getitems__(possibly_batched_index_)
                    else:
                        data[i] = [self.dataset[idx] for idx in possibly_batched_index_]
                else:
                    data[i] = self.dataset[possibly_batched_index_]
        else:
            # Standard batch handling
            if self.auto_collation:
                if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                    data = self.dataset.__getitems__(possibly_batched_index)
                else:
                    data = [self.dataset[idx] for idx in possibly_batched_index]
            else:
                data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)


def patch_fetchers():
    """Apply patches to PyTorch's DataLoader components."""
    torch.utils.data._utils.fetch._MapDatasetFetcher = _MapDatasetFetcher
    torch.utils.data.dataloader._utils.fetch._MapDatasetFetcher = _MapDatasetFetcher


def patched_worker_loop(*args, **kwargs):
    """Worker loop that ensures patches are applied in worker processes."""
    patch_fetchers()
    return _worker_loop(*args, **kwargs)


def apply_multipack_dataloader_patch():
    """
    This patch allows DataLoader to correctly process batches that contain multiple bins
    of packed sequences.
    """
    # pylint: disable=global-statement
    global _ORIGINAL_MAP_DATASET_FETCHER, _ORIGINAL_WORKER_LOOP, _IS_PATCHED

    if _IS_PATCHED:
        return

    # Store original implementations
    _ORIGINAL_MAP_DATASET_FETCHER = torch.utils.data._utils.fetch._MapDatasetFetcher
    _ORIGINAL_WORKER_LOOP = torch.utils.data._utils.worker._worker_loop

    # Apply patches
    patch_fetchers()
    torch.utils.data._utils.worker._worker_loop = patched_worker_loop

    _IS_PATCHED = True


def remove_multipack_dataloader_patch():
    """Remove the monkeypatch and restore original PyTorch DataLoader behavior."""
    # pylint: disable=global-statement
    global _IS_PATCHED

    if not _IS_PATCHED:
        return

    if _ORIGINAL_MAP_DATASET_FETCHER:
        torch.utils.data._utils.fetch._MapDatasetFetcher = _ORIGINAL_MAP_DATASET_FETCHER
        torch.utils.data.dataloader._utils.fetch._MapDatasetFetcher = (
            _ORIGINAL_MAP_DATASET_FETCHER
        )

    if _ORIGINAL_WORKER_LOOP:
        torch.utils.data._utils.worker._worker_loop = _ORIGINAL_WORKER_LOOP

    _IS_PATCHED = False
