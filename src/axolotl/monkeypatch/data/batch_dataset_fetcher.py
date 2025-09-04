"""monkey patches for the dataset fetcher to handle batches of packed indexes"""

# pylint: disable=protected-access

import torch
from torch.utils.data._utils.fetch import _BaseDatasetFetcher
from torch.utils.data._utils.worker import _worker_loop


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if isinstance(possibly_batched_index[0], list):
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
            if self.auto_collation:
                if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                    data = self.dataset.__getitems__(possibly_batched_index)
                else:
                    data = [self.dataset[idx] for idx in possibly_batched_index]
            else:
                data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)


def patch_fetchers():
    torch.utils.data._utils.fetch._MapDatasetFetcher = _MapDatasetFetcher
    torch.utils.data.dataloader._utils.fetch._MapDatasetFetcher = _MapDatasetFetcher


def patched_worker_loop(*args, **kwargs):
    patch_fetchers()
    return _worker_loop(*args, **kwargs)


torch.utils.data._utils.worker._worker_loop = patched_worker_loop
patch_fetchers()
