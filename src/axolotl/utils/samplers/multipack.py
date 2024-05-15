"""
Multipack Batch Sampler
"""
import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import numba
import numpy as np
from torch.utils.data import BatchSampler

LOG = logging.getLogger("axolotl.utils.samplers.multipack")


# First-fit-decreasing bin packing.
@numba.njit
def pack_group(items, group_offset, bin_capacity, max_items_per_bin):
    idxs = np.argsort(items)[::-1]
    sorted_items = items[idxs]
    num_bins = len(items)
    bins = np.full(num_bins, bin_capacity, dtype=np.int32)
    bin_counts = np.zeros(num_bins, dtype=np.int32)
    group_packing = np.full((num_bins, max_items_per_bin), -1, dtype=np.int32)

    for idx, item in enumerate(sorted_items):
        global_idx = idxs[idx] + group_offset

        placed = False
        for i in range(num_bins):
            if bins[i] >= item and bin_counts[i] < max_items_per_bin:
                bins[i] -= item
                group_packing[i, bin_counts[i]] = global_idx
                bin_counts[i] += 1
                placed = True
                break

        if not placed:
            raise ValueError(
                f"Item could not be packed. Try increasing cfg.sample_packing_bin_size ({max_items_per_bin})."
            )

    return group_packing


def pack(items, bin_capacity, group_size, max_items_per_bin):
    num_items = len(items)
    num_processes = max(1, min(num_items // group_size, cpu_count()))
    tasks = [
        (items[i : i + group_size], i, bin_capacity, max_items_per_bin)
        for i in range(0, num_items, group_size)
    ]

    packed_bins = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for group_packing in executor.map(pack_group, *zip(*tasks)):
            for bin_pack in group_packing:
                filtered_pack = bin_pack[bin_pack != -1]
                if filtered_pack.size > 0:
                    packed_bins.append(filtered_pack.tolist())

    return packed_bins


class MultipackBatchSampler(BatchSampler):
    """
    Batch Sampler class for multipack
    """

    def __init__(
        self,
        sampler,
        lengths,
        batch_max_len,
        batch_size,
        group_size,
        bin_size,
        drop_last=False,
    ):
        self.sampler = sampler
        self.lengths = np.array(lengths, dtype=np.int32)
        self.batch_max_len = batch_max_len
        self.batch_size = batch_size
        self.group_size = group_size
        self.bin_size = bin_size
        self.drop_last = drop_last

        self._efficiency = None
        self._batches = None

    def efficiency(self):
        if self._efficiency is None:
            self._batches = self._pack_batches()
        return self._efficiency

    def _pack_batches(self):
        # Get possibly shuffled indices from sampler.
        sample_idxs = np.arange(len(self.sampler))
        lengths = self.lengths[sample_idxs]

        pack_idxs = pack(
            lengths,
            self.batch_max_len,
            self.group_size,
            self.bin_size,
        )

        used_tokens = self.lengths.sum()
        available_tokens = len(pack_idxs) * self.batch_max_len
        self._efficiency = used_tokens / available_tokens

        # Wrap packs into batches.
        batch_idxs = [
            pack_idxs[i : i + self.batch_size]
            for i in range(0, len(pack_idxs), self.batch_size)
        ]

        # Drop last batch if needed.
        if self.drop_last and len(batch_idxs[-1]) < self.batch_size:
            batch_idxs = batch_idxs[:-1]

        return batch_idxs

    def __iter__(self):
        self._batches = self._pack_batches()
        return iter(self._batches)

    def __len__(self):
        if self._batches is None:
            self._batches = self._pack_batches()
        return len(self._batches)
