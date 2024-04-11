"""
Multipack Batch Sampler
"""
import logging
import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import numba
import numpy as np
from torch.utils.data import BatchSampler

from axolotl.utils.distributed import get_rank, get_world_size, is_main_process

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

    for idx in range(len(sorted_items)):
        item = sorted_items[idx]
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
        self._batches = None

    def _pack_batches(self):
        # Initially, calculate packs for all ranks.
        pack_idxs = pack(
            self.lengths,
            self.batch_max_len,
            self.group_size,
            self.bin_size,
        )

        if is_main_process():
            used_tokens = self.lengths.sum()
            available_tokens = len(pack_idxs) * self.batch_max_len
            efficiency = used_tokens / available_tokens
            LOG.debug(f"Sample packing efficiency: {efficiency * 100:.2f}%")

        # Select pack indices for this rank.
        world_size = get_world_size()
        if self.drop_last:
            batches_per_rank = len(pack_idxs) // world_size
        else:
            batches_per_rank = math.ceil(len(pack_idxs) / world_size)

        start_idx = batches_per_rank * get_rank()
        end_idx = min(start_idx + batches_per_rank, len(pack_idxs))

        batch_idxs = pack_idxs[start_idx:end_idx]
        return batch_idxs

    def __iter__(self):
        self._batches = self._pack_batches()
        return iter(self._batches)

    def __len__(self):
        if self._batches is None:
            self._batches = self._pack_batches()
        return len(self._batches)
