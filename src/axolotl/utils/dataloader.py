# pylint: skip-file
import itertools
import logging
import math
import os
from typing import Any, Callable, List, Union

import numba
import numpy as np
from torch.utils.data import DistributedSampler, Sampler

LOG = logging.getLogger("axolotl.utils.dataloader")


@numba.njit
def ffd_check(a: np.ndarray, c: int, n: int):
    # First-fit-decreasing bin packing
    # Check if a[] could fit in n bins with capacity c
    # https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing

    a = np.sort(a)[::-1]
    bins = np.full((n,), c, dtype=a.dtype)
    for size in a:
        not_found = True
        for idx in range(n):
            if bins[idx] >= size:
                bins[idx] -= size
                not_found = False
                break

        if not_found:
            return False

    return True


@numba.njit
def ffd_with_result(a: np.ndarray, c: int, start_index: int):
    # First-fit-decreasing bin packing (with result return)

    indices = np.argsort(a)[::-1]
    a = a[indices]

    bins: List[Any] = []
    bins_result: List[Any] = []
    for a_id, size in enumerate(a):
        add_new = True
        for idx in range(len(bins)):
            if bins[idx] >= size:
                bins[idx] -= size
                bins_result[idx].append(indices[a_id] + start_index)
                add_new = False
                break

        if add_new:
            bins.append(c - size)
            bins_result.append([indices[a_id] + start_index])

    return bins_result, len(a)


@numba.njit
def allocate(
    lengths: np.ndarray, lengths_cumsum: np.ndarray, rank: int, c: int, n: int
):
    # Dynamic batch allocator, similar to Multifit
    # https://en.wikipedia.org/wiki/Multifit_algorithm
    # ~99.5% efficiency on OpenChat training set (12 * 2048 ctx len)

    s = 0
    start_index = 0
    result = []
    result_totseqs = []

    while True:
        # binary search [left, right)
        left = 1
        right = 1 + np.searchsorted(lengths_cumsum[start_index:], s + c * n, "right")

        while right - left > 1:
            mid = (left + right) // 2
            if ffd_check(lengths[start_index : start_index + mid], c, n):
                left = mid
            else:
                right = mid

        # use length left
        batch, tot_seqs = ffd_with_result(
            lengths[start_index : start_index + left], c, start_index
        )
        if len(batch) < n:
            break

        start_index += left
        s = lengths_cumsum[start_index - 1]

        # add local rank
        result.append(batch[rank])
        # add total seqs for all ranks
        result_totseqs.append(tot_seqs)

    return result, result_totseqs, s, len(result) * c * n


def chunk(iterable, n):
    """
    Chunk data into tuples of length n
    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


class MultipackDistributedDataloader:
    """Unpadded data loading using Multipack.
    Adapted from https://github.com/imoneoi/openchat/blob/v3_fix_mle_loss/ochat/training_deepspeed/multipack_dataloader.py
    Approximate (at most ~1.22x) the optimal solution of the identical-machines scheduling problem, which is NP-hard.
    """

    def __init__(
        self,
        dataset: Any,
        collate_fn: Callable,
        seq_max_length: int = 2048,
        batch_size: int = 1,
        sampler: Union[Sampler, DistributedSampler] = None,
        packing_efficiency_estimate: float = 1.0,
        seq_len_multiple: int = 1,
    ):
        # Dataset
        self.dataset = dataset
        self.lengths: np.ndarray = np.array(
            [len(sample["input_ids"]) for sample in self.dataset]
        )
        assert isinstance(self.lengths, np.ndarray)
        assert batch_size % seq_len_multiple == 0
        self.sampler = sampler
        self.batch_size = batch_size
        self.seq_len_multiple = seq_len_multiple
        self.seq_max_length = seq_max_length
        self.batch_max_length = batch_size * seq_max_length
        self.collate_fn = collate_fn

        self.num_replicas = 1
        self.rank = 0

        # statistics
        self.eff_total_used = 0
        self.eff_total_slots = 0
        self.packing_efficiency_estimate = packing_efficiency_estimate or 1.0

    def generate_batches(self, set_stats=False):
        if self.sampler:
            indices = [idx for idx in self.sampler]
        else:
            indices = range(0, len(self.dataset))

        lengths = self.lengths[indices]
        lengths_cumsum = np.cumsum(lengths)

        batches, totseqs, total_used, total_slots = allocate(
            lengths=lengths,
            lengths_cumsum=lengths_cumsum,
            rank=self.rank,
            # c=self.batch_max_length,
            c=self.seq_max_length * self.seq_len_multiple,
            n=self.num_replicas,
        )

        batches = [[indices[b_idx] for b_idx in batch] for batch in batches]

        # statistics
        if set_stats:
            self.eff_total_used += total_used
            self.eff_total_slots += total_slots

        return batches, totseqs

    def __iter__(self):
        all_batches, _ = self.generate_batches(set_stats=True)
        features = self.dataset.features.keys()
        len_remaining = self._len_est()
        for batches in chunk(all_batches, self.batch_size / self.seq_len_multiple):
            chunked_data = []
            attn_mask_cum_idx = 0
            for batch in batches:
                concatenated = {}
                batched_data = [self.dataset[batch_idx] for batch_idx in batch]
                for feature in features:
                    if feature == "attention_mask":
                        arrays = [
                            (attn_mask_cum_idx + idx + 1) * np.array(item[feature])
                            for idx, item in enumerate(batched_data)
                            if feature in item
                        ]
                        attn_mask_cum_idx += len(batched_data)
                        concatenated[feature] = np.concatenate(arrays)
                    else:
                        arrays = [
                            np.array(item[feature])
                            for item in batched_data
                            if feature in item
                        ]
                        concatenated[feature] = np.concatenate(arrays)
                chunked_data.append(concatenated)
                # num_chunks = int(
                #     np.ceil(len(next(iter(concatenated.values()))) / self.seq_max_length)
                # )
                # chunked_data = []
                #
                # for i in range(num_chunks):
                #     chunk = {
                #         feature: array[
                #             i * self.seq_max_length : (i + 1) * self.seq_max_length
                #         ]
                #         for feature, array in concatenated.items()
                #     }
                #     chunked_data.append(chunk)
                # yield self.collate_fn(chunked_data)
            yield self.collate_fn(chunked_data)
            len_remaining -= 1
            if not len_remaining:
                return

    def _len_est(self):
        indices = range(0, len(self.dataset))
        lengths = self.lengths[indices]
        lengths_sum = np.cumsum(lengths)[-1]
        lengths_sum_per_device = lengths_sum // int(os.environ.get("WORLD_SIZE", 1))
        LOG.info(
            f"packing_efficiency_estimate: {self.packing_efficiency_estimate} "
            f"total_num_tokens per device: {lengths_sum_per_device}"
        )

        # shave off 1% + 1 for dealing with variance in packing from random sampler to sampler
        return (
            math.floor(
                0.99
                * lengths_sum_per_device
                / self.packing_efficiency_estimate
                / self.seq_max_length
                // self.batch_size
            )
            - 1
        )

    def __len__(self):
        # this doesn't return the actual length b/c with distributed samplers, not all dataloaders get
        # the same share of total tokens
        if not self.eff_total_used:
            batches, _ = self.generate_batches(set_stats=True)
        LOG.info(
            f"packing_efficiency_estimate: {self.packing_efficiency_estimate} "
            f"actual packing efficiency: {self.efficiency()}"
        )
        return self._len_est()

    def efficiency(self):
        return self.eff_total_used / self.eff_total_slots
