# pylint: skip-file
"""
Multipack Batch Sampler
"""
import logging
import math
from typing import Any, Iterable, List, Union

import numba
import numpy as np
from torch.utils.data import BatchSampler, Sampler, SequentialSampler

from axolotl.utils.distributed import reduce_and_broadcast

LOG = logging.getLogger(__name__)

LOG.setLevel(logging.INFO)


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

    return bins_result


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

    while True:
        # binary search [l, r)
        left = 1
        right = 1 + np.searchsorted(lengths_cumsum[start_index:], s + c * n, "right")

        while right - left > 1:
            mid = (left + right) // 2
            if ffd_check(lengths[start_index : start_index + mid], c, n):
                left = mid
            else:
                right = mid

        # use length l
        batch = ffd_with_result(
            lengths[start_index : start_index + left], c, start_index
        )
        assert len(batch) <= n
        if len(batch) < n:
            break

        start_index += left
        s = lengths_cumsum[start_index - 1]

        # add local rank
        result.append(batch[rank])

    return result, s, len(result) * c * n


@numba.njit
def allocate_sequentially(lengths: np.ndarray, rank: int, c: int, n: int):
    """
    Sequential allocator that preserves example order

    Parameters:
    - lengths: The lengths of all examples
    - rank: The current rank (for distributed training)
    - c: The capacity of each bin (maximum sequence length)
    - n: Number of ranks

    Returns:
    - result: List of batches for the current rank
    - total_used: Number of actual example tokens
    - total_slots: Maximum theoretical number of example tokens (number of bins * bin capacity)
    """
    result = []
    total_used = 0

    # First, do sequential packing into bins
    all_bins = []
    current_bin = [0 for i in range(0)]  # numba hint
    remaining_capacity = c

    for idx, size in enumerate(lengths):
        if size <= remaining_capacity:
            # Example fits in current bin
            current_bin.append(idx)
            remaining_capacity -= size
            total_used += size
        else:
            # Example doesn't fit, start a new bin
            if current_bin:  # Add non-empty bin to all_bins
                all_bins.append(current_bin)
            current_bin = [idx]
            remaining_capacity = c - size
            total_used += size

    # Add the last bin if not empty
    if current_bin:
        all_bins.append(current_bin)

    # Assign bins to ranks - each rank gets every n-th bin
    for bin_idx in range(rank, len(all_bins), n):
        result.append(all_bins[bin_idx])

    return result, total_used, len(all_bins) * c


class MultipackBatchSampler(BatchSampler):
    """Batch sampler class for multipack"""

    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,
        batch_max_len: int,
        lengths: np.ndarray,
        packing_efficiency_estimate: float = 1.0,
        drop_last: bool = False,
        num_count_samples: int = 16,
        sequential: bool = False,
        **kwargs,
    ):
        super().__init__(sampler, batch_size, drop_last)
        self.batch_size = batch_size
        self.batch_max_len = batch_max_len
        self.lengths: np.ndarray = lengths
        self.packing_efficiency_estimate = packing_efficiency_estimate or 1.0
        self.sequential = sequential

        assert isinstance(self.lengths, np.ndarray)

        self.epoch = 0

        # statistics
        self.eff_total_used = 0
        self.eff_total_slots = 0

        # The number of times to calculate the batches to determine the minimum packed dataset length for the local rank
        self.num_count_samples = num_count_samples
        # the minimum packed dataset length across all ranks determined by a gather/broadcast
        self.len_across_ranks = None

        if self.sequential and not isinstance(sampler, SequentialSampler):
            LOG.warn(
                "using sequential sample packing with non-sequential sampler, did you want to also enable curriculum_sampling?"
            )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def generate_batches(self, set_stats=False):
        indices = [idx for idx in self.sampler]

        lengths = self.lengths[indices]
        lengths_cumsum = np.cumsum(lengths)

        if self.sequential:
            batches, total_used, total_slots = allocate_sequentially(
                lengths=lengths,
                rank=0,
                c=self.batch_max_len,
                n=1,
            )
        else:
            batches, total_used, total_slots = allocate(
                lengths=lengths,
                lengths_cumsum=lengths_cumsum,
                rank=0,
                c=self.batch_max_len,
                n=1,
            )

        batches = [
            [
                [indices[b_idx] for b_idx in batch]
                for batch in batches[i : i + self.batch_size]
            ]
            for i in range(0, len(batches), self.batch_size)
        ]

        # statistics
        if set_stats:
            self.eff_total_used += total_used
            self.eff_total_slots += total_slots

        return batches

    def __iter__(self):
        batches = self.generate_batches(set_stats=True)
        if self.len_across_ranks:
            # make sure the batches we iterate over is truncated to the same min length across all ranks
            batches = batches[: self.len_across_ranks]
        return iter(batches)

    def num_batches(self):
        batches = self.generate_batches(set_stats=True)
        return len(batches)

    def efficiency(self):
        return self.eff_total_used / self.eff_total_slots

    def gather_efficiency(self):
        def calc_sample_packing_eff_est(estimates: List[float]):
            LOG.debug(f"sample_packing_eff_est across ranks: {repr(estimates)}")
            return math.floor(0.997 * max(estimates))

        sample_packing_actual_eff_all = reduce_and_broadcast(
            lambda: self.efficiency(),  # pylint: disable=unnecessary-lambda
            calc_sample_packing_eff_est,
        )
        sample_packing_eff_est = (
            math.ceil(sample_packing_actual_eff_all * 200.0) / 200.0
        )
        return sample_packing_eff_est

    def gather_len_batches(self, num):
        def calc_min_len(estimates: list[(int, float)]):
            LOG.info(f"gather_len_batches: {repr(estimates)}")
            return math.floor(min(estimates))

        min_len_batches = reduce_and_broadcast(lambda: num, calc_min_len)
        return min_len_batches

    def __len__(self):
        if not self.len_across_ranks:
            len_batches = min(
                [self.num_batches() for _ in range(self.num_count_samples)]
            )
            self.len_across_ranks = self.gather_len_batches(len_batches)
        return self.len_across_ranks
