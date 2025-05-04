# pylint: skip-file
"""
Multipack Batch Sampler
"""
import logging
import math
from typing import Iterable, List, Union

import numba
import numpy as np
from torch.utils.data import BatchSampler, Sampler, SequentialSampler

from axolotl.utils.distributed import reduce_and_broadcast

LOG = logging.getLogger(__name__)

LOG.setLevel(logging.INFO)


@numba.njit
def ffd_check(sequence_lengths: np.ndarray, bin_capacity: int, num_bins: int):
    # First-fit-decreasing bin packing algorithm
    # Checks if sequences with lengths in sequence_lengths[] could fit in num_bins bins, each with capacity bin_capacity
    # Returns True if all sequences can be packed, False otherwise
    # https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing

    # Sort sequence lengths in descending order for optimal packing
    sequence_lengths = np.sort(sequence_lengths)[::-1]
    # Initialize all bins with full capacity
    bins = np.full((num_bins,), bin_capacity, dtype=sequence_lengths.dtype)

    # Try to place each sequence in the first bin it fits
    for size in sequence_lengths:
        not_found = True
        for idx in range(num_bins):
            if bins[idx] >= size:
                bins[idx] -= size
                not_found = False
                break

        # If no bin could fit this sequence, packing failed
        if not_found:
            return False

    return True


@numba.njit
def ffd_with_result(sequence_lengths: np.ndarray, bin_capacity: int, start_index: int):
    # First-fit-decreasing bin packing that returns the actual bin assignments
    # Returns a list of bins, where each bin contains indices of sequences assigned to it

    # Get sorting indices and sort sequence lengths in descending order
    indices = np.argsort(sequence_lengths)[::-1]
    sequence_lengths = sequence_lengths[indices]

    bins_remaining_space: list = []  # Tracks remaining capacity in each bin
    bins_assigned_sequences: list = []  # Tracks sequence indices assigned to each bin

    # Place each sequence in the first bin it fits
    for seq_id, size in enumerate(sequence_lengths):
        add_new_bin = True
        for bin_idx in range(len(bins_remaining_space)):
            if bins_remaining_space[bin_idx] >= size:
                bins_remaining_space[bin_idx] -= size
                bins_assigned_sequences[bin_idx].append(indices[seq_id] + start_index)
                add_new_bin = False
                break

        # If no existing bin could fit this sequence, create a new bin
        if add_new_bin:
            bins_remaining_space.append(bin_capacity - size)
            bins_assigned_sequences.append([indices[seq_id] + start_index])

    return bins_assigned_sequences


@numba.njit
def allocate(
    sequence_lengths: np.ndarray,
    lengths_cumsum: np.ndarray,
    rank: int,
    bin_capacity: int,
    num_ranks: int,
):
    # Dynamic batch allocator, similar to Multifit algorithm
    # Efficiently packs sequences into fixed-capacity bins for distributed training
    # https://en.wikipedia.org/wiki/Multifit_algorithm
    # ~99.5% efficiency on OpenChat training set (12 * 2048 ctx len)

    total_processed_tokens = 0
    start_index = 0
    rank_batches = []  # Batches assigned to the current rank

    while True:
        # Binary search to find maximum number of sequences that can be packed into num_ranks bins
        # [left, right) defines the search range
        left = 1
        right = 1 + np.searchsorted(
            lengths_cumsum[start_index:],
            total_processed_tokens + bin_capacity * num_ranks,
            "right",
        )

        while right - left > 1:
            mid = (left + right) // 2
            if ffd_check(
                sequence_lengths[start_index : start_index + mid],
                bin_capacity,
                num_ranks,
            ):
                left = mid
            else:
                right = mid

        # Pack the identified sequences into bins
        all_rank_batches = ffd_with_result(
            sequence_lengths[start_index : start_index + left],
            bin_capacity,
            start_index,
        )
        assert len(all_rank_batches) <= num_ranks

        # If we couldn't fill all ranks, we're done
        if len(all_rank_batches) < num_ranks:
            break

        # Update indices and processed token count
        start_index += left
        total_processed_tokens = lengths_cumsum[start_index - 1]

        # Add the batch for the current rank
        rank_batches.append(all_rank_batches[rank])

    # Return batches for this rank, total tokens used, and total token slots available
    return (
        rank_batches,
        total_processed_tokens,
        len(rank_batches) * bin_capacity * num_ranks,
    )


@numba.njit
def allocate_sequentially(
    sequence_lengths: np.ndarray, rank: int, bin_capacity: int, num_ranks: int
):
    """
    Sequential allocator that preserves example order (no sorting by length)

    Arguments:
    - sequence_lengths: The lengths of all examples
    - rank: The current rank (for distributed training)
    - bin_capacity: The capacity of each bin (maximum sequence length)
    - num_ranks: Number of ranks (processes/GPUs)

    Returns:
    - rank_batches: List of batches for the current rank
    - total_tokens_used: Number of actual example tokens
    - total_token_slots: Maximum theoretical number of example tokens (number of bins * bin capacity)
    """
    rank_batches = []
    total_tokens_used = 0

    # First, do sequential packing into bins
    all_bins = []
    current_bin = [0 for i in range(0)]  # numba hint for empty list of integers
    remaining_capacity = bin_capacity

    # Process each sequence in order
    for idx, size in enumerate(sequence_lengths):
        if size <= remaining_capacity:
            # Example fits in current bin
            current_bin.append(idx)
            remaining_capacity -= size
            total_tokens_used += size
        else:
            # Example doesn't fit, start a new bin
            if current_bin:  # Add non-empty bin to all_bins
                all_bins.append(current_bin)
            current_bin = [idx]
            remaining_capacity = bin_capacity - size
            total_tokens_used += size

    # Add the last bin if not empty
    if current_bin:
        all_bins.append(current_bin)

    # Assign bins to ranks - each rank gets every num_ranks-th bin
    for bin_idx in range(rank, len(all_bins), num_ranks):
        rank_batches.append(all_bins[bin_idx])

    return rank_batches, total_tokens_used, len(all_bins) * bin_capacity


class MultipackBatchSampler(BatchSampler):
    """
    Batch sampler class for efficient packing of variable-length sequences.

    This sampler packs sequences into fixed-capacity bins (batches) to maximize
    GPU memory utilization and training throughput by reducing padding.

    It supports both length-optimized packing (using FFD algorithm) and
    sequential packing (preserving original sequence order).
    """

    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,  # Number of bins per batch
        batch_max_len: int,  # Maximum sequence length (bin capacity)
        lengths: np.ndarray,  # Sequence lengths
        packing_efficiency_estimate: float = 1.0,  # Initial efficiency estimate
        drop_last: bool = False,  # Whether to drop incomplete batches
        num_count_samples: int = 16,  # Number of samples to estimate batch count
        sequential: bool = False,  # Whether to use sequential packing instead of FFD
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

        # Efficiency statistics tracking
        self.eff_total_used = 0  # Total tokens used
        self.eff_total_slots = 0  # Total token slots available

        # The number of times to calculate batches to determine minimum packed dataset length
        self.num_count_samples = num_count_samples
        # Minimum packed dataset length across all ranks (determined by gather/broadcast)
        self.len_across_ranks = None

        if self.sequential and not isinstance(sampler, SequentialSampler):
            LOG.warning(
                "using sequential sample packing with non-sequential sampler, did you want to also enable curriculum_sampling?"
            )

    def set_epoch(self, epoch: int):
        """Set the epoch number, used for reproducible shuffling across epochs"""
        self.epoch = epoch

    def generate_batches(self, set_stats=False):
        """
        Generate packed batches for training

        Args:
            set_stats: Whether to update efficiency statistics

        Returns:
            List of batches, where each batch contains multiple bins,
            and each bin contains multiple sequence indices
        """
        # Get indices from the sampler
        indices = [idx for idx in self.sampler]

        # Get lengths of the selected sequences
        lengths = self.lengths[indices]
        lengths_cumsum = np.cumsum(lengths)

        # Pack sequences into bins using either sequential or FFD allocation
        if self.sequential:
            bins, total_used, total_slots = allocate_sequentially(
                lengths=lengths,
                rank=0,
                bin_capacity=self.batch_max_len,
                num_ranks=1,
            )
        else:
            bins, total_used, total_slots = allocate(
                lengths=lengths,
                lengths_cumsum=lengths_cumsum,
                rank=0,
                bin_capacity=self.batch_max_len,
                num_ranks=1,
            )

        # Group bins into batches (each batch contains batch_size bins)
        batches = [
            [
                [indices[b_idx] for b_idx in bin_indices]
                for bin_indices in bins[i : i + self.batch_size]
            ]
            for i in range(0, len(bins), self.batch_size)
        ]

        # Update statistics if requested
        if set_stats:
            self.eff_total_used += total_used
            self.eff_total_slots += total_slots

        return batches

    def __iter__(self):
        """
        Return an iterator over batches

        The batches are truncated to match the minimum number of batches across all ranks
        to ensure distributed training balance
        """
        batches = self.generate_batches(set_stats=True)
        if self.len_across_ranks:
            # Truncate batches to ensure all ranks have the same number of batches
            batches = batches[: self.len_across_ranks]
        return iter(batches)

    def num_batches(self):
        """Calculate the number of batches for this rank"""
        batches = self.generate_batches(set_stats=True)
        return len(batches)

    def efficiency(self):
        """
        Calculate the packing efficiency (ratio of tokens used to total token slots)
        Higher is better - 1.0 would mean perfect packing with no wasted space
        """
        return self.eff_total_used / self.eff_total_slots

    def gather_efficiency(self):
        """
        Gather and synchronize packing efficiency estimates across all distributed ranks
        Returns a conservative efficiency estimate based on the measurements
        """

        def calc_sample_packing_eff_est(estimates: List[float]):
            LOG.debug(f"sample_packing_eff_est across ranks: {repr(estimates)}")
            # Use 99.7% of max observed efficiency as a safe estimate
            return math.floor(0.997 * max(estimates))

        # Gather efficiency from all ranks and apply the calculation function
        sample_packing_actual_eff_all = reduce_and_broadcast(
            lambda: self.efficiency(),  # pylint: disable=unnecessary-lambda
            calc_sample_packing_eff_est,
        )

        # Quantize to 0.5% intervals for stability
        sample_packing_eff_est = (
            math.ceil(sample_packing_actual_eff_all * 200.0) / 200.0
        )
        return sample_packing_eff_est

    def gather_len_batches(self, num):
        """
        Gather and synchronize batch counts across all distributed ranks
        Returns the minimum number of batches available on any rank
        """

        def calc_min_len(estimates: list[(int, float)]):
            LOG.info(f"gather_len_batches: {repr(estimates)}")
            return math.floor(min(estimates))

        # Find minimum batch count across ranks to ensure balance
        min_len_batches = reduce_and_broadcast(lambda: num, calc_min_len)
        return min_len_batches

    def __len__(self):
        """
        Return the total number of batches that will be yielded by this sampler

        This is calculated as the minimum number of batches available on any rank
        to ensure balanced distributed training
        """
        if not self.len_across_ranks:
            # Sample multiple times to get stable estimate
            len_batches = min(
                [self.num_batches() for _ in range(self.num_count_samples)]
            )
            # Gather minimum across all ranks
            self.len_across_ranks = self.gather_len_batches(len_batches)
        return self.len_across_ranks
