"""
Multipack Batch Sampler - An efficient batch sampler for packing variable-length sequences
into fixed-capacity batches to optimize memory usage and training throughput.
"""

import logging
import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from typing import Iterable, Union

import numba
import numpy as np
from torch.utils.data import BatchSampler, Sampler, SequentialSampler

from axolotl.utils.distributed import reduce_and_broadcast

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


@numba.njit
def ffd_check(sequence_lengths: np.ndarray, bin_capacity: int, num_bins: int):
    """
    First-fit-decreasing bin packing algorithm check

    Checks if sequences with the given lengths could fit in the specified number of bins

    Args:
        sequence_lengths: Array of sequence lengths
        bin_capacity: Maximum capacity of each bin
        num_bins: Number of bins available

    Returns:
        True if all sequences can be packed, False otherwise
    """
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
def pack_group(
    sequence_lengths: np.ndarray,
    group_offset: int,
    bin_capacity: int,
    max_bins: int,
    bin_size: int,
    safe_mode: bool = True,
):
    """
    Pack a group of sequences into bins using First-Fit Decreasing algorithm

    Args:
        sequence_lengths: Array of sequence lengths
        group_offset: Offset to apply to indices when returning results
        bin_capacity: Maximum capacity of each bin
        max_bins: Maximum number of bins to use
        bin_size: Maximum number of sequences per bin
        safe_mode: If True, use a more conservative packing approach

    Returns:
        List of bins, where each bin contains indices of sequences assigned to it
    """
    # Get sorting indices and sort lengths in descending order
    indices = np.argsort(sequence_lengths)[::-1]
    sorted_lengths = sequence_lengths[indices]

    bins_remaining_space: list = []  # Tracks remaining capacity in each bin
    bins_assigned_sequences: list = []  # Tracks sequence indices assigned to each bin

    for seq_id, size in enumerate(sorted_lengths):
        global_idx = indices[seq_id] + group_offset

        # Try to place sequence in existing bins
        add_new_bin = True
        for bin_idx, _ in enumerate(bins_remaining_space):
            if (
                bins_remaining_space[bin_idx] >= size
                and len(bins_assigned_sequences[bin_idx]) < bin_size
            ):
                bins_remaining_space[bin_idx] -= size
                bins_assigned_sequences[bin_idx].append(global_idx)
                add_new_bin = False
                break

        # Create a new bin if needed and if we haven't reached the limit
        if add_new_bin:
            if len(bins_remaining_space) >= max_bins and safe_mode:
                # In safe mode, skip items that would exceed max_bins
                continue
            bins_remaining_space.append(bin_capacity - size)
            bins_assigned_sequences.append([global_idx])

            # Safety check to avoid infinite bins
            if len(bins_remaining_space) > len(sequence_lengths):
                break

    return bins_assigned_sequences


# Define a standalone function for multiprocessing
def _process_group(args):
    group_lengths, start_idx, bin_capacity, max_bins, bin_size, safe_mode = args
    return pack_group(
        group_lengths, start_idx, bin_capacity, max_bins, bin_size, safe_mode
    )


def pack_parallel(
    sequence_lengths: np.ndarray,
    bin_capacity: int,
    group_size: int,
    bin_size: int,
    num_processes: int | None = None,
    safe_mode: bool = True,
):
    """
    Pack sequences into bins using parallel processing

    Args:
        sequence_lengths: Array of sequence lengths
        bin_capacity: Maximum capacity of each bin as total number of tokens
        group_size: Number of sequences to process in each group
        bin_size: Maximum number of bins to use
        num_processes: Number of parallel processes to use
        safe_mode: If True, use a more conservative packing approach

    Returns:
        List of bins, where each bin contains indices of sequences assigned to it
    """
    num_items = len(sequence_lengths)
    if num_processes is None:
        num_processes = max(1, min(num_items // group_size, cpu_count()))

    # Create tasks for parallel processing
    tasks = []
    for i in range(0, num_items, group_size):
        group_lengths = sequence_lengths[i : i + group_size]
        max_bins = len(group_lengths)  # Allow as many bins as items in the group
        tasks.append((group_lengths, i, bin_capacity, max_bins, bin_size, safe_mode))

    # Process groups in parallel
    all_bins = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for group_bins in executor.map(_process_group, tasks):
            all_bins.extend(group_bins)

    return all_bins


@numba.njit
def allocate_sequentially(
    sequence_lengths: np.ndarray, rank: int, bin_capacity: int, num_ranks: int
):
    """
    Sequential allocator that preserves example order

    Args:
        sequence_lengths: The lengths of all examples
        rank: The current rank (for distributed training)
        bin_capacity: The capacity of each bin (maximum sequence length)
        num_ranks: Number of ranks (processes/GPUs)

    Returns:
        rank_batches: List of batches for the current rank
        total_tokens_used: Number of actual example tokens
        total_token_slots: Maximum theoretical number of example tokens (number of bins * bin capacity)
    """
    result = []
    total_used = 0

    # First, do sequential packing into bins
    all_bins = []
    current_bin = [0 for i in range(0)]  # numba hint
    remaining_capacity = bin_capacity

    for idx, size in enumerate(sequence_lengths):
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
            remaining_capacity = bin_capacity - size
            total_used += size

    # Add the last bin if not empty
    if current_bin:
        all_bins.append(current_bin)

    # Assign bins to ranks - each rank gets every n-th bin
    for bin_idx in range(rank, len(all_bins), num_ranks):
        result.append(all_bins[bin_idx])

    return result, total_used, len(all_bins) * bin_capacity


class MultipackBatchSampler(BatchSampler):
    """
    Batch sampler class for efficient packing of variable-length sequences

    This sampler packs sequences into fixed-capacity bins (batches) to maximize
    GPU memory utilization and training throughput by reducing padding.

    It supports both parallel packing (using FFD algorithm) and
    sequential packing (preserving original sequence order).
    """

    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,  # Number of bins per batch
        batch_max_len: int,  # Maximum sequence length (bin capacity)
        lengths: np.ndarray,  # Sequence lengths
        packing_efficiency_estimate: float = 1.0,  # Initial efficiency estimate
        drop_last: bool = False,  # Whether to drop final batches (might be incomplete)
        num_count_samples: int = 16,  # Number of times to estimate batch count
        sequential: bool = False,  # Whether to use sequential packing
        group_size: int = 100_000,  # Size of groups for parallel packing
        bin_size: int = 200,  # The max number of samples that can be packed in a single bin
        num_processes: int | None = None,  # Number of processes for parallel packing
        safe_mode: bool = True,  # Conservative packing to prevent training instability
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(sampler, batch_size, drop_last)
        self.batch_size = batch_size
        self.batch_max_len = batch_max_len
        self.lengths = np.array(lengths, dtype=np.int32)
        self.packing_efficiency_estimate = packing_efficiency_estimate or 1.0
        self.sequential = sequential
        self.group_size = group_size
        self.bin_size = bin_size
        self.num_processes = num_processes
        self.safe_mode = safe_mode

        assert isinstance(self.lengths, np.ndarray)

        self.epoch = 0

        # Efficiency statistics tracking
        self.total_tokens_used = 0
        self.total_token_slots = 0

        # The number of times to calculate batches to determine minimum packed dataset length
        self.num_count_samples = num_count_samples
        # Minimum packed dataset length across all ranks (determined by gather/broadcast)
        self.len_across_ranks = None

        # Cache for batches
        self._batches = None

        if self.sequential and not isinstance(sampler, SequentialSampler):
            LOG.warning(
                "using sequential sample packing with non-sequential sampler, did you want to also enable curriculum_sampling?"
            )

    def set_epoch(self, epoch: int):
        """Set the epoch number, used for reproducible shuffling across epochs"""
        self.epoch = epoch
        self._batches = None  # Invalidate batch cache

    def generate_batches(self, set_stats=False):
        """
        Generate packed batches for training

        Args:
            set_stats: Whether to update efficiency statistics

        Returns:
            List of batches, where each batch contains multiple bins,
            and each bin contains multiple sequence indices
        """
        if self._batches is not None:
            return self._batches

        # Get indices from the sampler
        indices = [  # pylint: disable=unnecessary-comprehension
            idx for idx in self.sampler
        ]

        # Get lengths of the selected sequences
        lengths = self.lengths[indices]

        # Pack sequences into bins using either sequential or parallel packing
        if self.sequential:
            bins, total_used, total_slots = allocate_sequentially(
                lengths,
                rank=0,
                bin_capacity=self.batch_max_len,
                num_ranks=1,
            )
            # Map bin indices back to original indices
            bins = [[indices[b_idx] for b_idx in bin_indices] for bin_indices in bins]
        else:
            # Use parallel packing
            all_bins = pack_parallel(
                lengths,
                bin_capacity=self.batch_max_len,
                group_size=self.group_size,
                bin_size=self.bin_size,
                num_processes=self.num_processes,
                safe_mode=self.safe_mode,
            )

            # Map bin indices back to original indices
            bins = [
                [indices[b_idx] for b_idx in bin_indices] for bin_indices in all_bins
            ]

            # Calculate efficiency statistics
            total_used = lengths.sum()
            total_slots = len(all_bins) * self.batch_max_len

        # Group bins into batches (each batch contains batch_size bins)
        batches = [
            bins[i : i + self.batch_size] for i in range(0, len(bins), self.batch_size)
        ]

        # Drop last batch if requested and it's incomplete
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
            # Adjust total_slots if we dropped a batch
            if not self.sequential:
                total_slots -= (self.batch_size - len(batches[-1])) * self.batch_max_len

        # Update statistics if requested
        if set_stats:
            self.total_tokens_used += total_used
            self.total_token_slots += total_slots

        self._batches = batches
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

    def efficiency(self):
        """
        Calculate the packing efficiency (ratio of tokens used to total token slots)
        Higher is better - 1.0 would mean perfect packing with no wasted space
        """
        if self.total_token_slots == 0:
            self.generate_batches(set_stats=True)
        if self.total_token_slots == 0:
            return 0.0
        # Return a Python float instead of potentially a numpy float
        return float(self.total_tokens_used / self.total_token_slots)

    def gather_efficiency(self):
        """
        Gather and synchronize packing efficiency estimates across all distributed ranks
        Returns a conservative efficiency estimate based on the measurements
        """

        def calc_sample_packing_eff_est(estimates: list[float]):
            LOG.debug(f"sample_packing_eff_est across ranks: {repr(estimates)}")
            # Use 99.7% of max observed efficiency as a safe estimate
            max_eff = max(float(eff) for eff in estimates)
            return math.floor(0.997 * max_eff)

        # Gather efficiency from all ranks and apply the calculation function
        sample_packing_actual_eff_all = reduce_and_broadcast(
            lambda: float(self.efficiency()),  # pylint: disable=unnecessary-lambda
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
        if self._batches is None:
            self._batches = self.generate_batches(set_stats=True)

        if self.len_across_ranks is None:
            # Sample multiple times to get stable estimate
            len_batches = min(  # pylint: disable=consider-using-generator
                [len(self._batches) for _ in range(self.num_count_samples)]
            )
            # Gather minimum across all ranks
            self.len_across_ranks = self.gather_len_batches(len_batches)

        return self.len_across_ranks
