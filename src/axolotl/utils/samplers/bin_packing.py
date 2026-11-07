"""Bin-packing kernels and the parallel packer behind MultipackBatchSampler.

Packing pool children unpickle ``_process_group`` from here, so this module must stay
off the torch/accelerate import path: numpy, numba and stdlib only.
"""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count, get_all_start_methods, get_context

import numba
import numpy as np

from axolotl.utils.forkserver import preload_forkserver_modules
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


@numba.njit
def ffd_check(sequence_lengths: np.ndarray, bin_capacity: int, num_bins: int) -> bool:
    """First-fit-decreasing bin packing algorithm check.

    Checks if sequences with the given lengths could fit in the specified number of
    bins.

    Args:
        sequence_lengths: Array of sequence lengths.
        bin_capacity: Maximum capacity of each bin.
        num_bins: Number of bins available.

    Returns:
        `True` if all sequences can be packed, `False` otherwise.
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
) -> list[list[int]]:
    """Pack a group of sequences into bins using First-Fit Decreasing algorithm.

    Args:
        sequence_lengths: Array of sequence lengths.
        group_offset: Offset to apply to indices when returning results.
        bin_capacity: Maximum capacity of each bin.
        max_bins: Maximum number of bins to use.
        bin_size: Maximum number of sequences per bin.
        safe_mode: If True, use a more conservative packing approach.

    Returns:
        List of bins, where each bin contains indices of sequences assigned to it.
    """
    bins_remaining_space: list = []  # Tracks remaining capacity in each bin
    bins_assigned_sequences: list = []  # Tracks sequence indices assigned to each bin

    for seq_id, size in enumerate(sequence_lengths):
        global_idx = seq_id + group_offset

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


def _process_group(
    args: tuple[np.ndarray, int, int, int, int, bool],
) -> list[list[int]]:
    """Standalone function for multiprocessing."""
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
    mp_start_method: str | None = None,
) -> list[list[int]]:
    """Pack sequences into bins using parallel processing.

    Args:
        sequence_lengths: Array of sequence lengths.
        bin_capacity: Maximum capacity of each bin as total number of tokens.
        group_size: Number of sequences to process in each group.
        bin_size: Maximum number of bins to use.
        num_processes: Number of parallel processes to use.
        safe_mode: If True, use a more conservative packing approach.
        mp_start_method: Multiprocessing start method ('fork', 'spawn', 'forkserver').
                         None uses forkserver where available, else the system
                         default.
    Returns:
        List of bins, where each bin contains indices of sequences assigned to it.
    """
    num_items = len(sequence_lengths)
    if num_processes is None:
        num_processes = max(1, min(num_items // group_size, cpu_count(), 16))

    # Create tasks for parallel processing
    tasks = []
    for i in range(0, num_items, group_size):
        group_lengths = sequence_lengths[i : i + group_size]
        max_bins = len(group_lengths)  # Allow as many bins as items in the group
        tasks.append((group_lengths, i, bin_capacity, max_bins, bin_size, safe_mode))
    # a single group gains nothing from a pool but would pay its process start
    num_processes = max(1, min(num_processes, len(tasks)))

    # Process groups in parallel
    all_bins = []

    mp_ctx = None
    if mp_start_method is None and "forkserver" in get_all_start_methods():
        # forking the multi-threaded trainer risks deadlocks, and spawn re-imports per pool
        mp_start_method = "forkserver"
    if mp_start_method:
        try:
            mp_ctx = get_context(mp_start_method)
        except ValueError:
            LOG.warning(
                f"Failed to get multiprocessing context '{mp_start_method}'. "
                f"Falling back to default. Available: {get_context().get_all_start_methods()}"
            )
            mp_ctx = (
                None  # Fallback to default context if specified one is not available
            )

    if num_processes == 1:
        LOG.debug("Using single process for pack_parallel, running sequentially.")
        for task_args in tasks:
            group_bins = _process_group(task_args)
            all_bins.extend(group_bins)
    else:
        # Use ProcessPoolExecutor only if num_processes > 1
        # Pass mp_context if available
        preload_forkserver_modules()
        with ProcessPoolExecutor(
            max_workers=num_processes, mp_context=mp_ctx
        ) as executor:
            for group_bins in executor.map(_process_group, tasks):
                all_bins.extend(group_bins)

    return all_bins
