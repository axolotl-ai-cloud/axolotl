"""
Multipack Batch Sampler - An efficient batch sampler for packing variable-length sequences
into fixed-capacity batches to optimize memory usage and training throughput.
"""

import gc
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count, get_context
from typing import Iterable, Iterator, Union

import numba
import numpy as np
import torch
from torch.utils.data import BatchSampler, RandomSampler, Sampler, SequentialSampler

from axolotl.utils.distributed import reduce_and_broadcast
from axolotl.utils.logging import get_logger
from axolotl.utils.samplers.accumulation import (
    accumulation_metrics,
    balance_accumulation,
)
from axolotl.utils.samplers.label_balance import _batch_labels, balance_labels
from axolotl.utils.samplers.microbatch_balance import balance_microbatches
from axolotl.utils.samplers.rank_balance import order_batches_by_rank

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
    mp_start_method: str | None = "fork",
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
                         'spawn' is often safer with Numba/PyTorch.
                         Set to None to use system default.
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

    # Process groups in parallel
    all_bins = []

    mp_ctx = None
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
        with ProcessPoolExecutor(
            max_workers=num_processes, mp_context=mp_ctx
        ) as executor:
            for group_bins in executor.map(_process_group, tasks):
                all_bins.extend(group_bins)

    return all_bins


@numba.njit
def allocate_sequentially(
    sequence_lengths: np.ndarray, rank: int, bin_capacity: int, num_ranks: int
) -> tuple[list[list[int]], int, int]:
    """Sequential allocator that preserves example order.

    Args:
        sequence_lengths: The lengths of all examples.
        rank: The current rank (for distributed training).
        bin_capacity: The capacity of each bin (maximum sequence length).
        num_ranks: Number of ranks (processes / GPUs).

    Returns:
        rank_batches: List of batches for the current rank.
        total_tokens_used: Number of actual example tokens.
        total_token_slots: Maximum theoretical number of example tokens (number of bins
            * bin capacity).
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
    """Batch sampler class for efficient packing of variable-length sequences

    This sampler packs sequences into fixed-capacity bins (batches) to maximize
    GPU memory utilization and training throughput by reducing padding.

    It supports both parallel packing (using FFD algorithm) and
    sequential packing (preserving original sequence order).
    """

    _batches: list[list[list[int]]] | None = None
    _len_across_ranks: int | None = None

    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,  # Number of bins per batch
        batch_max_len: int,  # Maximum sequence length (bin capacity)
        lengths: np.ndarray,  # Sequence lengths
        bin_size: int,  # The max number of samples that can be packed in a single bin
        packing_efficiency_estimate: float = 1.0,  # Initial efficiency estimate
        drop_last: bool = True,  # Whether to drop final batches (might be incomplete)
        num_count_samples: int = 4,  # Number of times to estimate batch count
        sequential: bool = False,  # Whether to use sequential packing
        group_size: int = 100_000,  # Size of groups for parallel packing
        num_processes: int | None = None,  # Number of processes for parallel packing
        safe_mode: bool = True,  # Conservative packing to prevent training instability
        mp_start_method: str = "fork",
        label_counts: np.ndarray | None = None,
        label_start_counts: np.ndarray | None = None,
        seed: int = 0,
        padding_multiple: int | None = None,
        batches_per_optimizer_step: int = 1,
        dp_count: int = 1,
        label_balance_window_optim_steps: int = 1,
        **kwargs,
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
        self.mp_start_method = mp_start_method

        assert isinstance(self.lengths, np.ndarray)

        self.label_counts: np.ndarray | None = None
        self.label_start_counts: np.ndarray | None = None
        self.seed = seed
        self.padding_multiple = padding_multiple
        if batches_per_optimizer_step < 1:
            raise ValueError("batches_per_optimizer_step must be positive")
        self.batches_per_optimizer_step = batches_per_optimizer_step
        if dp_count < 1 or batches_per_optimizer_step % dp_count:
            raise ValueError("Optimizer-step width must be divisible by dp_count")
        self.dp_count = dp_count
        if (
            isinstance(label_balance_window_optim_steps, bool)
            or not isinstance(label_balance_window_optim_steps, (int, np.integer))
            or label_balance_window_optim_steps < 1
        ):
            raise ValueError(
                "label_balance_window_optim_steps must be a positive integer"
            )
        self.label_balance_window_optim_steps = label_balance_window_optim_steps
        if label_counts is not None:
            if sequential:
                raise ValueError(
                    "Label balancing is incompatible with sequential packing"
                )
            counts = np.asarray(label_counts)
            starts = (
                np.zeros_like(counts)
                if label_start_counts is None
                else np.asarray(label_start_counts)
            )
            if (
                counts.shape != self.lengths.shape
                or starts.shape != counts.shape
                or not np.issubdtype(counts.dtype, np.integer)
                or not np.issubdtype(starts.dtype, np.integer)
                or np.any(counts < 0)
                or np.any(counts > self.lengths)
                or np.any(starts < 0)
                or np.any(starts > 1)
                or np.any(starts > counts)
                or np.any(self.lengths <= 0)
                or np.any(self.lengths > batch_max_len)
            ):
                raise ValueError("Invalid lengths or label counts for label balancing")
            self.label_counts = counts.astype(np.int64)
            self.label_start_counts = starts.astype(np.int64)
        elif label_start_counts is not None:
            raise ValueError("label_start_counts requires label_counts")

        self.label_metrics: dict[str, dict[str, float | int]] | None = None
        self._label_metrics_logged = False
        self.epoch = 0

        # Efficiency statistics tracking
        self.total_tokens_used = 0
        self.total_token_slots = 0

        # The number of times to calculate batches to determine minimum packed dataset length
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.num_count_samples = (
            1 if world_size >= num_count_samples else num_count_samples
        )

        if self.sequential and not isinstance(sampler, SequentialSampler):
            LOG.warning(
                "using sequential sample packing with non-sequential sampler, did you want to also enable curriculum_sampling?"
            )

    def set_epoch(self, epoch: int):
        """Set the epoch number, used for reproducible shuffling across epochs"""
        self.epoch = epoch
        self._batches = None  # Invalidate batch cache
        self.label_metrics = None
        self._label_metrics_logged = False

    def generate_batches(self, set_stats: bool = False) -> list[list[list[int]]]:
        """Generate packed batches for training.

        Args:
            set_stats: Whether to update efficiency statistics and log label metrics.

        Returns:
            List of batches, where each batch contains multiple bins, and each bin
                contains multiple sequence indices.
        """
        if self._batches is not None:
            if set_stats:
                self._log_label_metrics()
            return self._batches

        # Get indices from the sampler
        base_sampler = self.sampler
        if self.label_counts is not None and isinstance(base_sampler, RandomSampler):
            # Packing happens before rank sharding, so rank-local RNG must not affect it.
            base_sampler = RandomSampler(
                base_sampler.data_source,
                replacement=base_sampler.replacement,
                num_samples=base_sampler.num_samples,
                generator=torch.Generator().manual_seed(self.seed + self.epoch),
            )
        indices = list(base_sampler)
        self.label_metrics = None
        self._label_metrics_logged = False

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
            num_processes = self.num_processes or 1
            all_bins = pack_parallel(
                lengths,
                bin_capacity=self.batch_max_len,
                group_size=self.group_size,
                bin_size=self.bin_size or self.batch_max_len,
                num_processes=min(4, num_processes) if num_processes else 4,
                safe_mode=self.safe_mode,
                mp_start_method=self.mp_start_method,
            )

            # Map bin indices back to original indices
            bins = [
                [indices[b_idx] for b_idx in bin_indices] for bin_indices in all_bins
            ]

            # Calculate efficiency statistics
            total_used = lengths.sum()
            total_slots = len(all_bins) * self.batch_max_len
            del all_bins

        # Group bins into batches (each batch contains batch_size bins)
        batches = [
            bins[i : i + self.batch_size] for i in range(0, len(bins), self.batch_size)
        ]

        # Drop last batch if requested and it's incomplete
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            dropped_batch = batches[-1]
            batches = batches[:-1]
            # Exclude dropped bins from efficiency statistics
            total_slots -= len(dropped_batch) * self.batch_max_len
            total_used -= sum(
                self.lengths[idx]
                for bin_indices in dropped_batch
                for idx in bin_indices
            )

        if self.label_counts is not None:
            assert self.label_start_counts is not None
            before = self._get_label_metrics(batches)
            limit = len(batches) - int(
                bool(batches) and len(batches[-1]) < self.batch_size
            )
            if self._len_across_ranks is not None:
                limit = min(limit, self._len_across_ranks)
            limit -= limit % self.batches_per_optimizer_step
            batches[:limit] = balance_labels(
                batches[:limit],
                self.lengths,
                self.label_counts,
                self.label_start_counts,
                self.batch_max_len,
                self.bin_size or self.batch_max_len,
                self.seed + self.epoch,
                padding_multiple=self.padding_multiple,
            )
            before_accumulation = self._get_label_metrics(batches)
            batches[:limit] = balance_accumulation(
                batches[:limit],
                [
                    _batch_labels(batch, self.label_counts, self.label_start_counts)
                    for batch in batches[:limit]
                ],
                self.batches_per_optimizer_step,
            )

            before_microbatch = self._get_label_metrics(batches)
            window = self.label_balance_window_optim_steps
            refinement_windows = (window, 1) if window > 1 else (1,)
            for refinement_window in refinement_windows:
                batches[:limit] = balance_microbatches(
                    batches[:limit],
                    self.lengths,
                    self.label_counts,
                    self.label_start_counts,
                    self.batches_per_optimizer_step,
                    window_steps=refinement_window,
                    padding_multiple=self.padding_multiple or 1,
                    capacity=self.batch_max_len,
                )
            before_rank = self._get_label_metrics(batches)
            if self.dp_count > 1 and limit:
                multiple = self.padding_multiple or 1
                costs = [
                    len(batch)
                    * (
                        (
                            max(sum(int(self.lengths[i]) for i in row) for row in batch)
                            + multiple
                            - 1
                        )
                        // multiple
                        * multiple
                    )
                    for batch in batches[:limit]
                ]
                order = order_batches_by_rank(
                    costs,
                    dp=self.dp_count,
                    gas=self.batches_per_optimizer_step // self.dp_count,
                )
                prefix = batches[:limit]
                batches[:limit] = [prefix[i] for i in order]
            self.label_metrics = {
                "before_rank": before_rank,
                "before_microbatch": before_microbatch,
                "before": before,
                "before_accumulation": before_accumulation,
                "after": self._get_label_metrics(batches),
            }

        # Update statistics if requested
        if set_stats:
            self.total_tokens_used += total_used
            self.total_token_slots += total_slots
            self._log_label_metrics()

        self._batches = batches
        gc.collect()
        return batches

    def _get_label_metrics(self, batches) -> dict[str, float | int]:
        assert self.label_counts is not None
        assert self.label_start_counts is not None
        labels = [
            sum(
                sum(int(self.label_counts[i]) for i in bin_)
                - int(self.label_start_counts[bin_[0]])
                for bin_ in batch
            )
            for batch in batches
        ]
        lengths = [
            sum(int(self.lengths[i]) for i in bin_)
            for batch in batches
            for bin_ in batch
        ]
        return {
            "batches": len(batches),
            "packed_rows": len(lengths),
            "mean_packed_length": float(np.mean(lengths)) if lengths else 0.0,
            "mean_label_count": float(np.mean(labels)) if labels else 0.0,
            "std_label_count": float(np.std(labels)) if labels else 0.0,
            "total_label_count": sum(labels),
            **accumulation_metrics(
                labels[
                    : len(batches)
                    - int(bool(batches) and len(batches[-1]) < self.batch_size)
                ],
                self.batches_per_optimizer_step,
            ),
        }

    def _log_label_metrics(self):
        if self.label_metrics is not None and not self._label_metrics_logged:
            LOG.info(
                "Packed label metrics (before rank sharding/truncation; labels per "
                "microbatch, lengths per unpadded row): %s",
                self.label_metrics,
            )
            self._label_metrics_logged = True

    def __iter__(self) -> Iterator[list[list[int]]]:
        """Return an iterator over batches.

        The batches are truncated to match the minimum number of batches across all
        ranks to ensure distributed training balance.
        """
        batches = self.generate_batches(set_stats=True)
        if self._len_across_ranks:
            # Truncate batches to ensure all ranks have the same number of batches
            batches = batches[: self._len_across_ranks]
        return iter(batches)

    def efficiency(self) -> float:
        """Calculate the packing efficiency (ratio of tokens used to total token slots).
        Higher is better - 1.0 would mean perfect packing with no wasted space.
        """
        if self.total_token_slots == 0:
            self.generate_batches(set_stats=True)
        if self.total_token_slots == 0:
            return 0.0
        # Return a Python float instead of potentially a numpy float
        return float(self.total_tokens_used / self.total_token_slots)

    def gather_efficiency(self) -> float:
        """Gather and synchronize packing efficiency estimates across all distributed
        ranks.

        Returns:
            A conservative efficiency estimate based on the measurements.
        """

        def calc_sample_packing_eff_est(estimates: list[float]):
            LOG.debug(f"sample_packing_eff_est across ranks: {repr(estimates)}")
            # Use 99.7% of max observed efficiency as a safe estimate
            max_eff = max(float(eff) for eff in estimates)
            return math.floor(0.997 * max_eff)

        # Gather efficiency from all ranks and apply the calculation function
        sample_packing_actual_eff_all = reduce_and_broadcast(
            lambda: float(self.efficiency()),
            calc_sample_packing_eff_est,
        )

        # Quantize to 0.5% intervals for stability
        sample_packing_eff_est = (
            math.ceil(sample_packing_actual_eff_all * 200.0) / 200.0
        )
        return sample_packing_eff_est

    def gather_len_batches(self, num: int) -> int:
        """Gather and synchronize batch counts across all distributed ranks. Returns
        the minimum number of batches available on any rank.
        """

        def calc_min_len(estimates: list[int]) -> int:
            LOG.info(f"gather_len_batches: {repr(estimates)}")
            return math.floor(min(estimates))

        # Find minimum batch count across ranks to ensure balance
        min_len_batches = reduce_and_broadcast(lambda: num, calc_min_len)
        return min_len_batches

    def __len__(self) -> int:
        """Return the total number of batches that will be yielded by this sampler.

        This is calculated as the minimum number of batches available on any rank to
        ensure balanced distributed training.
        """
        if self._batches is None:
            self._batches = self.generate_batches(set_stats=True)

        if (
            self._len_across_ranks is None
            and self.label_counts is not None
            and isinstance(self.sampler, (RandomSampler, SequentialSampler))
        ):
            self._len_across_ranks = self.gather_len_batches(len(self._batches))

        if self._len_across_ranks is None:
            # Sample multiple times to get stable estimate
            _sampled_lens = []
            for _ in range(self.num_count_samples):
                self._batches = None  # Reset cached batches
                # log timer for generating batches
                start_time = time.time()
                _sampled_lens.append(len(self.generate_batches(set_stats=False)))
                LOG.debug(f"generate_batches time: {time.time() - start_time}")
            len_batches = min(_sampled_lens)

            # Gather minimum across all ranks
            if self._len_across_ranks is None:
                self._len_across_ranks = self.gather_len_batches(len_batches)
            else:
                self._len_across_ranks = min(
                    self._len_across_ranks, self.gather_len_batches(len_batches)
                )

        return self._len_across_ranks
