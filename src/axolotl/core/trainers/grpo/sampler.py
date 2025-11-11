"""Repeat random sampler (similar to the one implemented in
https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py) that adds
sequence parallelism functionality; i.e., duplicating data across ranks in the same
sequence parallel group.
"""

from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class SequenceParallelRepeatRandomSampler(Sampler):
    """Sampler for GRPO training with sequence parallelism.

    This sampler ensures:
    - Ranks in the same sequence parallel (SP) group receive identical data.
    - Each index is repeated multiple times for sampling different completions.
    - Entire batches are repeated for reuse in multiple updates.
    - Data is properly distributed across SP groups.

    In the table below, the values represent dataset indices. Each SP group has
    `context_parallel_size = 2` GPUs working together on the same data. There are 2
    SP groups (SP0 and SP1), with `world_size = 4` total GPUs.

                                               Sequence Parallel Groups
                                        |       SP0        |       SP1        |
                                        |  GPU 0  |  GPU 1 |  GPU 2  |  GPU 3 |
                    global_step  step    <---> mini_repeat_count=3
                                            <----------> batch_size=2 per SP group
    grad_accum=2   ▲  ▲  0       0         [0 0 0  1 1 1]     [2 2 2  3 3 3]   <- SP groups get different data
                   ▼  |  0       1         [0 0 0  1 1 1]     [2 2 2  3 3 3]   <- Same data for each SP group GPU
                      |
                      |  1       2         [0 0 0  1 1 1]     [2 2 2  3 3 3]   <- Repeat same indices for iterations
    num_iterations=2  ▼  1       3         [0 0 0  1 1 1]     [2 2 2  3 3 3]   <- When using gradient accumulation

                         2       4         [4 4 4  5 5 5]     [6 6 6  7 7 7]   <- New batch of data indices
                         2       5         [4 4 4  5 5 5]     [6 6 6  7 7 7]
                                            ...

    Args:
        dataset: Dataset to sample from.
        mini_repeat_count: How many times to repeat each sample immediately.
        world_size: Total number of processes.
        rank: Rank of current process.
        batch_size: Number of samples per batch.
        repeat_count: How many times to repeat the full sampling process.
        context_parallel_size: Number of ranks in a sequence parallel group.
        shuffle: Whether to shuffle the dataset.
        seed: Random seed for shuffling.
        drop_last: Whether to drop the last incomplete batch.
    """

    def __init__(
        self,
        dataset: Sized,
        mini_repeat_count: int,
        world_size: int,
        rank: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        context_parallel_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.world_size = world_size
        self.rank = rank

        # Sequence parallelism parameters
        self.context_parallel_size = context_parallel_size
        self.num_sp_groups = world_size // context_parallel_size
        self.sp_group_id = rank // context_parallel_size

        # Adjust dataset size for distributed sampling
        self.num_samples = len(self.dataset)
        self.total_size = self.num_samples

        # Calculate effective number of samples per SP group
        if (
            self.drop_last
            and self.total_size % (self.num_sp_groups * self.batch_size) != 0
        ):
            # Drop last incomplete batch if drop_last is True
            self.num_samples_per_sp_group = (
                self.total_size // self.batch_size // self.num_sp_groups
            ) * self.batch_size
        else:
            # Round up to include last batch if drop_last is False
            self.num_samples_per_sp_group = (
                (self.total_size + self.batch_size * self.num_sp_groups - 1)
                // (self.batch_size * self.num_sp_groups)
                * self.batch_size
            )

        if shuffle:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

    def __iter__(self) -> Iterator[int]:
        """Creates iterator over dataset indices.

        Returns:
            Iterator that yields indices into the dataset.
        """
        # Deterministically shuffle based on epoch and seed
        if self.shuffle:
            indices = torch.randperm(
                self.num_samples, generator=self.generator
            ).tolist()
        else:
            indices = list(range(self.num_samples))

        # Add extra samples to make it evenly divisible by batch_size
        if len(indices) % self.batch_size != 0:
            padding = indices[: self.batch_size - len(indices) % self.batch_size]
            indices += padding

        # Subsample based on SP group ID
        # Each SP group gets distinct batches of data
        batch_indices = []
        for i in range(0, len(indices), self.batch_size * self.num_sp_groups):
            start_idx = i + self.sp_group_id * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(indices))
            if start_idx < len(indices):
                for j in range(self.batch_size):
                    if start_idx + j < end_idx:
                        batch_indices.append(indices[start_idx + j])

        # Make sure batch_indices is exactly batch_size * num_batches_per_sp_group
        if self.drop_last:
            num_batches_per_sp_group = self.num_samples_per_sp_group // self.batch_size
            target_len = self.batch_size * num_batches_per_sp_group
            if len(batch_indices) > target_len:
                batch_indices = batch_indices[:target_len]

        # Apply the GRPO repeat pattern
        final_indices = []
        for _ in range(self.repeat_count):
            for idx in batch_indices:
                for _ in range(self.mini_repeat_count):
                    final_indices.append(idx)

        return iter(final_indices)

    def __len__(self) -> int:
        """Returns the total length of the iterable including repetitions.

        Returns:
            Total number of samples.
        """
        # Total length including all repetitions
        return (
            self.num_samples_per_sp_group * self.mini_repeat_count * self.repeat_count
        )

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        Args:
            epoch: Epoch number to use for shuffling.
        """
        self.epoch = epoch
