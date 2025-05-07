"""
Repeat random sampler (similar to the one implemented in
https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py) that adds
sequence parallelism functionality; i.e., duplicating data across ranks in the same
sequencee parallel group.
"""

from typing import Sized

import torch
from torch.utils.data import Sampler


class SequenceParallelRepeatRandomSampler(Sampler):
    """
    SequenceParallelRepeatRandomSampler for GRPO training with sequence parallelism.

    This sampler ensures:
    - Ranks in the same sequence parallel (SP) group receive identical data.
    - Each index is repeated multiple times for sampling different completions.
    - Entire batches are repeated for reuse in multiple updates.
    - Data is properly distributed across SP groups.

    In the figure below, the values represent dataset indices. Each SP group has
    `sequence_parallel_degree = 2` GPUs working together on the same data. There are 2
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

    Key behaviors:
    1. Each GPU in the same SP group (e.g., GPU 0 and GPU 1) gets identical data.
    2. Different SP groups (e.g., SP0 vs SP1) get different data slices.
    3. Each index is repeated mini_repeat_count times consecutively.
    4. The entire pattern repeats repeat_count times for multiple updates.

    The total samples processed = num_sp_groups * batch_size * mini_repeat_count
    * repeat_count where num_sp_groups = world_size / sequence_parallel_degree.
    """

    def __init__(
        self,
        dataset: Sized,
        mini_repeat_count: int,
        world_size: int,
        rank: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        sequence_parallel_degree: int = 1,
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
        self.sequence_parallel_degree = sequence_parallel_degree
        self.num_sp_groups = world_size // sequence_parallel_degree
        self.sp_group_id = rank // sequence_parallel_degree

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

    def __iter__(self):
        # Deterministically shuffle based on epoch and seed
        if self.shuffle:
            # Use same seed for all ranks in the same SP group
            g = torch.Generator()
            seed_value = self.seed + self.epoch + self.sp_group_id * 10000
            g.manual_seed(seed_value)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

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

    def __len__(self):
        # Total length including all repetitions
        return (
            self.num_samples_per_sp_group * self.mini_repeat_count * self.repeat_count
        )

    def set_epoch(self, epoch):
        """Sets the epoch for this sampler"""
        self.epoch = epoch
