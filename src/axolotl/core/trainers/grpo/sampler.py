"""
Repeat random sampler (akin to the one implemented in
https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py) that adds
sequence parallelism functionality; i.e., duplicating data across ranks in the same
sequencee parallel group.
"""

from typing import Optional, Sized

import torch
from torch.utils.data import Sampler


class SequenceParallelRepeatRandomSampler(Sampler):
    """
    Sampler for GRPO training with sequence parallelism that ensures:
    1. Ranks in the same sequence parallel group receive identical data
    2. Each index is repeated multiple times for sampling different completions
    3. Entire batches are repeated for reuse in multiple updates
    """

    def __init__(
        self,
        dataset: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        sequence_parallel_degree: int = 1,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
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
