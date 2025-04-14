"""Module for Axolotl trainer sequence parallelism mixin"""

import logging
from contextlib import contextmanager

import torch
import torch.distributed as dist
from datasets import Dataset
from torch.utils.data import DistributedSampler, Sampler

from axolotl.monkeypatch.attention.ring_attn import (
    get_ring_attn_group, update_ring_attn_params
)

LOG = logging.getLogger(__name__)


class SequenceParallelMixin:
    """
    Mixin class for sequence parallelism support in trainers.

    This mixin provides functionality for handling sequence parallelism,
    specifically for creating appropriate data samplers.
    """

    args = None  # type: "AxolotlTrainingArguments"  # type: ignore[name-defined]

    def _setup_sequence_parallel(self):
        """Set up sequence parallelism environment."""
        self.ring_attn_group = get_ring_attn_group()

    def _create_sequence_parallel_sampler(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        is_eval: bool = False,
    ) -> DistributedSampler:
        """
        Helper method to create sampler for sequence parallelism (SP).

        We create a distributed sampler with rank equal to the SP group ID, which
        means that all ranks in the SP group receive the same sample / set of samples
        per training step. We also set the number of replicas equal to the number of
        SP groups, which is a bit of a hack / unintended use, but works!

        Args:
            dataset: Dataset to sample from.
            shuffle: Whether to shuffle the dataset.
            is_eval: Whether we are creating a sampler for evaluation or training.

        Returns:
            Distributed sampler.
        """
        num_sp_groups = self.args.world_size // self.args.sequence_parallel_degree
        sp_group_id = dist.get_rank() // self.args.sequence_parallel_degree

        return DistributedSampler(
            dataset,
            num_replicas=num_sp_groups,
            rank=sp_group_id,
            seed=self.args.seed if shuffle else None,
            shuffle=shuffle,
            drop_last=not is_eval,
        )

    def _sp_get_train_sampler(self, dataset) -> Sampler | None:
        """
        Get a training sampler configured for sequence parallelism.

        Args:
            dataset: The training dataset

        Returns:
            Configured sequence parallel sampler.
        """
        return self._create_sequence_parallel_sampler(
            dataset,
            shuffle=not self.args.curriculum_sampling,
        )

    def _sp_get_eval_sampler(self, eval_dataset) -> Sampler | None:
        """
        Get an evaluation sampler configured for sequence parallelism.

        Args:
            eval_dataset: The evaluation dataset.

        Returns:
            Configured sequence parallel sampler.
        """
        return self._create_sequence_parallel_sampler(
            eval_dataset, shuffle=False, is_eval=True
        )


class SequenceParallelismManager:
    def __init__(self, local_rank, local_world_size):
        self.local_rank = local_rank
        self.local_world_size = local_world_size
        
    @contextmanager
    def apply(self, batch):
        """
        Context manager that applies sequence parallelism slicing to a batch,
        and restores the original batch afterward if needed.
        
        Args:
            batch: Batch dictionary from parent collator.
            
        Yields:
            Sliced batch dictionary for use in the model.
        """
        # Get local (start, end) for sequence parallelism slicing
        total_seq_len = batch["input_ids"].size(1)
        slice_size = total_seq_len // self.local_world_size
        start = self.local_rank * slice_size
        end = start + slice_size
        
        # Update params for varlen ring attention calculation
        if batch.get("position_ids") is not None:
            update_ring_attn_params(
                input_ids=batch["input_ids"], position_ids=batch["position_ids"]
            )

        # Slice batch for sequence parallel processing
        for key in batch:
            if isinstance(batch[key], torch.Tensor) and batch[key].size(1) == total_seq_len:
                batch[key] = batch[key][:, start:end]

        yield batch
