"""Module for Axolotl trainer sequence parallelism mixin"""

import functools
from axolotl.utils.logging import get_logger

import torch
import torch.distributed as dist
from datasets import Dataset
from torch.utils.data import DistributedSampler, Sampler
from axolotl.utils.schemas.enums import RingAttnFunc

from axolotl.monkeypatch.attention.ring_attn import (
    get_ring_attn_group,
)

LOG = get_logger(__name__)


def apply_sequence_parallelism(
    batch: dict[str, torch.Tensor],
    local_rank: int,
    local_world_size: int,
    ring_attn_func: RingAttnFunc,
) -> dict[str, torch.Tensor]:
    """
    Apply sequence parallelism slicing to a batch.

    Args:
        batch: Batch dictionary (e.g., input_ids, attention_mask, etc.)
        local_rank: Local rank in the sequence parallel group
        local_world_size: World size of the sequence parallel group
        ring_attn_func: The ring attention function to use

    Returns:
        Sliced batch dictionary.
    """
    # Update ring attention params if needed
    if batch.get("position_ids") is not None:
        update_ring_attn_params(position_ids=batch["position_ids"])

    # Slice batch for sequence parallel processing
    total_seq_len = batch["input_ids"].size(1)
    for key in batch:
        if (
            key in batch and
            isinstance(batch[key], torch.Tensor) and
            batch[key].dim() > 1 and
            batch[key].size(1) == total_seq_len
        ):

            if ring_attn_func in [
                RingAttnFunc.VARLEN_LLAMA3,
                RingAttnFunc.BATCH_RING,
            ]:
                # Split in sequential fashion and grab this rank's chunk
                batch[key] = (
                    batch[key].chunk(local_world_size, dim=1)[local_rank].contiguous()
                )
            elif ring_attn_func is RingAttnFunc.BATCH_ZIGZAG:
                chunks = batch[key].chunk(2 * local_world_size, dim=1)

                # Take rank's chunk and opposing chunk for zigzag pattern
                selected_chunks = [
                    chunks[local_rank],
                    chunks[2 * local_world_size - local_rank - 1],
                ]
                batch[key] = torch.cat(selected_chunks, dim=1).contiguous()
            elif ring_attn_func is RingAttnFunc.BATCH_STRIPE:
                # Split into striped data and stack
                tensor = torch.stack(
                    batch[key].split(local_world_size, dim=1),
                    dim=1,
                ).transpose(1, 2)
                batch[key] = tensor[:, local_rank].contiguous()

    return batch


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
