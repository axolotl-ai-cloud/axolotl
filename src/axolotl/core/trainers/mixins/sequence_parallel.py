"""Module for Axolotl trainer sequence parallelism mixin"""

import logging
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from torch import nn
from torch.utils.data import DistributedSampler, Sampler

from axolotl.monkeypatch.attention.ring_attn import get_ring_attn_group

LOG = logging.getLogger(__name__)

try:
    from ring_flash_attn import update_ring_flash_attn_params
except ImportError:
    # We pass silently here, but raise an ImportError in our Axolotl config validation
    # if cfg.sequence_parallel_degree > 1 and `ring-flash-attn` is not installed.
    pass


class SequenceParallelMixin:
    """
    Mixin class for sequence parallelism support in trainers.

    This mixin provides functionality for handling sequence parallelism,
    including creating appropriate samplers, managing data partitioning,
    and updating ring flash attention parameters during training.
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

    def _update_ring_flash_attn_params(self, inputs: dict[str, torch.Tensor | Any]):
        """
        Calculate the cu_seqlens for the current forward pass and pass the value to
        the substituted ring_flash_attn. This is accomplished by using the passed
        `input_ids`.

        Args:
            inputs: Current batch of inputs.
        """
        # At this point, inputs should already be partitioned by the sequence
        # parallel data collator
        batch_size = inputs["input_ids"].shape[0]
        seq_len = inputs["input_ids"].shape[1]
        packed_seq_lens = [seq_len] * batch_size

        # Calculate the full sequence length across all GPUs in this SP group
        total_seq_len = seq_len * self.args.sequence_parallel_degree

        cu_seqlens = torch.cumsum(
            torch.tensor(
                packed_seq_lens, device=torch.cuda.current_device(), dtype=torch.int32
            ),
            dim=-1,
            dtype=torch.int32,
        )
        cu_seqlens = F.pad(
            F.pad(cu_seqlens, (1, 0), value=0), (0, 1), value=total_seq_len
        )

        update_ring_flash_attn_params(cu_seqlens, self.ring_attn_group)

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs. Overrides the
        `transformers.trainer.Trainer` method to handle sequence parallelism if
        enabled.

        Args:
            model: Model to perform training step for.
            inputs: Dictionary mapping.
        """
        # Set up sequence parallelism for this step if enabled
        if self.args.sequence_parallel_degree > 1:
            self._update_ring_flash_attn_params(inputs)

        # Proceed with normal training step
        return super().training_step(model, inputs, num_items_in_batch)  # type: ignore

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Perform a prediction step on a batch of inputs. Overrides the
        `transformers.trainer.Trainer` method to handle sequence parallelism if
        enabled.

        Args:
            model: Model to perform prediction step for.
            inputs: Dictionary mapping of inputs.
            prediction_loss_only: Whether to return only the loss.
            ignore_keys: Keys to ignore in the inputs.

        Returns:
            Tuple of (loss, logits, labels).
        """
        # Set up sequence parallelism for this prediction step if enabled
        if self.args.sequence_parallel_degree > 1:
            self._update_ring_flash_attn_params(inputs)

        # Proceed with normal prediction step
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)  # type: ignore
