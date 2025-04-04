"""Handler class for sequence parallel trainer logic"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DistributedSampler


class SequenceParallelHandler:
    """
    Handler class that encapsulates sequence parallelism functionality.
    This replaces the SequenceParallelMixin with a composition-based approach.
    """
    
    def __init__(self, args=None):
        """
        Initialize the sequence parallel handler.
        
        Args:
            args: The arguments object containing sequence parallelism settings.
        """
        self.args = args
        self.ring_attn_group = None
        
        # Set up sequence parallelism if enabled
        if self.args.sequence_parallel_degree > 1:
            self._setup_sequence_parallel()
    
    def _setup_sequence_parallel(self):
        """Set up sequence parallelism environment."""
        from ring_flash_attn import update_ring_flash_attn_params
        from axolotl.monkeypatch.attention.ring_attn import get_ring_attn_group

        self.update_ring_flash_attn_params = update_ring_flash_attn_params
        self.ring_attn_group = get_ring_attn_group()
    
    def create_sequence_parallel_sampler(
        self,
        dataset,
        shuffle=True,
        is_eval=False,
    ):
        """
        Helper method to create sampler for sequence parallelism (SP).
        
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
    
    def _get_train_sampler(self, dataset):
        """
        Get a training sampler configured for sequence parallelism.
        
        Args:
            dataset: The training dataset.
            
        Returns:
            Configured sequence parallel sampler.
        """
        return self.create_sequence_parallel_sampler(
            dataset,
            shuffle=not self.args.curriculum_sampling,
        )
    
    def _get_eval_sampler(self, eval_dataset):
        """
        Get an evaluation sampler configured for sequence parallelism.
        
        Args:
            eval_dataset: The evaluation dataset.
            
        Returns:
            Configured sequence parallel sampler.
        """
        return self.create_sequence_parallel_sampler(
            eval_dataset, shuffle=False, is_eval=True
        )
    
    def _update_ring_flash_attn_params(self, inputs):
        """
        Calculate the cu_seqlens for the current forward pass and pass the value to
        the substituted ring_flash_attn.
        
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
        
        self.update_ring_flash_attn_params(cu_seqlens, self.ring_attn_group)
