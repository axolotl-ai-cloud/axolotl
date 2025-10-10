"""Monkeypatch for FA utils to accept 1D position_ids from Pixtral's position_ids_in_meshgrid"""

import torch


def apply_patch_is_packed_sequence():
    """Apply patch to FA utils to accept 1D position_ids from Pixtral's position_ids_in_meshgrid"""
    from transformers import modeling_flash_attention_utils

    def fixed_is_packed_sequence(position_ids, batch_size):
        """
        Check the position ids whether packed sequences are indicated or not
            1. Position ids exist
            2. Flattened sequences only are supported
            3. Compile-friendly `not (torch.diff(position_ids, dim=-1) >= 0).all()`, i.e. we have multiple increasing sequences
        """
        if position_ids is None:
            return False

        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)  # [N] -> [1, N]

        increasing_position_sequences = (
            torch.arange(position_ids.shape[1], device=position_ids.device)
            + position_ids.min()
        )
        return (
            batch_size == 1
            and (increasing_position_sequences - position_ids).abs().sum().bool().item()
        )

    # Store original method
    old_fn = modeling_flash_attention_utils._is_packed_sequence

    # Apply the patch
    modeling_flash_attention_utils._is_packed_sequence = fixed_is_packed_sequence

    def unpatch():
        """Restore the original method"""
        modeling_flash_attention_utils._is_packed_sequence = old_fn

    return unpatch
