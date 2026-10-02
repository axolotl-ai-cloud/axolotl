"""Monkeypatches for TRL's vLLM integration and trainer utils.

Adds:
- extract_logprobs: NaN→0.0 fix (prevents downstream NaN propagation)
- split_tensor_dict / shuffle_sequence_dict: scalar type handling (int/float/bool passthrough)
"""

import math

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _patched_extract_logprobs(all_outputs):
    """extract_logprobs with NaN→0.0 fix (stock TRL uses None which causes downstream errors)."""
    all_logprobs = []
    all_token_ids = []

    for outputs in all_outputs:
        for output in outputs.outputs:
            if output.logprobs is None:
                return None, None
            seq_logprobs = []
            seq_token_ids = []
            for lp in output.logprobs:
                sorted_items = sorted(lp.items(), key=lambda x: x[1].rank)
                seq_token_ids.append([token_id for token_id, _ in sorted_items])
                seq_logprobs.append(
                    [
                        0.0 if math.isnan(item.logprob) else item.logprob
                        for _, item in sorted_items
                    ]
                )
            all_logprobs.append(seq_logprobs)
            all_token_ids.append(seq_token_ids)

    return all_logprobs, all_token_ids


def _patched_split_tensor_dict(tensor_dict, num_chunks):
    """split_tensor_dict that handles scalar types (int/float/bool) for num_items_in_batch."""
    first_tensor = next(
        tensor
        for tensor in tensor_dict.values()
        if tensor is not None and isinstance(tensor, torch.Tensor) and tensor.ndim > 0
    )
    chunk_size = first_tensor.shape[0] // num_chunks
    chunks = []
    for i in range(num_chunks):
        chunk_dict = {}
        for key, tensor in tensor_dict.items():
            if isinstance(tensor, (int, float, bool)):
                chunk_dict[key] = tensor
            elif tensor is not None and (isinstance(tensor, list) or tensor.ndim > 0):
                chunk_dict[key] = tensor[i * chunk_size : (i + 1) * chunk_size]
            elif tensor is not None and tensor.ndim == 0:
                chunk_dict[key] = tensor
            else:
                chunk_dict[key] = None
        chunks.append(chunk_dict)
    return chunks


def _patched_shuffle_sequence_dict(seq_dict):
    """shuffle_sequence_dict that handles scalar types (int/float/bool)."""
    first_seq = next(
        v
        for v in seq_dict.values()
        if v is not None and isinstance(v, (torch.Tensor, list)) and len(v) > 0
    )
    perm = torch.randperm(len(first_seq))

    def permute(v):
        if v is None:
            return None
        if isinstance(v, (int, float, bool)):
            return v
        if isinstance(v, torch.Tensor) and v.ndim == 0:
            return v
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            return v[perm]
        if isinstance(v, list):
            return [v[i] for i in perm.tolist()]
        return v

    return {k: permute(v) for k, v in seq_dict.items()}


def patch_trl_vllm():
    """Apply all TRL vLLM monkeypatches."""
    import trl.generation.vllm_generation
    import trl.trainer.utils

    trl.generation.vllm_generation.extract_logprobs = _patched_extract_logprobs
    LOG.info("Patched extract_logprobs with NaN→0.0 fix")

    # Patch split_tensor_dict and shuffle_sequence_dict
    trl.trainer.utils.split_tensor_dict = _patched_split_tensor_dict
    trl.trainer.utils.shuffle_sequence_dict = _patched_shuffle_sequence_dict
    LOG.info("Patched split_tensor_dict and shuffle_sequence_dict for scalar types")
