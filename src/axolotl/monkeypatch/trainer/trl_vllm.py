"""Monkeypatches for TRL's vLLM integration and trainer utils.

Adds:
- VLLMClient.batch_update_named_params: chunked weight sync over vLLM's native NCCL
  weight-transfer engine, inside a single weight update, with lazy communicator init
- extract_logprobs: NaN→0.0 fix (prevents downstream NaN propagation)
- split_tensor_dict / shuffle_sequence_dict: scalar type handling (int/float/bool passthrough)
"""

import math

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _batch_update_named_params(
    self, params: list[tuple[str, torch.Tensor]], chunk_size: int | None = None
):
    """Stream params over vLLM's NCCL weight-transfer engine in one weight update.

    The communicator is initialised on first use when trainer init skipped it.
    Generation is paused (in-flight requests kept) for the update, since vLLM
    unloads layers while the update is open.
    """
    if not params:
        return

    if getattr(self, "communicator", None) is None:
        self.init_communicator(device=params[0][1].device)

    if chunk_size is None:
        chunks = [params]
    else:
        chunks = []
        current_chunk: list[tuple[str, torch.Tensor]] = []
        current_elements = 0
        for name, weights in params:
            n_elem = weights.numel()
            if current_chunk and current_elements + n_elem > chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
                current_elements = 0
            current_chunk.append((name, weights))
            current_elements += n_elem
        if current_chunk:
            chunks.append(current_chunk)

    self._post(f"{self.base_url}/pause", params={"mode": "keep"})
    try:
        with self.weight_update():
            for chunk in chunks:
                metadata = [
                    (
                        name,
                        str(weights.dtype).removeprefix("torch."),
                        list(weights.shape),
                    )
                    for name, weights in chunk
                ]
                self.update_named_params(metadata, iter(chunk))
    finally:
        self._post(f"{self.base_url}/resume")


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
    import trl.generation.vllm_client
    import trl.generation.vllm_generation
    import trl.trainer.utils

    VLLMClient = trl.generation.vllm_client.VLLMClient

    # 1. Add batch_update_named_params to VLLMClient
    if not hasattr(VLLMClient, "batch_update_named_params"):
        VLLMClient.batch_update_named_params = _batch_update_named_params
        LOG.info("Patched VLLMClient with batch_update_named_params")

    # 2. Patch extract_logprobs (NaN→0.0)
    trl.generation.vllm_generation.extract_logprobs = _patched_extract_logprobs
    LOG.info("Patched extract_logprobs with NaN→0.0 fix")

    # 3. Patch split_tensor_dict and shuffle_sequence_dict
    trl.trainer.utils.split_tensor_dict = _patched_split_tensor_dict
    trl.trainer.utils.shuffle_sequence_dict = _patched_shuffle_sequence_dict
    LOG.info("Patched split_tensor_dict and shuffle_sequence_dict for scalar types")
