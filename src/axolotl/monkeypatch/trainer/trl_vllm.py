"""Monkeypatches for TRL's vLLM integration and trainer utils.

Adds:
- VLLMClient.batch_update_named_params: batched weight sync (fewer HTTP round-trips)
- extract_logprobs: NaN→0.0 fix (prevents downstream NaN propagation)
- VLLMGeneration: weight_sync_chunk_size + batched sync path for non-FSDP/non-ZeRO
- split_tensor_dict / shuffle_sequence_dict: scalar type handling (int/float/bool passthrough)
"""

import logging
import math
from functools import wraps

import torch
from torch import nn

LOG = logging.getLogger(__name__)


def _batch_update_named_params(
    self, params: list[tuple[str, torch.Tensor]], chunk_size: int | None = None
):
    """Batched weight sync — uses NCCL if communicator available, HTTP otherwise."""
    has_communicator = getattr(self, "communicator", None) is not None

    if has_communicator:
        # Fast path: metadata via HTTP, tensors via NCCL
        from transformers import is_torch_xpu_available

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

        for chunk in chunks:
            param_metadata = [
                {
                    "name": name,
                    "dtype": str(weights.dtype),
                    "shape": list(weights.shape),
                }
                for name, weights in chunk
            ]
            url = f"{self.base_url}/batch_update_named_params/"
            response = self.session.post(
                url, json={"params": param_metadata}, timeout=120
            )
            if response.status_code != 200:
                raise Exception(
                    f"Request failed: {response.status_code}, {response.text}"
                )

            for _name, weights in chunk:
                if is_torch_xpu_available():
                    self.communicator.broadcast(weights, root=self.rank)
                else:
                    self.communicator.broadcast(weights, src=self.rank)

            if is_torch_xpu_available():
                self.communicator.barrier()
            else:
                self.communicator.group.barrier()
    else:
        # HTTP-only path: encode tensor data in request body (no NCCL needed).
        # Batch by byte size to avoid huge HTTP payloads.
        MAX_BYTES_PER_REQUEST = 10 * 1024 * 1024  # 10 MB
        HTTP_TIMEOUT = 120  # seconds per request

        payload: list[dict] = []
        payload_bytes = 0
        url = f"{self.base_url}/http_update_weights/"

        def _flush(p: list[dict]) -> None:
            if not p:
                return
            response = self.session.post(url, json={"params": p}, timeout=HTTP_TIMEOUT)
            if response.status_code != 200:
                raise Exception(
                    f"Request failed: {response.status_code}, {response.text}"
                )

        from axolotl.utils.weight_serde import encode_for_http

        for name, weights in params:
            entry = encode_for_http(name, weights)
            entry_bytes = weights.nelement() * weights.element_size()

            # Flush current batch if adding this entry would exceed limit
            if payload and payload_bytes + entry_bytes > MAX_BYTES_PER_REQUEST:
                _flush(payload)
                payload = []
                payload_bytes = 0

            payload.append(entry)
            payload_bytes += entry_bytes

        _flush(payload)  # send remaining


def _update_model_params(self, model: nn.Module, chunk_size: int | None = None):
    """Updates all model params using batch_update_named_params."""
    params = [(name, param.data) for name, param in model.named_parameters()]
    self.batch_update_named_params(params, chunk_size=chunk_size)


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


def _patch_sync_weights_batched(original_init):
    """Wrap VLLMGeneration.__init__ to accept weight_sync_chunk_size."""

    @wraps(original_init)
    def patched_init(self, *args, weight_sync_chunk_size=None, **kwargs):
        original_init(self, *args, **kwargs)
        self.weight_sync_chunk_size = weight_sync_chunk_size

    return patched_init


def _make_batched_sync_weights(original_sync_weights):
    """Wrap sync_weights to use batched sync for non-FSDP/non-ZeRO paths."""

    @wraps(original_sync_weights)
    def patched_sync_weights(self):
        from accelerate.utils import is_peft_model

        # Check if we're in a non-PEFT, non-FSDP, non-ZeRO scenario where batching helps
        accelerator = self.accelerator
        model = self.model
        is_fsdp_enabled = self.is_fsdp_enabled

        deepspeed_plugin = accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        is_peft = is_peft_model(model)

        # If PEFT, FSDP, or ZeRO-3, fall back to original (which handles those cases)
        if is_peft or is_fsdp_enabled or zero_stage_3:
            return original_sync_weights(self)

        # Non-PEFT, non-FSDP, non-ZeRO: use batched sync
        if self.mode == "colocate" and getattr(self, "enable_sleep_mode", False):
            from vllm.distributed.device_communicators.cuda_wrapper import (
                empty_cache,
            )

            empty_cache()
            self.llm.wake_up(tags=["weights"])

        if self.mode == "server" and accelerator.is_main_process:
            params = [
                (self._fix_param_name_to_vllm(name), param.data)
                for name, param in model.named_parameters()
            ]
            self.vllm_client.batch_update_named_params(
                params, chunk_size=getattr(self, "weight_sync_chunk_size", None)
            )
        elif self.mode == "colocate":
            llm_model = (
                self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            )
            weights = [
                (self._fix_param_name_to_vllm(name), param.data)
                for name, param in model.named_parameters()
            ]
            llm_model.load_weights(weights=weights)

        # Reset cache
        if self.mode == "server" and accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.mode == "colocate":
            self.llm.reset_prefix_cache()

    return patched_sync_weights


def patch_trl_vllm():
    """Apply all TRL vLLM monkeypatches."""
    import trl.generation.vllm_client
    import trl.generation.vllm_generation
    import trl.trainer.utils

    VLLMClient = trl.generation.vllm_client.VLLMClient
    VLLMGeneration = trl.generation.vllm_generation.VLLMGeneration

    # 1. Add batch_update_named_params to VLLMClient
    if not hasattr(VLLMClient, "batch_update_named_params"):
        VLLMClient.batch_update_named_params = _batch_update_named_params
        VLLMClient.update_model_params = _update_model_params
        LOG.info("Patched VLLMClient with batch_update_named_params")

    # 2. Patch extract_logprobs (NaN→0.0)
    trl.generation.vllm_generation.extract_logprobs = _patched_extract_logprobs
    LOG.info("Patched extract_logprobs with NaN→0.0 fix")

    # 3. Patch VLLMGeneration.__init__ to accept weight_sync_chunk_size
    VLLMGeneration.__init__ = _patch_sync_weights_batched(VLLMGeneration.__init__)

    # 4. Patch sync_weights for batched non-FSDP/non-ZeRO path
    VLLMGeneration.sync_weights = _make_batched_sync_weights(
        VLLMGeneration.sync_weights
    )
    LOG.info("Patched VLLMGeneration with batched sync_weights")

    # 5. Patch split_tensor_dict and shuffle_sequence_dict
    trl.trainer.utils.split_tensor_dict = _patched_split_tensor_dict
    trl.trainer.utils.shuffle_sequence_dict = _patched_shuffle_sequence_dict
    LOG.info("Patched split_tensor_dict and shuffle_sequence_dict for scalar types")
