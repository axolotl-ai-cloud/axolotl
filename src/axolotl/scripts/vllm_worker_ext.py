"""Extended vLLM worker extension with batch weight sync support.

Subclasses TRL's WeightSyncWorkerExtension to add:
- batch_update_named_params: receives multiple params in one call
- Auto-close stale communicator on re-init
- _direct_set_weight: proper handling for stacked (qkv_proj, gate_up_proj) params,
  including LoRA-wrapped models where vLLM inserts base_layer into the hierarchy
"""

import logging

import torch

try:
    from transformers import is_torch_xpu_available
except ImportError:
    is_torch_xpu_available = lambda: False  # noqa: E731

from trl.scripts.vllm_serve import WeightSyncWorkerExtension

logger = logging.getLogger(__name__)

# Stacked param name mapping: shard_name -> (packed_name, shard_order)
_STACKED_PARAMS = {
    "q_proj": ("qkv_proj", 0),
    "k_proj": ("qkv_proj", 1),
    "v_proj": ("qkv_proj", 2),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}


class BatchWeightSyncWorkerExtension(WeightSyncWorkerExtension):
    """Worker extension that adds batch weight update and direct weight setting."""

    def init_communicator(self, host, port, world_size, client_device_uuid):
        """Auto-close stale communicator before re-initializing."""
        if self.communicator is not None:
            self.close_communicator()
        super().init_communicator(host, port, world_size, client_device_uuid)

    def _direct_set_weight(self, name: str, weight: torch.Tensor) -> None:
        """Directly copy weight data into the model, handling stacked params.

        Bypasses model.load_weights() which may fail on vLLM 0.17's new
        module-tree weight loader for stacked params (qkv_proj, gate_up_proj).

        Handles LoRA-wrapped params where vLLM inserts ``base_layer`` into the
        parameter hierarchy (e.g. ``qkv_proj.base_layer.weight``).
        """
        model = self.model_runner.model
        params_dict = dict(model.named_parameters())

        # Handle VLM models where trainer and vLLM use different prefixes.
        # Trainer (PEFT stripped): "model.layers.X..." or "model.language_model.layers.X..."
        # vLLM (Qwen3.5):         "language_model.model.layers.X..."
        if name not in params_dict:
            # Try common prefix remappings
            for src_prefix, dst_prefix in [
                ("model.language_model.layers.", "language_model.model.layers."),
                ("model.layers.", "language_model.model.layers."),
            ]:
                if name.startswith(src_prefix):
                    name = dst_prefix + name[len(src_prefix) :]
                    break

        # Check if this is a simple direct param (exists as-is)
        if name in params_dict:
            params_dict[name].data.copy_(weight.to(params_dict[name].dtype))
            return

        # Also check with base_layer inserted: x.y.weight -> x.y.base_layer.weight
        parts_bl = name.rsplit(".", 1)
        if len(parts_bl) == 2:
            base_layer_name = f"{parts_bl[0]}.base_layer.{parts_bl[1]}"
            if base_layer_name in params_dict:
                params_dict[base_layer_name].data.copy_(
                    weight.to(params_dict[base_layer_name].dtype)
                )
                return

        # Handle stacked params: e.g. "model.layers.0.self_attn.q_proj.weight"
        # -> "model.layers.0.self_attn.qkv_proj.weight" with shard offset
        parts = name.rsplit(".", 2)  # [prefix, layer_name, suffix]
        if len(parts) == 3:
            prefix, layer_name, suffix = parts
            if layer_name in _STACKED_PARAMS:
                packed_name, shard_idx = _STACKED_PARAMS[layer_name]
                for packed_full in [
                    f"{prefix}.{packed_name}.{suffix}",
                    f"{prefix}.{packed_name}.base_layer.{suffix}",
                ]:
                    if packed_full not in params_dict:
                        continue
                    param = params_dict[packed_full]
                    # Navigate to the packed module to find shard sizes
                    module_path = packed_full.rsplit(".", 1)[0]  # strip .weight/.bias
                    if ".base_layer" in module_path:
                        module_path = module_path.replace(".base_layer", "")
                    module = model
                    for attr in module_path.split("."):
                        module = getattr(module, attr, None)
                        if module is None:
                            break
                    # LoRA wrappers don't have output_sizes directly;
                    # check base_layer for the underlying parallel linear
                    if module is not None and not hasattr(module, "output_sizes"):
                        base = getattr(module, "base_layer", None)
                        if base is not None and hasattr(base, "output_sizes"):
                            module = base
                    if module is not None and hasattr(module, "output_sizes"):
                        tp_size = getattr(module, "tp_size", 1)
                        sizes = [s // tp_size for s in module.output_sizes]
                        offset = sum(sizes[:shard_idx])
                        shard_size = sizes[shard_idx]
                        param.data[offset : offset + shard_size].copy_(
                            weight.to(param.dtype)
                        )
                        return

        # Fallback: try load_weights (may work for non-stacked params)
        # Log the actual param names available for debugging
        sample_keys = [
            k for k in params_dict if "layers.31.mlp" in k or "layers.31.self_attn" in k
        ][:3]
        logger.warning(
            "Falling back to load_weights for param: %s (sample vLLM keys: %s)",
            name,
            sample_keys,
        )
        model.load_weights(weights=[(name, weight)])

    def update_named_param(self, name, dtype, shape):
        """Override to use _direct_set_weight instead of load_weights."""
        if self.communicator is None:
            raise RuntimeError("Communicator not initialized.")

        dtype = getattr(torch, dtype.split(".")[-1])
        weight = torch.empty(shape, dtype=dtype, device=self.device)

        if is_torch_xpu_available():
            self.communicator.broadcast(weight, root=self.client_rank)
            self.communicator.barrier()
        else:
            self.communicator.broadcast(weight, src=self.client_rank)
            self.communicator.group.barrier()

        self._direct_set_weight(name, weight)

    def batch_update_named_params(self, params_list: list[tuple[str, str, tuple]]):
        """Receive and apply multiple weight tensors in sequence.

        Args:
            params_list: List of (name, dtype_str, shape) tuples.
        """
        if self.communicator is None:
            raise RuntimeError("Communicator not initialized.")

        weights_to_load = []
        for name, dtype_str, shape in params_list:
            dtype = getattr(torch, dtype_str.split(".")[-1])
            weight = torch.empty(shape, dtype=dtype, device=self.device)

            if is_torch_xpu_available():
                self.communicator.broadcast(weight, root=self.client_rank)
            else:
                self.communicator.broadcast(weight, src=self.client_rank)

            weights_to_load.append((name, weight))

        # Single barrier after all broadcasts
        if is_torch_xpu_available():
            self.communicator.barrier()
        else:
            self.communicator.group.barrier()

        # Load weights using direct set (handles stacked params)
        for name, weight in weights_to_load:
            self._direct_set_weight(name, weight)

    def http_load_weights(self, weights: list[tuple[str, torch.Tensor]]):
        """Load weights received via HTTP (no NCCL needed)."""
        for name, weight in weights:
            self._direct_set_weight(name, weight.to(self.device))

    def http_load_weight(self, **kwargs):
        """Load a single weight received via HTTP (no NCCL needed).

        Reconstructs the tensor from raw bytes since tensors don't survive
        vLLM's multiproc IPC serialization.  Uses vLLM's ``load_weights``
        which handles TP sharding and stacked-param packing automatically.
        """
        from axolotl.utils.weight_serde import decode_from_ipc

        name, weight = decode_from_ipc(kwargs)
        model = self.model_runner.model
        model.load_weights(weights=[(name, weight)])

    def http_load_weights_batch(self, params: list[dict]):
        """Load multiple weights in a single IPC call.

        Uses vLLM's ``load_weights`` which handles TP sharding automatically.
        """
        from axolotl.utils.weight_serde import decode_from_ipc

        model = self.model_runner.model
        weights = [decode_from_ipc(p) for p in params]
        model.load_weights(weights=weights)
