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
        logger.warning("Falling back to load_weights for param: %s", name)
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
