"""Expert ownership for compact adapters and portable checkpoints."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist


def adapter_owner(wrapper, kind):
    """Map local compact factors into the saved global compact expert order."""
    base = wrapper.get_base_layer()
    selected = wrapper.global_selected_experts
    offset = getattr(base, "local_expert_offset", 0)
    return {
        "kind": kind,
        "indices": [selected.index(i + offset) for i in wrapper.selected_experts],
        "total": len(selected),
        "local": len(wrapper.selected_experts),
        "rank": wrapper.r[wrapper.active_adapters[0]],
    }


def select_adapter(tensor, kind, positions, total, rank):
    """Select experts without confusing B's rank-major packing with expert order."""
    if kind == "A":
        return tensor.reshape(total, rank, tensor.shape[1])[positions].flatten(0, 1)
    return tensor.reshape(tensor.shape[0], rank, total)[:, :, positions].flatten(1, 2)


@contextmanager
def local_adapter_checkpoint(model, config, checkpoint):
    """Present PEFT with this EP rank's compact factors while keeping global metadata."""
    from peft.utils.save_and_load import load_peft_weights
    from safetensors.torch import save_file

    from .selection import validate_selection

    modules = validate_selection(model, config.moe_sieve_selection)
    if not any(hasattr(module, "num_experts_global") for module, _ in modules.values()):
        yield checkpoint
        return
    weights = load_peft_weights(checkpoint, device="cpu")
    for key, tensor in list(weights.items()):
        if not key.endswith((".lora_A.weight", ".lora_B.weight")):
            continue
        name, kind, _ = key.rsplit(".", 2)
        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
        if name not in modules:
            continue
        module = modules[name][0]
        ids = config.moe_sieve_selection[name]["selected_experts"]
        offset = getattr(module, "local_expert_offset", 0)
        positions = [
            j for j, i in enumerate(ids) if offset <= i < offset + module.num_experts
        ]
        dim = 0 if kind == "lora_A" else 1
        if tensor.shape[dim] % len(ids):
            raise ValueError(f"Invalid compact adapter shape for {key}")
        rank = tensor.shape[dim] // len(ids)
        weights[key] = select_adapter(
            tensor, kind[-1], positions, len(ids), rank
        ).contiguous()
    with TemporaryDirectory(prefix="axolotl-moe-sieve-") as directory:
        config.save_pretrained(directory)
        save_file(weights, str(Path(directory) / "adapter_model.safetensors"))
        yield directory


def gather_adapter(tensor, wrapper, kind, group):
    """Assemble uneven expert selections across EP, including empty owners."""
    owner = adapter_owner(wrapper, kind)
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if "cpu:" not in dist.get_backend_config(group)
        else tensor.device
    )
    indices = torch.tensor(owner["indices"], device=device, dtype=torch.long)
    tensor = tensor.to(device)
    rank, total, local = owner["rank"], owner["total"], owner["local"]
    if kind == "A":
        full = tensor.new_zeros(total, rank, tensor.shape[1])
        full.index_copy_(0, indices, tensor.reshape(local, rank, tensor.shape[1]))
        full = full.flatten(0, 1)
    else:
        full = tensor.new_zeros(tensor.shape[0], rank, total)
        full.index_copy_(2, indices, tensor.reshape(tensor.shape[0], rank, local))
        full = full.flatten(1, 2)
    dist.all_reduce(full, group=group)
    return full
