import os
import re
from pathlib import Path
from typing import Tuple

import safetensors
import torch
from huggingface_hub import snapshot_download, split_torch_state_dict_into_shards
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig
from transformers.utils import SAFE_WEIGHTS_NAME


def extract_layer_number(key):
    """Extract layer number from parameter key."""
    match = re.search(r'layers\.(\d+)\.', key)
    return int(match.group(1)) if match else None


def iter_parameter_weights(model_path, device="mps"):
    """
    iterator over parameter weights in the model shards

    :param model_path: Path to model shards
    :param device: Computing device
    :return: generator yielding (parameter key, parameter weight, layer index) tuples
    """
    shards = list(model_path.glob('model*.safetensors'))
    if not shards:
        raise ValueError(f"No model shards found in {model_path}")

    for shard in tqdm(shards, desc="Processing shards"):
        with safetensors.safe_open(shard, framework='pt', device=device) as f:
                for key in f.keys():
                    layer_idx = extract_layer_number(key)
                    weight = f.get_tensor(key)
                    yield key, weight, layer_idx

def iter_recursive_parameter_weights(model_path, modules_to_recurse: list[str], device="mps", recurse_layers=12):
    # setup placeholder state_dict for recursive weights, need to keep in float32 precision
    # to avoid precision loss when averaging weights across layers
    rrt_avg_model_state_dict = {}

    # iterate over all parameter weights in the model shards
    for key, weight, layer_idx in iter_parameter_weights(model_path):
        # get the matching module name in modules_to_recurse for the current parameter key
        matched_module_name = next(
            (module for module in modules_to_recurse if module in key),
            None
        )
        if matched_module_name is None:
            continue

        recurse_idx = layer_idx % recurse_layers
        suffix = f"{recurse_idx}.{matched_module_name}"
        if rrt_avg_model_state_dict.get(suffix) is None:
            # setup as storage for suffix with torch.stack
            rrt_avg_model_state_dict[suffix] = [weight.to(torch.float32).detach().cpu()]
        else:
            rrt_avg_model_state_dict[suffix].append(weight.to(torch.float32).detach().cpu())

    for module_name in modules_to_recurse:
        for recurse_idx in range(recurse_layers):
            suffix = f"{recurse_idx}.{module_name}"
            prefix = f"model.layers.{suffix}"
            avg_weight = torch.stack(rrt_avg_model_state_dict[suffix]).mean(dim=0)
            yield f"{prefix}.weight_base", avg_weight

    # compute the decomposed lora diff from the weight base to the actual weight for each module

def low_rank_decomposition(
        weight: torch.Tensor, max_rank: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose a 2D matrix into low-rank matrices L and R using SVD.

    :param weight: The matrix to decompose, of shape (H, W)
    :param max_rank: The maximum rank of the decomposition
    :return: A tuple of tensors (L, R)
    """
    assert (
            weight.dim() == 2
    ), f"Only support 2D matrix, but input has {weight.dim()} dimensions."
    assert (
            max_rank >= 1
    ), f"Maximum rank must be a positive integer, but input max_rank={max_rank}."

    dtype = weight.dtype

    U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)

    final_rank = min(min(weight.shape), max_rank)

    # Distribute S to both to improve numerical precision.
    sqrt_S = torch.sqrt(torch.diag(S[:final_rank]))
    L = sqrt_S @ Vh[:final_rank, :]
    R = U[:, :final_rank] @ sqrt_S

    return L.to(dtype), R.to(dtype)

def decompose_delta_weight(layer_weight, avg_weight, alpha, rank):
    """
    Decompose the difference in directions (ΔV) via SVD,
    and return (magnitudes, L, R).
    """
    device = "cuda" if torch.cuda.is_available() else "mps"

    base_weight = avg_weight.to(device)
    finetuned_weight = layer_weight.to(device)

    # 1. Compute column norms and directions
    #    (shape: base_norms, finetuned_norms => (k,))
    base_norms = torch.norm(base_weight, dim=0) + 1e-9
    finetuned_norms = torch.norm(finetuned_weight, dim=0) + 1e-9

    # shape (d, k)
    base_dir = base_weight / base_norms
    finetuned_dir = finetuned_weight / finetuned_norms

    # 2. Delta direction
    delta_dir = finetuned_dir - base_dir

    # 3. Low-rank factorization of the delta direction
    A, B = low_rank_decomposition(delta_dir, rank)
    # The final magnitudes are just finetuned_norms
    return A.cpu(), B.cpu(), finetuned_norms.cpu()


def iter_dora_parameter_weights(model_path, avg_recursive_weights, modules_to_recurse: list[str], alpha, rank, device="mps", recurse_layers=12):
    rrt_avg_model_state_dict = {}

    # iterate over all parameter weights in the model shards
    for key, weight, layer_idx in iter_parameter_weights(model_path):
        # get the matching module name in modules_to_recurse for the current parameter key
        matched_module_name = next(
            (module for module in modules_to_recurse if module in key),
            None
        )
        if matched_module_name is None:
            if "input_layernorm" in key:
                # map to input_layernorm_list in the recursive layers and account for the layer_idx and loop_idx
                loop_idx = layer_idx // recurse_layers
                layer_idx = layer_idx % recurse_layers
                layernorm_key = f"model.layers.{layer_idx}.input_layernorm_list.{loop_idx}"
                yield layernorm_key, weight
            elif "post_attention_layernorm" in key:
                # map to input_layernorm_list in the recursive layers and account for the layer_idx and loop_idx
                loop_idx = layer_idx // recurse_layers
                layer_idx = layer_idx % recurse_layers
                layernorm_key = f"model.layers.{layer_idx}.post_attention_layernorm_list.{loop_idx}"
                yield layernorm_key, weight
            else:
                yield key, weight
            continue

        # figure out the base weight layer for this key
        loop_idx = layer_idx // recurse_layers
        layer_idx = layer_idx % recurse_layers
        suffix = f"{layer_idx}.{matched_module_name}"
        prefix = f"model.layers.{suffix}.weight_base"
        avg_weight = avg_recursive_weights[prefix]
        lora_a_key =  f"model.layers.{suffix}.lora_A_list.{loop_idx}"
        lora_b_key =  f"model.layers.{suffix}.lora_B_list.{loop_idx}"
        lora_magnitude_key =  f"model.layers.{suffix}.lora_magnitude_vector_list.{loop_idx}"
        lora_a, lora_b, lora_magnitude = decompose_delta_weight(weight, avg_weight, alpha, rank)
        yield lora_a_key, lora_a
        yield lora_b_key, lora_b
        yield lora_magnitude_key, lora_magnitude

def save_state_dict_to_safetensors(state_dict, save_directory):
    weights_name = SAFE_WEIGHTS_NAME

    filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
    state_dict_split = split_torch_state_dict_into_shards(
        state_dict, filename_pattern=filename_pattern, max_shard_size="1GB"
    )
    # Save index if sharded
    index = None
    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }

    # Clean the folder from a previous save
    for filename in os.listdir(save_directory):
        full_filename = os.path.join(save_directory, filename)
        # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
        # in distributed settings to avoid race conditions.
        weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")

        # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
        filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
        reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

        if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in state_dict_split.filename_to_tensors.keys()
                and reg.fullmatch(filename_no_suffix) is not None
        ):
            os.remove(full_filename)

    filename_to_tensors = state_dict_split.filename_to_tensors.items()
    for shard_file, tensors in filename_to_tensors:
        shard = {}
        for tensor in tensors:
            shard[tensor] = state_dict[tensor].contiguous()
            del state_dict[tensor]

        save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "pt"})

def convert_llama_to_rrt(model_name, output_dir, recurse_layers: int = 12, rank=32):
    modules_to_recurse = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.down_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
    ]

    config = AutoConfig.from_pretrained(model_name)
    num_hidden_layers = config.num_hidden_layers
    if num_hidden_layers % recurse_layers != 0:
        raise ValueError(
            f"The number of hidden layers ({num_hidden_layers}) in the model must be "
            f"divisible by the recurse layers ({recurse_layers})"
        )

    model_path = Path(snapshot_download(model_name, ignore_patterns="*.pth"))

    # create a new state_dict to store the RRT model weights
    rrt_model_state_dict = {}

    for key, weight in iter_recursive_parameter_weights(model_path, modules_to_recurse, device="mps", recurse_layers=recurse_layers):
        rrt_model_state_dict[key] = weight.to(torch.bfloat16).detach().cpu()

    # now that we have the average weights, we need to loop over the shards again to calculate the decomposed lora diff
    rrt_lora_state_dict = {}
    for key, weight in iter_dora_parameter_weights(model_path, rrt_model_state_dict, modules_to_recurse, alpha=32, rank=rank, device="mps", recurse_layers=recurse_layers):
        rrt_lora_state_dict[key] = weight.to(torch.bfloat16).detach().cpu()

    # combine state dicts into a single state_dict
    rrt_model_state_dict.update(rrt_lora_state_dict)

    # save state dict as sharded safetensors to disk using split_torch_state_dict_into_shards
    save_state_dict_to_safetensors(rrt_model_state_dict, output_dir)


if __name__ == "__main__":
    convert_llama_to_rrt("meta-llama/Llama-3.2-1B", "/tmp/rrt_model", recurse_layers=4, rank=32)
