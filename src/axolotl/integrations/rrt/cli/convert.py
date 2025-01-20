from pathlib import Path
from typing import re

import safetensors
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoConfig


def extract_layer_number(key):
    """Extract layer number from parameter key."""
    match = re.search(r'layers\.(\d+)\.', key)
    return int(match.group(1)) if match else None


def iter_parameter_weights(model_path, device="cpu"):
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

def iter_recursive_parameter_weights(model_path, modules_to_recurse: list[str], device="cpu", recurse_layers=12):
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
            if "input_layernorm" in key:
                # map to input_layernorm_list in the recursive layers and account for the layer_idx and loop_idx
                yield
            else:
                yield key, weight

        recurse_idx = layer_idx % recurse_layers
        suffix = f"{recurse_idx}.{matched_module_name}"
        prefix = f"model.layers.{suffix}."
        if rrt_avg_model_state_dict.get(suffix) is None:
            # setup as storage for suffix with torch.stack
            rrt_avg_model_state_dict[suffix] = torch.stack([weight.to(torch.float32).detach().cpu()])
        else:
            rrt_avg_model_state_dict[suffix] = torch.cat([rrt_avg_model_state_dict[suffix], weight.to(torch.float32).detach().cpu()])

    for module_name in modules_to_recurse:
        for recurse_idx in range(recurse_layers):
            suffix = f"{recurse_idx}.{module_name}"
            prefix = f"model.layers.{suffix}."
            avg_weight = rrt_avg_model_state_dict[suffix].mean(dim=0)
            yield f"{prefix}.weight", avg_weight


def convert_llama_to_rrt(model_name, output_dir, recurse_layers: int = 12):
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

    model_path = Path(snapshot_download(model_name))

    # create a new state_dict to store the RRT model weights
    rrt_model_state_dict = {}

    for key, weight in iter_recursive_parameter_weights(model_path, modules_to_recurse, device="cpu", recurse_layers=recurse_layers):
        rrt_model_state_dict[key] = weight.to(torch.bfloat16).detach().cpu()

    # split_torch_state_dict_into_shards(...)
