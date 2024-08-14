"""
This module contains utility functions for saving very large models
"""
import json
import os

from accelerate.utils.other import save
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType


def save_sharded_fsdp_model(model, output_dir, chunk_size=1024 * 1024 * 1024):
    os.makedirs(output_dir, exist_ok=True)
    metadata = {
        "metadata": {"total_size": 0, "num_chunks": 0, "chunk_size": chunk_size},
        "weight_map": {},
        "chunks": [],
    }

    chunk_id = 0
    buffer = {}
    buffer_size = 0
    chunk_tensor_count = 0

    for name, param in model.named_parameters():
        if isinstance(param, FSDP):
            with FSDP.state_dict_type(param, StateDictType.FULL_STATE_DICT):
                param_state = param.state_dict()
            for subname, tensor in param_state.items():
                full_name = f"{name}.{subname}"
                buffer[full_name] = tensor.detach().cpu()
                buffer_size += tensor.numel() * tensor.element_size()
                metadata["weight_map"][full_name] = f"model-{chunk_id:05d}.safetensors"
                chunk_tensor_count += 1
        else:
            buffer[name] = param.detach().cpu()
            buffer_size += param.numel() * param.element_size()
            metadata["weight_map"][name] = f"model-{chunk_id:05d}.safetensors"
            chunk_tensor_count += 1

        if buffer_size >= chunk_size:
            chunk_file = os.path.join(output_dir, f"model-{chunk_id:05d}.safetensors")
            save(buffer, chunk_file)
            metadata["chunks"].append(
                {
                    "filename": f"model-{chunk_id:05d}.safetensors",
                    "size": buffer_size,
                    "num_tensors": chunk_tensor_count,
                }
            )
            metadata["metadata"]["total_size"] += buffer_size
            metadata["metadata"]["num_chunks"] += 1
            print(f"Saved chunk {chunk_id} to {chunk_file}")
            buffer = {}
            buffer_size = 0
            chunk_tensor_count = 0
            chunk_id += 1

    if buffer:
        chunk_file = os.path.join(output_dir, f"model-{chunk_id:05d}.safetensors")
        save(buffer, chunk_file)
        metadata["chunks"].append(
            {
                "filename": f"model-{chunk_id:05d}.safetensors",
                "size": buffer_size,
                "num_tensors": chunk_tensor_count,
            }
        )
        metadata["metadata"]["total_size"] += buffer_size
        metadata["metadata"]["num_chunks"] += 1
        print(f"Saved final chunk {chunk_id} to {chunk_file}")

    # go back and rename weight_map files to model-{chunk_id:05d}-of-{num_chunks:05d}.safetensors
    for name, filename in metadata["weight_map"].items():
        chunk_id = int(filename.split("-")[1])
        metadata["weight_map"][
            name
        ] = f"model-{chunk_id:05d}-of-{metadata['metadata']['num_chunks']:05d}.safetensors"
        # rename the files already saved to disk in the output directory
        os.rename(
            os.path.join(output_dir, filename),
            os.path.join(output_dir, metadata["weight_map"][name]),
        )

    with open(
        os.path.join(output_dir, "model.safetensors.metadata.json"),
        "w",
        encoding="utf-8",
    ) as fout:
        json.dump(metadata, fout, indent=2)
