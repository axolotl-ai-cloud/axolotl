"""CLI to merge sharded FSDP model checkpoints into a single combined checkpoint."""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Union

import fire
import torch
import torch.distributed.checkpoint as dist_cp
import torch.distributed.checkpoint.format_utils as dist_cp_format_utils
import transformers
from accelerate.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    is_torch_version,
)
from dotenv import load_dotenv
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file as safe_save_file
from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner

from axolotl.cli.args import TrainerCliArgs
from axolotl.cli.art import print_axolotl_text_art
from axolotl.cli.config import load_cfg

LOG = logging.getLogger(__name__)


class BFloat16CastPlanner(_EmptyStateDictLoadPlanner):
    """A custom planner to cast tensors to bfloat16 on the fly during loading."""

    def commit_tensor(self, read_item, tensor):  # pylint: disable=unused-argument
        tensor.copy_(tensor.to(torch.bfloat16))


def _distributed_checkpoint_to_merged_weights(
    checkpoint_dir: Union[str, Path],
    save_path: str,
    safe_serialization: bool = False,
    max_shard_size: str = "5GB",
) -> Path:
    """
    Passthrough to `torch.distributed.checkpoint.format_utils.dcp_to_torch_save`. Will
    save under `save_path` as either `model.safetensors` or `pytorch_model.bin`.

    Args:
        checkpoint_dir: Directory where distributed checkpoint is saved.
        save_path: Path to save model to.
        safe_serialization: Whether to save in safetensors format.
        max_shard_size: Max size of model shards to save.

    Returns:
        Path where model is saved.
    """

    state_dict: Dict = {}
    save_path_ = Path(save_path)
    save_path_.mkdir(exist_ok=True)
    dist_cp_format_utils._load_state_dict(  # pylint: disable=protected-access
        state_dict,
        storage_reader=dist_cp.FileSystemReader(checkpoint_dir),
        planner=BFloat16CastPlanner(),  # pylint: disable=protected-access
        no_dist=True,
    )

    # To handle if state is a dict like {model: {...}}
    if len(state_dict.keys()) == 1:
        state_dict = state_dict[list(state_dict)[0]]

    # Ensure all tensors are in bfloat16
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and value.dtype != torch.bfloat16:
            state_dict[key] = value.to(torch.bfloat16)

    weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME

    filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(
        ".safetensors", "{suffix}.safetensors"
    )
    state_dict_split = split_torch_state_dict_into_shards(
        state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
    )

    # Save index if sharded
    index = None
    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }

    # Save the model
    filename_to_tensors = state_dict_split.filename_to_tensors.items()

    for shard_file, tensors in filename_to_tensors:
        shard = {tensor: state_dict[tensor] for tensor in tensors}

        if safe_serialization:
            safe_save_file(
                shard, os.path.join(save_path_, shard_file), metadata={"format": "pt"}
            )
        else:
            torch.save(shard, os.path.join(save_path_, shard_file))

    if index is not None:
        save_index_file = (
            SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
        )
        save_index_file = os.path.join(save_path_, save_index_file)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as fout:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            fout.write(content)

    return save_path_


def merge_fsdp_weights(
    checkpoint_dir: str,
    output_path: str,
    safe_serialization: bool = False,
    remove_checkpoint_dir: bool = False,
):
    """
    Merge the weights from sharded FSDP model checkpoints into a single combined checkpoint. Should be used if
    `SHARDED_STATE_DICT` was used for the model. Weights will be saved to `{output_path}/model.safetensors` if
    `safe_serialization` else `pytorch_model.bin`.

    Note: this is a CPU-bound process.

    Args:
        checkpoint_dir (`str`):
            The directory containing the FSDP checkpoints (can be either the model or optimizer).
        output_path (`str`):
            The path to save the merged checkpoint.
        safe_serialization (`bool`, *optional*, defaults to `True`):
            Whether to save the merged weights with safetensors (recommended).
        remove_checkpoint_dir (`bool`, *optional*, defaults to `False`):
            Whether to remove the checkpoint directory after merging.

    Raises:
        ValueError: If torch version < 2.3.0, or if `checkpoint_dir` does not exist.
    """
    checkpoint_dir_ = Path(checkpoint_dir)
    from accelerate.state import PartialState

    if not is_torch_version(">=", "2.3.0"):
        raise ValueError("`merge_fsdp_weights` requires PyTorch >= 2.3.0`")

    # Verify that the checkpoint directory exists
    if not checkpoint_dir_.exists():
        model_path_exists = (checkpoint_dir_ / "pytorch_model_fsdp_0").exists()
        optimizer_path_exists = (checkpoint_dir_ / "optimizer_0").exists()
        err = f"Tried to load from {checkpoint_dir_} but couldn't find a valid metadata file."
        if model_path_exists and optimizer_path_exists:
            err += (
                " However, potential model and optimizer checkpoint directories exist."
            )
            err += f"Please pass in either {checkpoint_dir_}/pytorch_model_fsdp_0 or {checkpoint_dir_}/optimizer_0"
            err += "instead."
        elif model_path_exists:
            err += " However, a potential model checkpoint directory exists."
            err += (
                f"Please try passing in {checkpoint_dir_}/pytorch_model_fsdp_0 instead."
            )
        elif optimizer_path_exists:
            err += " However, a potential optimizer checkpoint directory exists."
            err += f"Please try passing in {checkpoint_dir_}/optimizer_0 instead."
        raise ValueError(err)

    # To setup `save` to work
    state = PartialState()
    if state.is_main_process:
        LOG.info(f"Merging FSDP weights from {checkpoint_dir_}")
        save_path = _distributed_checkpoint_to_merged_weights(
            checkpoint_dir_, output_path, safe_serialization
        )
        LOG.info(f"Successfully merged FSDP weights and saved to {save_path}")
        if remove_checkpoint_dir:
            LOG.info(f"Removing old checkpoint directory {checkpoint_dir_}")
            shutil.rmtree(checkpoint_dir_)
    state.wait_for_everyone()


def do_cli(config: Union[Path, str] = Path("examples/"), **kwargs):
    """
    Parses `axolotl` config, CLI args, and calls `merge_fsdp_weights`.

    Args:
        config: Path to `axolotl` config YAML file.
        kwargs: Additional keyword arguments to override config file values.
    """
    # pylint: disable=duplicate-code
    print_axolotl_text_art()
    parser = transformers.HfArgumentParser(TrainerCliArgs)
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    parsed_cli_args.merge_lora = True
    parsed_cfg = load_cfg(config, **kwargs)

    fsdp_dir = Path(parsed_cfg.output_dir) / "pytorch_model_fsdp_0"
    merge_fsdp_weights(
        checkpoint_dir=str(fsdp_dir),
        output_path=str(Path(parsed_cfg.output_dir) / "merged"),
        safe_serialization=True,
    )


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(do_cli)
