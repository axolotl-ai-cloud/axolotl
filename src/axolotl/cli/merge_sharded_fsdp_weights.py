"""
This module provides a CLI to merge sharded FSDP model checkpoints into a single combined checkpoint
"""
import logging
import shutil
from pathlib import Path
from typing import Dict, Union

from accelerate.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME, is_torch_version, save

LOG = logging.getLogger("axolotl.cli.merge_sharded_fsdp_weights")


def _distributed_checkpoint_to_merged_weights(
    checkpoint_dir: Union[str, Path], save_path: str, safe_serialization: bool = True
):
    """
    Passthrough to `torch.distributed.checkpoint.format_utils.dcp_to_torch_save`

    Will save under `save_path` as either `model.safetensors` or `pytorch_model.bin`.
    """
    # Note: We import here to reduce import time from general modules, and isolate outside dependencies
    import torch.distributed.checkpoint as dist_cp
    import torch.distributed.checkpoint.format_utils as dist_cp_format_utils

    state_dict: Dict = {}
    save_path_ = Path(save_path)
    save_path_.mkdir(exist_ok=True)
    dist_cp_format_utils._load_state_dict(  # pylint: disable=protected-access
        state_dict,
        storage_reader=dist_cp.FileSystemReader(checkpoint_dir),
        planner=dist_cp_format_utils._EmptyStateDictLoadPlanner(),  # pylint: disable=protected-access
        no_dist=True,
    )
    save_path_ = (
        save_path_ / SAFE_WEIGHTS_NAME
        if safe_serialization
        else save_path_ / WEIGHTS_NAME
    )

    # To handle if state is a dict like {model: {...}}
    if len(state_dict.keys()) == 1:
        state_dict = state_dict[list(state_dict)[0]]
    save(state_dict, save_path_, safe_serialization=safe_serialization)
    return save_path_


def merge_fsdp_weights(
    checkpoint_dir: str,
    output_path: str,
    safe_serialization: bool = True,
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
