"""CLI to merge sharded FSDP model checkpoints into a single combined checkpoint."""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Union

import fire
import torch
import torch.distributed.checkpoint as dist_cp
import torch.distributed.checkpoint.format_utils as dist_cp_format_utils
from accelerate import PartialState
from accelerate.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    is_torch_version,
)
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file as safe_save_file
from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner

from axolotl.cli.config import load_cfg
from axolotl.telemetry.errors import send_errors
from axolotl.utils.logging import get_logger
from axolotl.utils.train import determine_last_checkpoint

LOG = get_logger(__name__)


class BFloat16CastPlanner(_EmptyStateDictLoadPlanner):
    """A custom planner to cast tensors to bfloat16 on the fly during loading."""

    def commit_tensor(self, read_item, tensor):
        tensor.copy_(tensor.to(torch.bfloat16))


def _distributed_checkpoint_to_merged_weights(
    checkpoint_dir: Union[str, Path],
    save_path: str,
    max_shard_size: str = "5GB",
) -> Path:
    """
    Passthrough to `torch.distributed.checkpoint.format_utils.dcp_to_torch_save`. Will
    save under `save_path` as `model.safetensors`.

    Args:
        checkpoint_dir: Directory where distributed checkpoint is saved.
        save_path: Path to save model to.
        max_shard_size: Max size of model shards to save.

    Returns:
        Path where model is saved.
    """

    state_dict: Dict = {}
    save_path_ = Path(save_path)
    save_path_.mkdir(exist_ok=True)
    dist_cp_format_utils._load_state_dict(
        state_dict,
        storage_reader=dist_cp.FileSystemReader(checkpoint_dir),
        planner=BFloat16CastPlanner(),
        no_dist=True,
    )

    # To handle if state is a dict like {model: {...}}
    if len(state_dict.keys()) == 1:
        state_dict = state_dict[list(state_dict)[0]]

    # Ensure all tensors are in bfloat16
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and value.dtype != torch.bfloat16:
            state_dict[key] = value.to(torch.bfloat16)

    filename_pattern = SAFE_WEIGHTS_NAME.replace(".safetensors", "{suffix}.safetensors")
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
        safe_save_file(
            shard, os.path.join(save_path_, shard_file), metadata={"format": "pt"}
        )

    if index is not None:
        save_index_file = os.path.join(save_path_, SAFE_WEIGHTS_INDEX_NAME)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as fout:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            fout.write(content)

    return save_path_


@send_errors
def merge_fsdp_weights(
    checkpoint_dir: str,
    output_path: str,
    remove_checkpoint_dir: bool = False,
):
    """
    Merge the weights from sharded FSDP model checkpoints into a single combined checkpoint. Should be used if
    `SHARDED_STATE_DICT` was used for the model. Weights will be saved to `{output_path}/model.safetensors`.

    Note: this is a CPU-bound process.

    Args:
        checkpoint_dir (`str`):
            The directory containing the FSDP checkpoints (can be either the model or optimizer).
        output_path (`str`):
            The path to save the merged checkpoint.
        remove_checkpoint_dir (`bool`, *optional*, defaults to `False`):
            Whether to remove the checkpoint directory after merging.

    Raises:
        ValueError: If torch version < 2.3.0, or if `checkpoint_dir` does not exist.
    """
    checkpoint_dir_ = Path(checkpoint_dir)

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
            checkpoint_dir_, output_path
        )
        LOG.info(f"Successfully merged FSDP weights and saved to {save_path}")
        if remove_checkpoint_dir:
            LOG.info(f"Removing old checkpoint directory {checkpoint_dir_}")
            shutil.rmtree(checkpoint_dir_)


def do_cli(config: Union[Path, str] = Path("examples/"), **kwargs):
    """
    Parses `axolotl` config, CLI args, and calls `merge_fsdp_weights`.

    Args:
        config: Path to `axolotl` config YAML file.
        kwargs: Additional keyword arguments to override config file values.
    """

    parsed_cfg = load_cfg(config, **kwargs)

    fsdp_dir = Path(parsed_cfg.output_dir) / "pytorch_model_fsdp_0"
    if not fsdp_dir.exists():
        checkpoint_dir = determine_last_checkpoint(parsed_cfg, update=False)
        if checkpoint_dir:
            fsdp_dir = Path(checkpoint_dir) / "pytorch_model_fsdp_0"
        if not fsdp_dir.exists():
            raise ValueError(
                f"Could not find FSDP checkpoint `pytorch_model_fsdp_0` in {checkpoint_dir}"
            )

    output_path = str(Path(parsed_cfg.output_dir) / "merged")
    merge_fsdp_weights(
        checkpoint_dir=str(fsdp_dir),
        output_path=output_path,
    )
    state = PartialState()
    state.wait_for_everyone()
    LOG.info(
        f"FSDP SHARDED_STATE_DICT weights successfully merged to: {output_path}",
        main_process_only=True,
    )
    LOG.info(
        "Merged weights are only the safetensors and doesn't include the model configuration "
        f"or tokenizer which may be found in {parsed_cfg.output_dir}.",
        main_process_only=True,
    )


if __name__ == "__main__":
    fire.Fire(do_cli)
