import logging

from .dist_flash_attn.monkey_patch import apply_dist_flash_attn_monkey_patch_llama
from .dist_flash_attn.prepare_input import prepare_dist_flash_attn_inputs
from .ulysses_attn.monkey_patch import apply_ulysses_attn_monkey_patch_llama
from .ulysses_attn.prepare_inputs import prepare_ulysses_attn_inputs
from .usp.monkey_patch import apply_usp_attn_monkey_patch_llama
from .usp.prepare_inputs import prepare_usp_attn_inputs
from .zigzag_ring_attn.monkey_patch import (
    apply_zigzag_ring_attn_monkey_patch_llama,
    apply_zigzag_ring_attn_monkey_patch_mistral,
)
from .zigzag_ring_attn.prepare_inputs import prepare_zigzag_ring_attn_inputs

logger = logging.getLogger(__name__)


def prepare_seq_parallel_inputs(
    seq_algo,
    input_ids,
    position_ids,
    target_ids,
    rank,
    world_size,
    device,
    *args,
    **kwargs,
):
    if seq_algo == "zigzag_ring_attn":
        return prepare_zigzag_ring_attn_inputs(
            input_ids, position_ids, target_ids, rank, world_size, device
        )
    elif seq_algo == "dist_flash_attn":
        return prepare_dist_flash_attn_inputs(
            input_ids, position_ids, target_ids, rank, world_size, device
        )
    elif seq_algo == "ulysses_attn":
        return prepare_ulysses_attn_inputs(
            input_ids, position_ids, target_ids, rank, world_size, device
        )
    elif seq_algo == "usp_attn":
        ring_degree = kwargs.get("ring_degree", 1)
        ulysses_degree = world_size // ring_degree
        logger.info(
            f"Applying USP: Ring degree: {ring_degree}, Ulysses degree: {ulysses_degree}"
        )
        return prepare_usp_attn_inputs(
            input_ids,
            position_ids,
            target_ids,
            rank,
            world_size,
            device,
            ulysses_degree,
            ring_degree,
        )
    elif seq_algo == "data_parallel":
        return {
            "local_input_ids": input_ids.to(device),
            "local_position_ids": position_ids.to(device),
            "local_target_ids": target_ids.to(device),
        }
    else:
        raise ValueError(f"Invalid seq_algo: {seq_algo}")


def apply_seq_parallel_monkey_patch(seq_algo, model):
    assert seq_algo in [
        "zigzag_ring_attn",
        "dist_flash_attn",
        "ulysses_attn",
        "data_parallel",
        "usp_attn",
    ], f"Invalid seq_algo: {seq_algo}"
    assert model in ["llama", "mistral"], f"Invalid model: {model}"
    if seq_algo == "data_parallel":
        return
    elif seq_algo == "zigzag_ring_attn" and model == "llama":
        apply_zigzag_ring_attn_monkey_patch_llama()
    elif seq_algo == "zigzag_ring_attn" and model == "mistral":
        apply_zigzag_ring_attn_monkey_patch_mistral()
    elif seq_algo == "dist_flash_attn" and model == "llama":
        apply_dist_flash_attn_monkey_patch_llama()
    elif seq_algo == "ulysses_attn" and model == "llama":
        apply_ulysses_attn_monkey_patch_llama()
    elif seq_algo == "usp_attn" and model == "llama":
        apply_usp_attn_monkey_patch_llama()
    else:
        raise ValueError(f"Invalid seq_algo: {seq_algo} or model: {model}")


def prepare_dataloader(seq_algo, dataloader, accelerator):
    if seq_algo == "data_parallel":
        return accelerator.prepare(dataloader)
    else:
        return dataloader
