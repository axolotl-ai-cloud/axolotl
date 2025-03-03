import torch
from yunchang import set_seq_parallel_pg
from yunchang.comm import zigzag_extract_local


def prepare_usp_attn_inputs(
    input_ids,
    position_ids,
    target_ids,
    rank,
    world_size,
    device,
    ring_degree,
    ulysses_degree,
):
    f"""
    prepare input for USP attention

    USP: A Unified Sequence Parallelism Approach for Long Context Generative AI
    https://arxiv.org/abs/2405.07719
    """

    set_seq_parallel_pg(ulysses_degree, ring_degree, rank, world_size)

    local_input_ids = zigzag_extract_local(
        input_ids,
        rank,
        world_size,
        ring_degree,
        ulysses_degree,
    ).to(device)

    # truncate position_ids to the same size as input_ids
    position_ids = position_ids[:, : local_input_ids.shape[1]]

    local_position_ids = zigzag_extract_local(
        position_ids,
        rank,
        world_size,
        ring_degree,
        ulysses_degree,
    ).to(device)

    if target_ids is not None:
        local_target_ids = zigzag_extract_local(
            target_ids,
            rank,
            world_size,
            ring_degree,
            ulysses_degree,
        ).to(device)
    else:
        local_target_ids = None
    return {
        "local_input_ids": local_input_ids,
        "local_position_ids": local_position_ids,
        "local_target_ids": local_target_ids,
    }
