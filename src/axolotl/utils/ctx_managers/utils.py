import contextlib
from typing import Callable, Generator, Optional, Union

import torch

from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import set_rotate_method
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import BlockMask
from transformers import PreTrainedModel


def _get_sdpa_context() -> (
    Callable[[Optional[Generator[None, None, None]]], Generator[None, None, None]]
):
    """
    Creates a context manager to confine to flash/efficient/cuDNN attention backends.

    Returns:
        A context manager function that takes an optional context parallel context.
    """

    @contextlib.contextmanager
    def context(cp_context: Union[Generator[None, None, None], None] = None):
        with contextlib.ExitStack() as stack:
            if cp_context is not None:
                stack.enter_context(
                    sdpa_kernel(
                        [
                            SDPBackend.FLASH_ATTENTION,
                            SDPBackend.EFFICIENT_ATTENTION,
                            SDPBackend.CUDNN_ATTENTION,
                        ]
                    )
                )
                stack.enter_context(cp_context)

            yield

    return context


def get_context_parallel_manager(
    *,
    enabled: bool = False,
    world_mesh: torch.distributed.DeviceMesh,
    model: PreTrainedModel,
) -> Callable[[list[torch.Tensor]], Generator[None, None, None]]:
    """
    Context manager for applying context parallelism to a model. In addition to applying the
    standard context manager to patch SDPA and shard model inputs and buffers along the sequence
    dimension, this context manager also calls into _get_sdpa_context to filter to acceptable SDPA backends.

    Args:
        enabled: Whether context parallel is enabled. Default: False
        world_mesh: Global device mesh.
        model: Model to apply context parallelism to.

    Returns:
        A context manager applying context parallelism if enabled is True. Otherwise a context manager
        disabling the math SDPA backend.

    Raises:
        ValueError: if enabled is True but world_mesh does not contain a "cp" dimension
    """

    if enabled and "cp" not in world_mesh.mesh_dim_names:
        raise ValueError(
            "Context parallel is enabled but no context parallel device mesh is provided."
        )
    # TODO: context parallel for multimodal models requires extra work
    if enabled and not isinstance(model, TransformerDecoder):
        raise ValueError("Context parallel is only supported for text models")
    # TODO: this is a hacky proxy for whether we use flex for chunked attention
    # remove this once flex is supported
    if enabled and any([layer.mask_mod is not None for layer in model.layers]):
        raise ValueError("Context parallel with flex attention is not yet supported")
    model_buffers = list(model.buffers())

    @contextlib.contextmanager
    def context(model_inputs: list[torch.Tensor]):
        # Create context parallel context if enabled
        cp_context = None
        if enabled and any([isinstance(input, BlockMask) for input in model_inputs]):
            raise ValueError(
                "Context parallel with flex attention is not yet supported"
            )
        if enabled:
            set_rotate_method("allgather")
            cp_context = context_parallel(
                world_mesh["cp"],
                buffers=model_inputs + model_buffers,
                buffer_seq_dims=[1] * len(model_inputs) + [0] * len(model_buffers),
                no_restore_buffers=set(model_inputs),
            )

        # Create and enter the train context with the optional cp_context
        sdpa_context = _get_sdpa_context()

        with sdpa_context(cp_context):
            yield

    return context