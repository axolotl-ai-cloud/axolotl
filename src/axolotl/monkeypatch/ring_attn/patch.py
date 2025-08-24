"""Ring attention group registration and flash attention patching.

Make use of the `ring-flash-attn` (https://github.com/zhuzilin/ring-flash-attention)
package, specifically the `hf_adapter.substitute_hf_flash_attn` function to patch in
their sequence parallel version of Flash Attention 2.

We also provide some patches for accelerate functions to prepare the dataloader for
sequence parallelism training.
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh

try:
    from transformers.modeling_flash_attention_utils import _flash_supports_window
except ImportError:
    try:
        from transformers.modeling_flash_attention_utils import (
            _flash_supports_window_size as _flash_supports_window,
        )
    except ImportError:
        _flash_supports_window = True

from axolotl.monkeypatch.utils import get_cu_seqlens_from_pos_ids
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.enums import RingAttnFunc

LOG = get_logger(__name__)

RING_ATTN_GROUP = None


def get_ring_attn_group() -> dist.ProcessGroup:
    """Getter for ring attention group on this rank."""
    if RING_ATTN_GROUP is None:
        raise RuntimeError("register_ring_attn_from_device_mesh() not yet called")
    return RING_ATTN_GROUP


def set_ring_attn_group(ring_attn_group: dist.ProcessGroup | None):
    """Setter for ring attention group on this rank."""
    global RING_ATTN_GROUP
    RING_ATTN_GROUP = ring_attn_group


def create_ring_flash_attention_forward(
    process_group: dist.ProcessGroup, heads_k_stride: int
):
    from ring_flash_attn import llama3_flash_attn_varlen_func
    from ring_flash_attn.adapters.hf_adapter import DATA_PARAMS

    def _flash_attention_forward_v3(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
        is_causal: bool,
        dropout: float = 0.0,
        position_ids: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        sliding_window: Optional[int] = None,
        use_top_left_mask: bool = False,
        softcap: Optional[float] = None,
        deterministic: bool = None,
        cu_seq_lens_q: Optional[torch.LongTensor] = None,
        cu_seq_lens_k: Optional[torch.LongTensor] = None,
        max_length_q: Optional[int] = None,
        max_length_k: Optional[int] = None,
        target_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
        **kwargs,
    ):
        if not use_top_left_mask:
            causal = is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__.
            causal = is_causal and query_length != 1

        # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
        use_sliding_windows = (
            _flash_supports_window
            and sliding_window is not None
            and key_states.shape[1] > sliding_window
        )
        flash_kwargs = (
            {"window_size": (sliding_window, sliding_window)}
            if use_sliding_windows
            else {}
        )

        if deterministic is None:
            deterministic = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
        flash_kwargs["deterministic"] = deterministic
        assert softcap is None, (
            "llama3_flash_attn_varlen_func does not support softcap yet."
        )
        # flash_kwargs["softcap"] = softcap
        flash_kwargs["group"] = process_group

        # not sure why attention_mask can be not None...
        assert causal, "only causal attention is supported yet."
        batch_size = query_states.size(0)
        assert batch_size == 1, "varlen data should be processed in advance."

        attn_output = llama3_flash_attn_varlen_func(
            query_states.squeeze(dim=0),
            key_states.squeeze(dim=0),
            value_states.squeeze(dim=0),
            cu_seqlens_q=DATA_PARAMS["cu_seqlens_q"],
            cu_seqlens_k=DATA_PARAMS["cu_seqlens_k"],
            max_seqlen_q=DATA_PARAMS["max_seqlen_q"],
            max_seqlen_k=DATA_PARAMS["max_seqlen_k"],
            heads_k_stride=heads_k_stride,
            local_k_slice=DATA_PARAMS["local_k_slice"],
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )

        attn_output = attn_output.unsqueeze(dim=0)

        return attn_output

    return [
        _flash_attention_forward_v3,
    ]


def register_ring_attn_from_device_mesh(
    device_mesh: "DeviceMesh",
    context_parallel_dim: tuple[str, ...],
    heads_k_stride: int | None,
    ring_attn_func: RingAttnFunc | None,
):
    """Create ring attention group using DeviceMesh and substitute flash attn with ring flash attn.

    Args:
        device_mesh: DeviceMesh object containing the parallelism topology.
        context_parallel_dim: Name of the sequence parallel dimension in the device mesh.
        heads_k_stride: Sequence parallelism K head stride size. Passed through to
            `varlen_llama3` `ring_flash_attn` implementation.
        ring_attn_func: `ring_flash_attn` ring attention implemention. If sample
            packing is enabled, it must be a `varlen` function; otherwise, it must be a
            `batch` function.
    """
    rank = dist.get_rank()

    LOG.info(
        f"Enabling ring attention sequence parallelism using DeviceMesh "
        f"dimension '{context_parallel_dim}'",
        main_process_only=True,
    )

    # Extract the sequence parallel submesh
    try:
        sequence_mesh = device_mesh[context_parallel_dim]
    except (KeyError, IndexError) as e:
        raise ValueError(
            f"Dimension '{context_parallel_dim}' not found in device_mesh. "
            f"Available dimensions: {device_mesh.mesh_dim_names}"
        ) from e

    # Get the process group for context parallelism
    sequence_pg = sequence_mesh.get_group()
    context_parallel_size = sequence_mesh.size()

    if rank == 0:
        LOG.info(
            f"Sequence parallel degree: {context_parallel_size}, "
            f"mesh shape: {sequence_mesh.mesh.shape}"
        )

    # Log which ranks are in the current process group
    if sequence_pg != dist.GroupMember.WORLD:
        ranks_in_group = dist.get_process_group_ranks(sequence_pg)
        LOG.info(f"Current sequence parallel group ranks: {ranks_in_group}")

    # Set the ring attention group
    set_ring_attn_group(sequence_pg)

    if ring_attn_func is RingAttnFunc.VARLEN_LLAMA3:
        # fmt: off
        import ring_flash_attn.adapters.hf_adapter

        from ring_flash_attn.adapters.hf_adapter import (  # isort: skip
            create_ring_flash_attention_forward as create_ring_flash_attention_forward_orig,
        )

        create_ring_flash_attention_forward_orig = (  # noqa: F811,F841
            create_ring_flash_attention_forward
        )
        ring_flash_attn.adapters.hf_adapter.create_ring_flash_attention_forward = create_ring_flash_attention_forward
        # fmt: on

        ring_flash_attn.adapters.hf_adapter.substitute_hf_flash_attn(
            process_group=get_ring_attn_group(), heads_k_stride=heads_k_stride or 1
        )
    elif ring_attn_func is RingAttnFunc.BATCH_RING:
        from axolotl.monkeypatch.ring_attn.adapters.batch import (
            substitute_hf_flash_attn,
        )

        substitute_hf_flash_attn(
            process_group=get_ring_attn_group(),
            ring_attn_func=ring_attn_func,
        )


def update_ring_attn_params(position_ids: torch.Tensor | None):
    """
    Calculate the cumulative sequence lengths for the current forward pass and pass the
    value to the substituted `ring_flash_attn`.

    Args:
        position_ids: Optional tensor of position IDs (for sample packed data).
    """
    from ring_flash_attn import update_ring_flash_attn_params

    cu_seqlens, _ = get_cu_seqlens_from_pos_ids(position_ids)
    cu_seqlens = cu_seqlens.squeeze().to(device=torch.cuda.current_device())
    update_ring_flash_attn_params(cu_seqlens, get_ring_attn_group())
