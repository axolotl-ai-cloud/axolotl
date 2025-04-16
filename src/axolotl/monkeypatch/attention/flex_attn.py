"""Flex attention monkey patch"""

import sys
from typing import Optional, Tuple, Union

import torch
import transformers


def patch_flex_wrapper(**flex_attn_compile_kwargs):
    # TODO remove this patch when transformers#37285 is merged and in a release
    is_torch_2_6 = torch.__version__.startswith("2.6")

    if not is_torch_2_6:
        return

    from torch.nn.attention.flex_attention import flex_attention

    class WrappedFlexAttention:
        """
        We are doing a singleton class so that flex attention is compiled once when it's first called.
        """

        _instance = None
        _is_flex_compiled = False
        _compiled_flex_attention = None

        def __new__(cls, *args, **kwargs):
            if cls._instance is None:
                # Create a new instance if one doesn't already exist
                cls._instance = super().__new__(cls)
            return cls._instance

        @classmethod
        def del_singleton(cls):
            cls._instance = None

        @torch.compiler.disable(recursive=False)
        def __init__(self, training):
            """
            Initialize or update the singleton instance.
            """
            self.training = None
            if not self._is_flex_compiled or training != self.training:
                # In PyTorch 2.6.0, there's a known issue with flex attention compilation which may
                # cause errors. The suggested fix is to compile with "max-autotune-no-cudagraphs"
                # see https://github.com/pytorch/pytorch/issues/146260 for training
                self.training = training
                self._compiled_flex_attention = torch.compile(
                    flex_attention,
                    **flex_attn_compile_kwargs,
                )
                self._is_flex_compiled = True

        def __call__(self):
            return self._compiled_flex_attention

    transformers.integrations.flex_attention.WrappedFlexAttention = WrappedFlexAttention
    setattr(
        sys.modules["transformers.integrations.flex_attention"],
        "WrappedFlexAttention",
        WrappedFlexAttention,
    )


def patch_flex_make_mask():
    is_torch_2_6 = torch.__version__.startswith("2.6")

    if not is_torch_2_6:
        return

    from torch.nn.attention.flex_attention import (
        _DEFAULT_SPARSE_BLOCK_SIZE as flex_default_block_size,
    )
    from torch.nn.attention.flex_attention import (
        BlockMask,
    )
    from torch.nn.attention.flex_attention import (
        create_block_mask as create_block_causal_mask_flex,
    )

    Offset = Union[torch.Tensor, int]

    def patched_make_flex_block_causal_mask(
        attention_mask_2d: torch.Tensor,
        attention_chunk_size: Optional[int] = None,
        query_length=None,
        key_length=None,
        offsets: Optional[Tuple[Offset, Offset]] = None,
    ) -> "BlockMask":
        """
        Create a block causal document mask for a batch of sequences, both packed and unpacked.
        Create Block causal logic and passing it into :func:`torch.nn.attention.flex_attention.create_block_mask`.
        The resultant BlockMask is a compressed representation of the full block causal
        mask. BlockMask is essential for performant computation of flex attention.
        See: https://pytorch.org/blog/flexattention/

        Args:
            attention_mask_2d (torch.Tensor): Attention mask for packed and padded sequences
            of shape (batch_size, total_seq_len). e.g.

            For unpacked sequence:
            [[1, 1, 1, 1, 0, 0, 0],
             [1, 1, 1, 1, 1, 0, 0]]

            For packed sequence:
            [[1, 1, 1, 2, 2, 2, 0],
             [1, 1, 2, 2, 2, 3, 3]]

        Returns:
            BlockMask
        """

        batch_size, total_seq_len = attention_mask_2d.shape
        if not key_length:
            key_length = total_seq_len
        if not query_length:
            query_length = total_seq_len
        attention_mask_2d = torch.nn.functional.pad(
            attention_mask_2d,
            value=0,
            pad=(0, abs(total_seq_len - max(key_length, flex_default_block_size))),
        )
        device = attention_mask_2d.device
        document_ids = attention_mask_2d.clone()

        if attention_chunk_size is not None:
            # we create an arange, then we just // by chunk size to get [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
            chunk_idxs = (document_ids.clone().fill_(1).cumsum(-1) - 1) // (
                attention_chunk_size
            )

        # Instead of passing a tensor mask, flex attention requires a mask_mod function
        # that determines which elements of QK^T should be included in the attention
        # computation prior to the softmax. For sample packing, we need both the
        # logic for both causal mask and document mask. See PyTorch's official
        # blog post for more details: https://pytorch.org/blog/flexattention/#mask-mods
        def causal_mask_mod(
            batch_idx, head_idx, q_idx, kv_idx
        ):  # pylint: disable=unused-argument
            """
            Defines the logic of a block causal mask by combining both a standard causal mask
            and a block diagonal document mask.

            See :func:`~torchtune.modules.attention_utils.create_block_causal_mask`
            for an illustration.
            """
            causal_mask = q_idx >= kv_idx  # not valid when decoding
            document_mask = (
                document_ids[batch_idx, q_idx] == document_ids[batch_idx, kv_idx]
            )
            padding_mask = attention_mask_2d[batch_idx, q_idx] > 0
            final_mask = causal_mask & padding_mask & document_mask
            return final_mask

        def chunk_causal_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
            """
            Combines the chunk mask with the causal mask for chunked attention.
            """
            chunk_mask = chunk_idxs[batch_idx, q_idx] == chunk_idxs[batch_idx, kv_idx]
            causal_doc_mask = causal_mask_mod(batch_idx, head_idx, q_idx, kv_idx)
            return chunk_mask & causal_doc_mask

        mask_mod_maybe_combined = (
            causal_mask_mod if attention_chunk_size is None else chunk_causal_mask_mod
        )

        if offsets is not None:
            q_offset = offsets[0]
            kv_offset = offsets[1]

            def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
                offset_q = q_idx + q_offset
                offset_kv = kv_idx + kv_offset
                return mask_mod_maybe_combined(batch_idx, head_idx, offset_q, offset_kv)

        else:
            mask_mod = mask_mod_maybe_combined
        return create_block_causal_mask_flex(
            mask_mod=mask_mod,
            B=batch_size,
            H=None,  # attention head
            Q_LEN=query_length,
            KV_LEN=key_length,
            device=device,
            _compile=True,
        )

    for n in tuple(sys.modules):
        if ".modeling_" in n:
            if hasattr(sys.modules[n], "make_flex_block_causal_mask"):
                sys.modules[n].make_flex_block_causal_mask = (
                    patched_make_flex_block_causal_mask
                )
                setattr(
                    sys.modules[n],
                    "make_flex_block_causal_mask",
                    patched_make_flex_block_causal_mask,
                )

    transformers.integrations.flex_attention.make_flex_block_causal_mask = (
        patched_make_flex_block_causal_mask
    )
