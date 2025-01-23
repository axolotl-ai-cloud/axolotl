"""
Patched LlamaAttention to use torch.nn.functional.scaled_dot_product_attention
"""


from axolotl.monkeypatch.utils import (
    patched_prepare_4d_causal_attention_mask,
    patched_prepare_4d_causal_attention_mask_for_sdpa,
)


def hijack_llama_prepare_4d_mask():
    from typing import Optional

    import torch
    from transformers import modeling_attn_mask_utils
    from transformers.models.llama.modeling_llama import LlamaModel

    # from transformers.models.llama.modeling_llama.LlamaModel import (
    #     _prepare_4d_causal_attention_mask_with_cache_position,
    # )
    from transformers.utils import is_torch_bf16_gpu_available

    from axolotl.monkeypatch.utils import mask_2d_to_4d

    # modeling_llama._prepare_4d_causal_attention_mask_for_sdpa = (  # pylint: disable=protected-access
    #    patched_prepare_4d_causal_attention_mask_for_sdpa
    # )

    @staticmethod
    def llama_patched_prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: Optional[torch.Tensor], **kwargs
    ):
        dtype = torch.bfloat16 if is_torch_bf16_gpu_available() else torch.float32
        # return LlamaModel._prepare_4d_causal_attention_mask_with_cache_position(
        #     mask_2d_to_4d(attention_mask, dtype=dtype), **kwargs
        # )
        return mask_2d_to_4d(attention_mask, dtype=dtype).bool()

    LlamaModel._prepare_4d_causal_attention_mask_with_cache_position = (  # pylint: disable=protected-access
        llama_patched_prepare_4d_causal_attention_mask_with_cache_position
    )
    modeling_attn_mask_utils._prepare_4d_causal_attention_mask_for_sdpa = (  # pylint: disable=protected-access
        patched_prepare_4d_causal_attention_mask_for_sdpa
    )
    LlamaModel._prepare_4d_causal_attention_mask = (  # pylint: disable=protected-access
        patched_prepare_4d_causal_attention_mask
    )
    modeling_attn_mask_utils._prepare_4d_causal_attention_mask = (  # pylint: disable=protected-access
        patched_prepare_4d_causal_attention_mask
    )
