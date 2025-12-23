"""
monkeypatch for flex + packing
"""

import sys
from typing import Callable, Optional, Union

import torch
from torch.nn.attention.flex_attention import BlockMask
from transformers import Cache, PretrainedConfig
from transformers.masking_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
    _preprocess_mask_arguments,
    and_masks,
    causal_mask_function,
    or_masks,
    packed_sequence_mask_function,
    sliding_window_causal_mask_function,
)
from transformers.utils import is_torch_greater_or_equal

_is_torch_greater_or_equal_than_2_6 = is_torch_greater_or_equal("2.6", accept_dev=True)


def create_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor] = None,
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
) -> Optional[Union[torch.Tensor, BlockMask]]:
    """
    Create a standard causal mask based on the attention implementation used (stored in the config). If `past_key_values`
    has an HybridCache structure, this function will return the mask corresponding to one of the "full_attention" layers (to align
    to what is needed in the `modeling_xxx.py` files).

    Args:
        config (`PretrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional):
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
    """
    # If we have an HybridCache structure, here we want to create the mask for the full layers
    if (
        past_key_values
        and hasattr(past_key_values, "is_sliding")
        and False in past_key_values.is_sliding
    ):
        layer_idx = past_key_values.is_sliding.index(False)
    else:
        layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset = (
        _preprocess_mask_arguments(
            config,
            input_embeds,
            attention_mask,
            cache_position,
            past_key_values,
            position_ids,
            layer_idx,
        )
    )
    if early_exit:
        return attention_mask

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype

    # Default mask function - following upstream: use causal_mask_function as base
    mask_factory_function = causal_mask_function

    # If we detected packing from position_ids, use and_masks to combine with packed sequence mask
    # This follows upstream transformers behavior
    if packed_sequence_mask is not None and _is_torch_greater_or_equal_than_2_6:
        mask_factory_function = and_masks(
            mask_factory_function, packed_sequence_mask_function(packed_sequence_mask)
        )

    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    # Do not allow skip if we are compiling (this is to match BC)
    allow_is_causal_skip = (
        not past_key_values.is_compileable if past_key_values is not None else True
    )

    # Allow slight deviations from causal mask
    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError(
                "Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6"
            )
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError(
                "Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6"
            )
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False

    # We now create the mask
    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
    )
    return causal_mask


def create_sliding_window_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor] = None,
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
) -> Optional[Union[torch.Tensor, BlockMask]]:
    """
    Create a sliding window causal mask based on the attention implementation used (stored in the config). This type
    of attention pattern was mostly democratized by Mistral. If `past_key_values` has an HybridCache structure, this
    function will return the mask corresponding to one of the "sliding_attention" layers (to align to what is needed in the
    `modeling_xxx.py` files).

    Args:
        config (`PretrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the sliding causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the sliding causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the sliding causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the sliding causal one, for example for image tokens handling.
    """
    # If we have an HybridCache structure, here we want to create the mask for the sliding layers
    if (
        past_key_values
        and hasattr(past_key_values, "is_sliding")
        and True in past_key_values.is_sliding
    ):
        layer_idx = past_key_values.is_sliding.index(True)
    else:
        layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset = (
        _preprocess_mask_arguments(
            config,
            input_embeds,
            attention_mask,
            cache_position,
            past_key_values,
            position_ids,
            layer_idx,
        )
    )
    if early_exit:
        return attention_mask

    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        raise ValueError(
            "Could not find a `sliding_window` argument in the config, or it is not set"
        )

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    # Start with sliding window causal mask function
    mask_factory_function = sliding_window_causal_mask_function(sliding_window)
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    # Do not allow skip if we are compiling (this is to match BC)
    allow_is_causal_skip = (
        not past_key_values.is_compileable if past_key_values is not None else True
    )

    # If we detected packing format, combine with packed sequence mask
    if packed_sequence_mask is not None and _is_torch_greater_or_equal_than_2_6:
        mask_factory_function = and_masks(
            mask_factory_function, packed_sequence_mask_function(packed_sequence_mask)
        )
        allow_is_causal_skip = False

    # Allow slight deviations from sliding causal mask
    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError(
                "Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6"
            )
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError(
                "Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6"
            )
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False

    # We now create the mask
    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
        local_size=sliding_window,  # Additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
    )
    return causal_mask


def patch_create_causal_mask(model_type):
    import transformers.masking_utils

    transformers.masking_utils.create_causal_mask = create_causal_mask

    if model_type:
        try:
            # Dynamically import the module and attention class
            module_path = f"transformers.models.{model_type}.modeling_{model_type}"
            module = __import__(module_path)
            module.create_causal_mask = create_causal_mask
            del sys.modules[module_path]
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Could not import attention class for model_type: {model_type}. "
                f"Error: {str(e)}"
            ) from e


def patch_create_sliding_window_causal_mask(model_type):
    import transformers.masking_utils

    transformers.masking_utils.create_sliding_window_causal_mask = (
        create_sliding_window_causal_mask
    )

    if model_type:
        try:
            # Dynamically import the module and attention class
            module_path = f"transformers.models.{model_type}.modeling_{model_type}"
            module = __import__(module_path)
            module.create_sliding_window_causal_mask = create_sliding_window_causal_mask
            del sys.modules[module_path]
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Could not import attention class for model_type: {model_type}. "
                f"Error: {str(e)}"
            ) from e
