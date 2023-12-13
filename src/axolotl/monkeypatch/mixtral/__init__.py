"""
Patches to support multipack for mixtral
"""
import transformers


def replace_mixtral_attn_with_multipack_flash_attn():
    from .modeling_mixtral import (
        MixtralMultipackFlashAttention2,
        mixtral_decoder_layer_forward,
        mixtral_model_forward,
    )

    transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer.forward = (
        mixtral_decoder_layer_forward
    )
    transformers.models.mixtral.modeling_mixtral.MixtralModel.forward = (
        mixtral_model_forward
    )
    transformers.models.mixtral.modeling_mixtral.MISTRAL_ATTENTION_CLASSES[
        "flash_attention_2"
    ] = MixtralMultipackFlashAttention2
