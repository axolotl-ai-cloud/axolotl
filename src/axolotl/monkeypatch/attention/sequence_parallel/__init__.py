from enum import Enum

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

class USPRingAttnType(Enum):
    BASIC = "basic"
    ZIGZAG = "zigzag"
    STRIDE = "stride"

def patch_seq_parallel():
    ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
    pass

def apply_usp_attn_patch(ring_impl_type: USPRingAttnType):
    from axolotl.monkeypatch.attention.sequence_parallel.usp import build_usp_fa_forward

    fa_forward = build_usp_fa_forward(ring_impl_type)
    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = fa_forward
