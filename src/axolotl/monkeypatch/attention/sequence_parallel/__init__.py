from enum import Enum
from functools import partial

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from yunchang import set_seq_parallel_pg, EXTRACT_FUNC_DICT

from axolotl.utils.distributed import get_world_size, get_rank


class USPRingAttnType(Enum):
    BASIC = "basic"
    ZIGZAG = "zigzag"
    STRIPE = "stripe"

def apply_usp_attn_patch(ring_impl_type: USPRingAttnType):
    from axolotl.monkeypatch.attention.sequence_parallel.usp import build_usp_fa_forward

    fa_forward = build_usp_fa_forward(ring_impl_type)
    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = fa_forward

def get_extract_fn(ring_impl_type: USPRingAttnType, sp_ulysses_degree: int):
    fn = EXTRACT_FUNC_DICT["basic"]
    if ring_impl_type.value in EXTRACT_FUNC_DICT.keys():
        fn = EXTRACT_FUNC_DICT[ring_impl_type.value]

    # map bad key upstream
    elif ring_impl_type == USPRingAttnType.STRIPE:
        fn = EXTRACT_FUNC_DICT["strip"]

    world_size = get_world_size()
    rd = world_size // sp_ulysses_degree

    return partial(fn, rank=get_rank(), world_size=world_size, rd=rd, ud=sp_ulysses_degree)

def set_usp_parallel_group(sp_ulysses_degree):
    """
    setup distributed parallel group for USP attention
    make sure this gets called before building any USP attention modules
    :param sp_ulysses_degree:
    :return:
    """
    world_size = get_world_size()
    rank = get_rank()
    sp_ring_degree = world_size // sp_ulysses_degree
    set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, world_size)
