"""CPU-only tests for the Mistral-Large-4 kernel integration (adapter + config + dispatch)."""

import torch
import torch.nn as nn

from axolotl.integrations.kernels.adapters import _all_adapters
from axolotl.integrations.kernels.adapters.mistral_large4 import MistralLarge4Adapter
from axolotl.integrations.kernels.args import KernelsArgs
from axolotl.integrations.kernels.libs.mistral_large4.lora_attn import (
    MLA_PROJ_SUFFIXES,
    _quant_kind,
)
from axolotl.utils.dict import DictDefault


def test_adapter_registered():
    names = [a.name for a in _all_adapters()]
    assert "mistral_large4" in names


def test_adapter_matches_by_flag():
    a = MistralLarge4Adapter()
    assert a.matches(DictDefault({"use_mistral_large4_kernels": True})) is True
    # no flag + unreachable base_model -> False (model_type probe fails closed)
    assert a.matches(DictDefault({"base_model": "/does/not/exist"})) is False


def test_config_flags_accepted():
    a = KernelsArgs(
        use_mistral_large4_kernels=True,
        mistral_large4_attn_lora_kernel=True,
        mistral_large4_attn_native_fp8=False,
        mistral_large4_attn_quantization="nvfp4",
    )
    assert a.use_mistral_large4_kernels is True
    assert a.mistral_large4_attn_native_fp8 is False
    assert a.mistral_large4_attn_quantization == "nvfp4"


def test_mla_proj_suffixes():
    # the four MLA projections the kernel adapts (clean q_proj, no k/v_proj)
    assert set(MLA_PROJ_SUFFIXES) == {
        "self_attn.q_proj",
        "self_attn.kv_a_proj_with_mqa",
        "self_attn.kv_b_proj",
        "self_attn.o_proj",
    }


def test_quant_kind_bf16_and_fp8():
    # bf16 base
    lin = nn.Linear(128, 64)
    assert _quant_kind(lin) == "bf16"
    # raw block-fp8 base (weight float8 + weight_scale_inv)
    fp8 = nn.Linear(128, 64)
    fp8.weight = nn.Parameter(
        torch.zeros(64, 128, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    fp8.weight_scale_inv = torch.ones(1, 1)
    fp8.block_size = [128, 128]
    assert _quant_kind(fp8) == "fp8"
