"""CPU tests for the Mistral-Large-4 kernel integration on transformers' native modeling
(adapter matching, config, attention-LoRA dispatch)."""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("triton")  # libs/dsv4/__init__ imports the fused Triton kernels

from transformers import (  # noqa: E402
    AutoModelForImageTextToText,
    MistralLarge4Config,
)

from axolotl.integrations.kernels.adapters import (  # noqa: E402
    _all_adapters,
    get_active_adapters,
    mistral_large4 as ml4_adapter,
)
from axolotl.integrations.kernels.adapters.mistral_large4 import (  # noqa: E402
    MistralLarge4Adapter,
)
from axolotl.integrations.kernels.args import KernelsArgs  # noqa: E402
from axolotl.integrations.kernels.libs.mistral_large4.lora_attn import (  # noqa: E402
    MLA_PROJ_SUFFIXES,
    _quant_kind,
    patch_mistral_large4_attn_lora,
)
from axolotl.utils.dict import DictDefault  # noqa: E402

TEXT = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    moe_intermediate_size=32,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_routed_experts=4,
    num_experts_per_tok=2,
    kv_lora_rank=32,
    qk_rope_head_dim=16,
    qk_nope_head_dim=16,
    v_head_dim=16,
    index_topk=8,
    index_head_dim=32,
    index_n_heads=2,
    first_k_dense_replace=1,
    # layer 2 reuses layer 0's KV latent, so it has no kv_a_proj_with_mqa
    kv_source_layers=[0, 1, 0, 3],
    max_position_embeddings=4096,
    pad_token_id=0,
)
VISION = dict(
    model_type="pixtral",
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=2,
    patch_size=14,
    image_size=28,
    head_dim=16,
)
MODELOPT_NVFP4 = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"}
BLOCK_FP8 = {"quant_method": "fp8", "weight_block_size": [128, 128]}


def tiny_config(**kwargs) -> MistralLarge4Config:
    cfg = MistralLarge4Config(text_config=TEXT, vision_config=VISION, **kwargs)
    cfg._attn_implementation = "eager"
    return cfg


def _saved(tmp_path, config) -> str:
    config.save_pretrained(tmp_path)
    return str(tmp_path)


def test_adapter_registered():
    names = [a.name for a in _all_adapters()]
    assert "mistral_large4" in names


def test_adapter_matches_native_model_type(tmp_path):
    cfg = DictDefault({"base_model": _saved(tmp_path, tiny_config())})
    assert MistralLarge4Adapter().matches(cfg) is True
    assert [a.name for a in get_active_adapters(cfg)] == ["mistral_large4"]


def test_adapter_skips_text_only_and_unreachable_configs(tmp_path):
    text_only = DictDefault({"base_model": _saved(tmp_path, tiny_config().text_config)})
    assert MistralLarge4Adapter().matches(text_only) is False
    missing = DictDefault({"base_model": "/does/not/exist"})
    assert MistralLarge4Adapter().matches(missing) is False


def test_nvfp4_checkpoint_owned_by_mistral_large4_adapter(tmp_path):
    path = _saved(tmp_path, tiny_config(quantization_config=MODELOPT_NVFP4))
    cfg = DictDefault({"base_model": path, "use_scattermoe": True})
    assert [a.name for a in get_active_adapters(cfg)] == ["mistral_large4"]


@pytest.mark.parametrize(
    "quantization_config,flag,expected",
    [
        (None, None, True),
        (BLOCK_FP8, None, True),
        (BLOCK_FP8, False, False),
        (MODELOPT_NVFP4, None, False),
        (MODELOPT_NVFP4, True, True),
    ],
)
def test_attn_lora_kernel_default(tmp_path, quantization_config, flag, expected):
    extra = {"quantization_config": quantization_config} if quantization_config else {}
    cfg = DictDefault(
        {
            "base_model": _saved(tmp_path, tiny_config(**extra)),
            "adapter": "lora",
            "use_scattermoe": True,
            "mistral_large4_attn_lora_kernel": flag,
        }
    )
    assert MistralLarge4Adapter()._attn_lora_kernel_enabled(cfg) is expected


def test_attn_lora_kernel_needs_lora(tmp_path):
    cfg = DictDefault(
        {"base_model": _saved(tmp_path, tiny_config()), "adapter": "qlora"}
    )
    assert MistralLarge4Adapter()._attn_lora_kernel_enabled(cfg) is False


def test_remote_code_checkpoint_rejected(monkeypatch):
    config = tiny_config()
    config.auto_map = {"AutoConfig": "configuration_mistral.Config"}
    monkeypatch.setattr(ml4_adapter, "load_base_model_config", lambda cfg: config)
    cfg = DictDefault({"base_model": "remote/ml4", "trust_remote_code": True})
    with pytest.raises(ValueError, match="trust_remote_code"):
        MistralLarge4Adapter().pre_model_load(cfg)


def test_config_flags_accepted():
    a = KernelsArgs(
        mistral_large4_attn_lora_kernel=True,
        mistral_large4_attn_native_fp8=False,
        mistral_large4_attn_quantization="nvfp4",
    )
    assert a.mistral_large4_attn_lora_kernel is True
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


def test_attn_patch_skips_vision_tower():
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(tiny_config())
    # bare "o_proj" / "q_proj" also match the Pixtral tower's attention.{q,o}_proj
    peft_model = get_peft_model(
        model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "o_proj"])
    )
    lora_names = [n for n, m in peft_model.named_modules() if hasattr(m, "lora_A")]
    assert any("vision_tower" in n for n in lora_names)

    assert patch_mistral_large4_attn_lora(peft_model) == 2 * TEXT["num_hidden_layers"]
    patched = [
        n for n, m in peft_model.named_modules() if hasattr(m, "_ml4_orig_forward")
    ]
    assert all(".language_model.layers." in n for n in patched)


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


MLA_TARGETS = ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]


def _tiny_peft_model(targets, lora_dropout=0.0):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(tiny_config()).float()
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=4, lora_alpha=8, lora_dropout=lora_dropout, target_modules=targets
        ),
    )
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(param, std=0.02)
    return peft_model


def _loss_and_grads(model, input_ids):
    model.zero_grad(set_to_none=True)
    logits = model(input_ids=input_ids).logits
    logits.pow(2).mean().backward()
    grads = {
        n: p.grad.clone()
        for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }
    return logits.detach(), grads


@pytest.mark.parametrize(
    "targets,n_patched",
    [
        (MLA_TARGETS, 15),  # kv_a_proj_with_mqa is absent on the KV-shared layer
        (["kv_b_proj"], 4),
        (["q_proj", "kv_a_proj_with_mqa", "o_proj"], 11),
    ],
)
def test_attn_lora_kernel_matches_peft(targets, n_patched):
    model = _tiny_peft_model(targets)
    input_ids = torch.randint(1, TEXT["vocab_size"], (2, 12))
    ref_logits, ref_grads = _loss_and_grads(model, input_ids)

    assert patch_mistral_large4_attn_lora(model) == n_patched
    logits, grads = _loss_and_grads(model, input_ids)

    torch.testing.assert_close(logits, ref_logits, rtol=1e-5, atol=1e-6)
    assert grads.keys() == ref_grads.keys()
    for name, ref in ref_grads.items():
        torch.testing.assert_close(grads[name], ref, rtol=1e-4, atol=1e-7, msg=name)


@pytest.mark.parametrize("lead", [(6,), (2, 6), (2, 1, 6)])
def test_attn_lora_generic_path_keeps_input_shape(lead):
    model = _tiny_peft_model(["kv_b_proj"])
    patch_mistral_large4_attn_lora(model)
    proj = model.base_model.model.model.language_model.layers[0].self_attn.kv_b_proj

    x = torch.randn(*lead, TEXT["kv_lora_rank"], requires_grad=True)
    out = proj(x)
    out.pow(2).sum().backward()
    got = (out.detach(), x.grad.clone(), proj.lora_A["default"].weight.grad.clone())

    x.grad = None
    proj.zero_grad(set_to_none=True)
    ref = proj._ml4_orig_forward(x)
    ref.pow(2).sum().backward()
    want = (ref.detach(), x.grad, proj.lora_A["default"].weight.grad)

    assert got[0].shape == (*lead, proj.out_features)
    for g, w in zip(got, want, strict=True):
        torch.testing.assert_close(g, w, rtol=1e-5, atol=1e-6)
