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
    is_mla_proj,
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


SAC_MATMUL_SAVING = [
    {"save": ["attention", "aten::mm"]},
    {"save": ["attention"], "save_matmul_min_k": 4096},
    {"save": ["attention"], "save_modules": ["o_proj"]},
    {"save": ["attention"], "save_modules": ["self_attn.kv_b_proj"]},
    {"save": ["attention"], "save_modules": ["layers"]},
]
SAC_NO_CONFLICT = [
    {"save": ["attention"]},
    {"save": ["attention"], "save_modules": ["down_proj"]},
]


@pytest.mark.parametrize("sac", SAC_MATMUL_SAVING)
def test_attn_lora_kernel_defers_to_selective_checkpointing(tmp_path, monkeypatch, sac):
    from axolotl.monkeypatch.accelerate import fsdp2_quantized

    monkeypatch.setattr(fsdp2_quantized, "_FP32_SHARD_CLASS_NAMES", set())
    base = {
        "base_model": _saved(tmp_path, tiny_config()),
        "adapter": "lora",
        "selective_checkpointing": sac,
    }
    adapter = MistralLarge4Adapter()
    assert adapter._attn_lora_kernel_enabled(DictDefault(base)) is False

    forced = DictDefault({**base, "mistral_large4_attn_lora_kernel": True})
    with pytest.raises(ValueError, match="selective_checkpointing"):
        adapter.pre_model_load(forced)
    off = DictDefault({**base, "mistral_large4_attn_lora_kernel": False})
    adapter.pre_model_load(off)


@pytest.mark.parametrize("sac", SAC_NO_CONFLICT)
def test_attn_lora_kernel_keeps_attention_only_selective_checkpointing(tmp_path, sac):
    cfg = DictDefault(
        {
            "base_model": _saved(tmp_path, tiny_config()),
            "adapter": "lora",
            "selective_checkpointing": sac,
        }
    )
    assert MistralLarge4Adapter()._attn_lora_kernel_enabled(cfg) is True


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


@pytest.mark.parametrize(
    "key,value",
    [
        ("peft_use_dora", True),
        ("load_in_8bit", True),
        ("load_in_4bit", True),
        ("lora_merge_aware", "bf16"),
    ],
)
@pytest.mark.parametrize(
    "overrides,rejected",
    [
        ({}, True),
        ({"mistral_large4_attn_quantization": "none"}, False),
    ],
)
def test_attn_quantization_rejects_incompatible_bases(
    tmp_path, monkeypatch, key, value, overrides, rejected
):
    from axolotl.monkeypatch.accelerate import fsdp2_quantized

    monkeypatch.setattr(fsdp2_quantized, "_FP32_SHARD_CLASS_NAMES", set())
    cfg = DictDefault(
        {
            "base_model": _saved(tmp_path, tiny_config()),
            "adapter": "lora",
            key: value,
            "mistral_large4_attn_quantization": "nvfp4",
            **overrides,
        }
    )
    if rejected:
        with pytest.raises(ValueError, match=key):
            MistralLarge4Adapter().pre_model_load(cfg)
    else:
        MistralLarge4Adapter().pre_model_load(cfg)


@pytest.mark.parametrize(
    "overrides",
    [
        {"mistral_large4_attn_lora_kernel": False},
        {"adapter": "qlora", "load_in_4bit": True},
        {"quantization_config": MODELOPT_NVFP4},
    ],
)
def test_attn_quantization_rejects_attn_lora_kernel_off(tmp_path, overrides):
    overrides = dict(overrides)
    extra = overrides.pop("quantization_config", None)
    config = tiny_config(quantization_config=extra) if extra else tiny_config()
    cfg = DictDefault(
        {
            "base_model": _saved(tmp_path, config),
            "adapter": "lora",
            "use_scattermoe": True,
            "mistral_large4_attn_quantization": "nvfp4",
            **overrides,
        }
    )
    with pytest.raises(ValueError, match="mistral_large4_attn_lora_kernel: true"):
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


@pytest.mark.parametrize("value", ["fp4", "int8", "nvfp8"])
def test_attn_quantization_rejects_unknown_format(value):
    with pytest.raises(ValueError, match="mistral_large4_attn_quantization"):
        KernelsArgs.model_validate({"mistral_large4_attn_quantization": value})


@pytest.mark.parametrize("value", [None, "none", "nvfp4", "MXFP4", "mxfp8"])
def test_attn_quantization_accepts_known_formats(value):
    args = KernelsArgs.model_validate({"mistral_large4_attn_quantization": value})
    assert args.mistral_large4_attn_quantization == value


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
    # Pixtral names its attention differently today; the anchor still keeps a self_attn tower out
    assert not is_mla_proj("model.vision_tower.transformer.layers.0.self_attn.q_proj")


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


def test_attn_lora_leaves_int8_base_to_peft():
    from peft import LoraConfig, get_peft_model

    lin = nn.Linear(64, 32, bias=False)
    # bnb 8-bit (adapter: lora + load_in_8bit): int8 storage the fused kernels cannot dequantize
    lin.weight = nn.Parameter(
        torch.randint(-127, 127, (32, 64), dtype=torch.int8), requires_grad=False
    )
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = lin
    root = nn.Module()
    root.language_model = nn.Module()
    root.language_model.layers = nn.ModuleList([layer])
    model = get_peft_model(root, LoraConfig(r=4, target_modules=["q_proj"]))
    proj = model.base_model.model.language_model.layers[0].self_attn.q_proj

    assert patch_mistral_large4_attn_lora(model) == 1
    peft_out = torch.zeros(1)
    proj._ml4_orig_forward = lambda x, *args, **kwargs: peft_out
    assert proj(torch.randn(2, 3, 64, dtype=torch.bfloat16)) is peft_out


def test_quantize_attn_bases_skips_int8_base():
    from peft import LoraConfig, get_peft_model

    from axolotl.integrations.kernels.libs.mistral_large4.lora_lowbit import (
        quantize_attn_bases,
    )

    lin = nn.Linear(64, 32, bias=False)
    # bnb 8-bit storage: casting it to bf16 without its SCB scales would requantize garbage
    int8 = torch.randint(-127, 127, (32, 64), dtype=torch.int8)
    lin.weight = nn.Parameter(int8.clone(), requires_grad=False)
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = lin
    root = nn.Module()
    root.language_model = nn.Module()
    root.language_model.layers = nn.ModuleList([layer])
    model = get_peft_model(root, LoraConfig(r=4, target_modules=["q_proj"]))

    assert quantize_attn_bases(model, "mxfp8", MLA_PROJ_SUFFIXES) == 0
    assert torch.equal(lin.weight, int8)


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


def test_attn_patch_keeps_lora_merge_aware_forward():
    from axolotl.monkeypatch.lora_merge_aware import install_lora_merge_aware_linears

    model = _tiny_peft_model(MLA_TARGETS)
    installed, skipped = install_lora_merge_aware_linears(model, "bf16")
    assert installed >= 15 and not skipped
    input_ids = torch.randint(1, TEXT["vocab_size"], (2, 12))
    ref_logits = model(input_ids=input_ids).logits.detach()

    assert patch_mistral_large4_attn_lora(model) == 0
    assert not any(hasattr(m, "_ml4_orig_forward") for m in model.modules())
    torch.testing.assert_close(model(input_ids=input_ids).logits.detach(), ref_logits)


# 128-aligned dims so q_proj / o_proj take the blockwise-fp8 path. Every expert per token and
# every key per query: bf16 rounding can't flip a discrete MoE / DSA top-k choice.
FP8_TEXT = dict(
    TEXT,
    num_experts_per_tok=TEXT["n_routed_experts"],
    index_topk=2048,
    hidden_size=256,
    intermediate_size=256,
    moe_intermediate_size=128,
    num_attention_heads=2,
    num_key_value_heads=2,
    kv_lora_rank=128,
    qk_rope_head_dim=64,
    qk_nope_head_dim=64,
    v_head_dim=128,
    index_head_dim=128,
)
FP8_BLOCK = [128, 128]


def _dequant_linear(
    input,
    weight,
    weight_scale_inv,
    block_size=None,
    bias=None,
    activation_scale=None,
    allow_deepgemm=True,
):
    from axolotl.kernels.quantize import dequantize_fp8

    out = input @ dequantize_fp8(weight, weight_scale_inv, input.dtype).t()
    return out if bias is None else out + bias


@pytest.fixture
def fp8_linear(monkeypatch):
    from unittest import mock

    from transformers.integrations import finegrained_fp8

    fake = mock.create_autospec(finegrained_fp8.fp8_linear, side_effect=_dequant_linear)
    monkeypatch.setattr(finegrained_fp8, "fp8_linear", fake)
    return fake


def _fp8_linear_from(linear: nn.Linear):
    from transformers.integrations.finegrained_fp8 import FP8Linear

    out_f, in_f = linear.weight.shape
    w = linear.weight.detach().float()
    blocks = w.reshape(out_f // 128, 128, in_f // 128, 128)
    scale = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-8) / 448.0
    qdata = (blocks / scale[:, None, :, None]).reshape(out_f, in_f)
    fp8 = FP8Linear(in_f, out_f, block_size=tuple(FP8_BLOCK), has_bias=False)
    fp8.weight = nn.Parameter(qdata.to(torch.float8_e4m3fn), requires_grad=False)
    fp8.weight_scale_inv = nn.Parameter(scale, requires_grad=False)
    return fp8


def _tiny_fp8_peft_model(lora_dropout):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    config = MistralLarge4Config(text_config=FP8_TEXT, vision_config=VISION)
    config._attn_implementation = "eager"
    # bf16 like a real run: the LoRA_O fallback dequantizes the fp8 base to bf16
    model = AutoModelForImageTextToText.from_config(config).to(torch.bfloat16)
    for layer in model.model.language_model.layers:
        layer.self_attn.q_proj = _fp8_linear_from(layer.self_attn.q_proj)
        layer.self_attn.o_proj = _fp8_linear_from(layer.self_attn.o_proj)
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=lora_dropout,
            target_modules=r".*language_model.*\.(q_proj|kv_b_proj|o_proj)",
        ),
    )
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(param, std=0.02)
    return peft_model


@pytest.mark.parametrize("lora_dropout", [0.0, 0.1])
def test_native_fp8_attn_lora_matches_peft(fp8_linear, lora_dropout):
    from axolotl.integrations.kernels.libs.dsv4 import lora_fp8

    model = _tiny_fp8_peft_model(lora_dropout).train()
    input_ids = torch.randint(1, FP8_TEXT["vocab_size"], (2, 12))
    torch.manual_seed(1)
    ref_logits, ref_grads = _loss_and_grads(model, input_ids)

    assert patch_mistral_large4_attn_lora(model, native_fp8=True) == 12
    native_calls = []
    orig_apply = lora_fp8._NativeFp8Lora.apply

    def counting_apply(*args):
        native_calls.append(args[0].shape)
        return orig_apply(*args)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(lora_fp8._NativeFp8Lora, "apply", counting_apply)
        torch.manual_seed(1)
        logits, grads = _loss_and_grads(model, input_ids)

    # q_proj + o_proj on each layer run the native fp8 GEMM, dropout included
    assert len(native_calls) == 2 * FP8_TEXT["num_hidden_layers"]
    # bf16 rounding only; a wrong dropout mask or a dropped base input is an O(1) error
    assert _rel_err(logits, ref_logits) < 3e-2
    assert grads.keys() == ref_grads.keys()
    for name, ref in ref_grads.items():
        assert _rel_err(grads[name], ref) < 8e-2, name


LOWBIT_IN, LOWBIT_OUT = 64, 32


def _lowbit_lora_layer(kind, lora_dropout=0.0, lora_bias=False):
    """A PEFT LoRA linear whose frozen base is quantized to ``kind``, plus its dequantized weight."""
    from peft import LoraConfig, get_peft_model

    from axolotl.integrations.kernels.libs.mistral_large4.lora_lowbit import (
        _quantize_weight,
    )

    torch.manual_seed(0)
    peft_model = get_peft_model(
        nn.ModuleDict({"proj": nn.Linear(LOWBIT_IN, LOWBIT_OUT)}),
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=lora_dropout,
            lora_bias=lora_bias,
            target_modules=["proj"],
        ),
    )
    layer = peft_model.base_model.model["proj"]
    try:
        weight = _quantize_weight(layer.base_layer.weight.detach().bfloat16(), kind)
    # torchao without CPU support for this format
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"{kind} quantization unavailable: {exc}")
    layer.base_layer.weight = nn.Parameter(weight, requires_grad=False)
    for name, param in layer.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(param, std=0.05)
    return layer, weight.dequantize().float()


def _lowbit_reference(layer, W, x, seed):
    """``x @ W^T + b0 + s * drop(x) @ A^T @ B^T (+ s * b)`` in plain autograd, same dropout draw."""
    A = layer.lora_A["default"].weight.detach().clone().requires_grad_()
    B = layer.lora_B["default"].weight.detach().clone().requires_grad_()
    bias = layer.lora_B["default"].bias
    bias = bias.detach().clone().requires_grad_() if bias is not None else None
    x = x.detach().clone().requires_grad_()
    s = layer.scaling["default"]
    torch.manual_seed(seed)
    x_lora = layer.lora_dropout["default"](x)
    out = x @ W.t() + layer.base_layer.bias + s * (x_lora @ A.t() @ B.t())
    if bias is not None:
        out = out + s * bias
    out.pow(2).sum().backward()
    return out.detach(), x.grad, A.grad, B.grad, None if bias is None else bias.grad


@pytest.mark.parametrize("lora_bias", [False, True])
@pytest.mark.parametrize("lora_dropout", [0.0, 0.5])
@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8"])
def test_lowbit_lora_matches_autograd_reference(kind, lora_dropout, lora_bias):
    from axolotl.integrations.kernels.libs.mistral_large4.lora_lowbit import (
        lowbit_lora_forward,
    )

    layer, W = _lowbit_lora_layer(kind, lora_dropout, lora_bias)
    layer.train()
    x = torch.randn(2, 5, LOWBIT_IN, requires_grad=True)
    want = _lowbit_reference(layer, W, x, seed=1)

    torch.manual_seed(1)
    out = lowbit_lora_forward(layer, x, kind)
    out.pow(2).sum().backward()

    got = (
        out.detach(),
        x.grad,
        layer.lora_A["default"].weight.grad,
        layer.lora_B["default"].weight.grad,
        layer.lora_B["default"].bias.grad if lora_bias else None,
    )
    assert (got[4] is None) == (want[4] is None)
    for g, w in zip(got, want, strict=True):
        if w is not None:
            torch.testing.assert_close(g, w, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("deepgemm_disabled", [False, True])
def test_native_fp8_attn_lora_honors_deepgemm_guard(fp8_linear, deepgemm_disabled):
    from transformers.integrations.finegrained_fp8 import FP8Linear

    model = _tiny_fp8_peft_model(0.0)
    for module in model.modules():
        if isinstance(module, FP8Linear):
            module._deepgemm_disabled = deepgemm_disabled
    patch_mistral_large4_attn_lora(model, native_fp8=True)

    with torch.no_grad():
        model(input_ids=torch.randint(1, FP8_TEXT["vocab_size"], (2, 12)))

    assert len(fp8_linear.call_args_list) == 2 * FP8_TEXT["num_hidden_layers"]
    for call in fp8_linear.call_args_list:
        assert call.kwargs["allow_deepgemm"] is (not deepgemm_disabled)


def _rel_err(got, want):
    got, want = got.float(), want.float()
    return ((got - want).norm() / want.norm().clamp_min(1e-12)).item()
