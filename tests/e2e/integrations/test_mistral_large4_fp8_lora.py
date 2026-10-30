"""Short LoRA training smoke for a tiny native Mistral Large 4 on a block-FP8 base: loaded through
the kernels plugin (scattermoe experts on the fp8_block grouped path that ``moe_grouped_mode: auto``
picks, + expert LoRA), with and without the fused MLA attention-LoRA kernels."""

import pytest
import torch

from axolotl.loaders import ModelLoader, load_tokenizer
from axolotl.utils.config import normalize_config, prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
        reason="CUDA with FP8 support (sm89+) required",
    ),
]

TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
TEXT = dict(
    vocab_size=49152,
    hidden_size=256,
    intermediate_size=256,
    moe_intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=2,
    num_key_value_heads=2,
    n_routed_experts=8,
    num_experts_per_tok=2,
    kv_lora_rank=128,
    qk_rope_head_dim=64,
    qk_nope_head_dim=64,
    v_head_dim=128,
    index_topk=16,
    index_head_dim=128,
    index_n_heads=2,
    first_k_dense_replace=1,
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
NOT_CONVERTED = [
    "model.vision_tower",
    "model.multi_modal_projector",
    "lm_head",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "indexer.wk",
    "indexer.weights_proj",
]


@pytest.fixture(scope="module")
def bf16_dir(tmp_path_factory):
    from transformers import AutoModelForImageTextToText, MistralLarge4Config

    torch.manual_seed(0)
    config = MistralLarge4Config(text_config=TEXT, vision_config=VISION)
    path = tmp_path_factory.mktemp("ml4_bf16")
    model = AutoModelForImageTextToText.from_config(config).to(torch.bfloat16)
    model.save_pretrained(path)
    return str(path)


@pytest.mark.parametrize("attn_kernel", [True, False], ids=["attn_kernel", "peft"])
def test_mistral_large4_fp8_lora_trains(bf16_dir, temp_dir, attn_kernel):
    from transformers.integrations.finegrained_fp8 import FP8Experts, FP8Linear

    native_forward = FP8Linear.forward
    cfg = DictDefault(
        {
            "base_model": bf16_dir,
            "tokenizer_config": TOKENIZER,
            "model_quantization_config": "FineGrainedFP8Config",
            "model_quantization_config_kwargs": {
                "weight_block_size": [128, 128],
                "modules_to_not_convert": NOT_CONVERTED,
            },
            "plugins": ["axolotl.integrations.kernels.KernelsPlugin"],
            "use_scattermoe": True,
            "mistral_large4_attn_lora_kernel": attn_kernel,
            "adapter": "lora",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "lora_target_modules": (
                r"model.language_model.layers.[\d]+."
                r"(self_attn.(q_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)"
                r"|mlp.(shared_experts.)?(gate_proj|up_proj|down_proj))"
            ),
            "lora_target_parameters": [
                "mlp.experts.gate_up_proj",
                "mlp.experts.down_proj",
            ],
            "attn_implementation": "sdpa",
            "bf16": True,
            "gradient_checkpointing": True,
            "sequence_len": 128,
            "micro_batch_size": 2,
            "num_epochs": 1,
            "learning_rate": 1e-3,
            "datasets": [{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
            "output_dir": temp_dir,
        }
    )
    prepare_plugins(cfg)
    cfg = validate_config(cfg)
    normalize_config(cfg)
    tokenizer = load_tokenizer(cfg)
    model, _ = ModelLoader(cfg, tokenizer).load()
    assert (
        cfg.moe_grouped_mode == "fp8_block"
    )  # auto, from the on-load FP8 quantization

    modules = dict(model.named_modules())
    assert any(isinstance(m, FP8Linear) for m in modules.values())
    assert any(isinstance(m, FP8Experts) for m in modules.values())
    assert FP8Linear.forward is native_forward
    patched = [n for n, m in modules.items() if hasattr(m, "_ml4_orig_forward")]
    assert len(patched) == (15 if attn_kernel else 0)

    trainable = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate)
    gen = torch.Generator(device="cpu").manual_seed(0)
    input_ids = torch.randint(1, len(tokenizer), (2, 128), generator=gen).cuda()
    model.train()
    losses = []
    for _ in range(8):
        loss = model(input_ids=input_ids, labels=input_ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    print(f"\nattn_kernel={attn_kernel} losses: {[round(v, 4) for v in losses]}")

    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    moved = {
        n
        for n, p, b in zip(names, trainable, before, strict=True)
        if not torch.equal(p, b)
    }
    assert any("experts" in n for n in moved)
    assert any("kv_b_proj" in n for n in moved)
    assert any("shared_experts" in n for n in moved)
