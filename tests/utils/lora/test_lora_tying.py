import pytest
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import Qwen2Config, Qwen2ForCausalLM

from axolotl.utils.lora_precision import upcast_lora_parameters
from axolotl.utils.lora_tying import TiedTransposedLinear, tie_lora_output_embeddings


def _tied_peft_model(dtype=torch.bfloat16, ensure_weight_tying=True):
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(config).to(dtype)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "embed_tokens", "lm_head"],
        ensure_weight_tying=ensure_weight_tying,
    )
    return get_peft_model(model, lora_config)


def _adapters(model):
    emb = model.base_model.model.model.embed_tokens
    head = model.base_model.model.lm_head
    return emb, head


def test_head_adapter_reads_embedding_adapter():
    model = _tied_peft_model()
    upcast_lora_parameters(model)
    assert tie_lora_output_embeddings(model) == ["base_model.model.lm_head"]

    emb, head = _adapters(model)
    assert isinstance(head.lora_A["default"], TiedTransposedLinear)
    assert not any("lm_head.lora_" in n for n, _ in model.named_parameters())
    assert torch.equal(
        head.lora_A["default"].weight, emb.lora_embedding_B["default"].t()
    )
    assert torch.equal(
        head.lora_B["default"].weight, emb.lora_embedding_A["default"].t()
    )


def test_gradients_from_both_uses_accumulate_into_one_parameter():
    model = _tied_peft_model()
    upcast_lora_parameters(model)
    tie_lora_output_embeddings(model)
    emb, head = _adapters(model)
    with torch.no_grad():
        emb.lora_embedding_A["default"].normal_()

    x = torch.randn(3, 16, dtype=torch.bfloat16)
    head(x).float().pow(2).sum().backward()
    grad_from_head = emb.lora_embedding_A["default"].grad.clone()
    assert grad_from_head.abs().sum() > 0

    emb(torch.tensor([[1, 2, 3]])).float().sum().backward()
    assert not torch.equal(emb.lora_embedding_A["default"].grad, grad_from_head)


def test_tie_survives_optimizer_steps():
    model = _tied_peft_model()
    upcast_lora_parameters(model)
    tie_lora_output_embeddings(model)
    emb, head = _adapters(model)
    params = [p for p in model.parameters() if p.requires_grad]
    assert len({id(p) for p in params}) == len(params)
    opt = torch.optim.AdamW(params, lr=1e-2)
    for _ in range(3):
        out = model(input_ids=torch.tensor([[1, 2, 3, 4]]))
        out.logits.float().pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
    assert emb.lora_embedding_A["default"].abs().sum() > 0
    assert torch.equal(
        head.lora_B["default"].weight, emb.lora_embedding_A["default"].t()
    )


def test_state_dict_keeps_peft_layout_and_round_trips():
    model = _tied_peft_model()
    upcast_lora_parameters(model)
    tie_lora_output_embeddings(model)
    emb, _ = _adapters(model)
    with torch.no_grad():
        emb.lora_embedding_A["default"].normal_()

    sd = get_peft_model_state_dict(model)
    head_b = next(v for k, v in sd.items() if k.endswith("lm_head.lora_B.weight"))
    assert torch.equal(head_b, emb.lora_embedding_A["default"].detach().t())

    fresh = _tied_peft_model()
    upcast_lora_parameters(fresh)
    tie_lora_output_embeddings(fresh)
    full = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    fresh.load_state_dict(full, strict=False)
    fresh_emb, fresh_head = _adapters(fresh)
    assert torch.equal(
        fresh_emb.lora_embedding_A["default"], emb.lora_embedding_A["default"]
    )
    assert torch.equal(
        fresh_head.lora_B["default"].weight, fresh_emb.lora_embedding_A["default"].t()
    )


@pytest.mark.parametrize("ensure_weight_tying", [False])
def test_untied_config_is_left_alone(ensure_weight_tying):
    model = _tied_peft_model(ensure_weight_tying=ensure_weight_tying)
    assert tie_lora_output_embeddings(model) == []
    _, head = _adapters(model)
    assert isinstance(head.lora_A["default"], torch.nn.Linear)
