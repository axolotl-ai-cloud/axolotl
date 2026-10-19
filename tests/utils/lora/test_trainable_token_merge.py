import json

import pytest
import safetensors.torch
import torch

from axolotl.cli.utils.lora_merge import merge_lora_sharded_efficient


def _tiny_base(vocab_size=32, tied=True):
    from transformers import LlamaConfig, LlamaForCausalLM

    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=tied,
        )
    )


def _write_sharded_base(path, model):
    path.mkdir()
    model.config.architectures = [type(model).__name__]
    model.config.save_pretrained(path)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    first = {"model.embed_tokens.weight": state.pop("model.embed_tokens.weight")}
    safetensors.torch.save_file(first, str(path / "model-00001-of-00002.safetensors"))
    safetensors.torch.save_file(state, str(path / "model-00002-of-00002.safetensors"))


def _save_token_adapter(path, model, indices):
    from peft import LoraConfig, get_peft_model

    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj"],
            trainable_token_indices=indices,
        ),
    )
    delta = next(
        value
        for name, value in peft_model.named_parameters()
        if name.endswith("trainable_tokens_delta.default")
    )
    with torch.no_grad():
        delta.copy_(
            torch.arange(delta.numel(), dtype=delta.dtype).reshape_as(delta) / 19
        )
        peft_model.get_submodule(
            "base_model.model.model.layers.0.self_attn.q_proj"
        ).lora_B["default"].weight.fill_(0.03125)
    peft_model.save_pretrained(path)
    return peft_model


def _load_merged_state(path):
    result = {}
    for shard in path.glob("*.safetensors"):
        result.update(safetensors.torch.load_file(str(shard)))
    return result


@pytest.mark.parametrize(
    ("tied", "indices", "expected_keys"),
    [
        (True, [3, 12], ("model.embed_tokens.weight", "lm_head.weight")),
        (False, {"model.embed_tokens": [3, 12]}, ("model.embed_tokens.weight",)),
    ],
)
def test_sharded_trainable_token_rows_match_peft_merge_and_preserve_other_rows(
    tmp_path, tied, indices, expected_keys
):
    from peft import PeftModel

    torch.manual_seed(3)
    base = _tiny_base(tied=tied).eval()
    base_state = {
        key: value.detach().clone() for key, value in base.state_dict().items()
    }
    _save_token_adapter(tmp_path / "adapter", base, indices)
    expected_base = _tiny_base(tied=tied).eval()
    expected_base.load_state_dict(base_state)
    expected = (
        PeftModel.from_pretrained(expected_base, tmp_path / "adapter")
        .merge_and_unload()
        .eval()
    )
    expected_state = expected.state_dict()
    base_for_shards = _tiny_base(tied=tied).eval()
    base_for_shards.load_state_dict(base_state)
    _write_sharded_base(tmp_path / "base", base_for_shards)

    merge_lora_sharded_efficient(
        tmp_path / "base", tmp_path / "adapter", tmp_path / "merged", device="cpu"
    )

    merged = _load_merged_state(tmp_path / "merged")
    assert merged.keys() == expected_state.keys()
    for key in expected_state:
        torch.testing.assert_close(merged[key], expected_state[key])
    for key in expected_keys:
        untouched = torch.ones(base_state[key].shape[0], dtype=torch.bool)
        untouched[[3, 12]] = False
        assert torch.equal(merged[key][untouched], base_state[key][untouched])
    assert not torch.equal(
        merged["model.embed_tokens.weight"][3],
        base_state["model.embed_tokens.weight"][3],
    )
    reloaded = type(expected).from_pretrained(tmp_path / "merged").eval()
    with torch.no_grad():
        inputs = torch.tensor([[3, 12, 5, 2]])
        torch.testing.assert_close(reloaded(inputs).logits, expected(inputs).logits)


@pytest.mark.parametrize(
    "indices, payload, message",
    [
        ([1, 1], torch.ones(2, 4), "unique"),
        ([8], torch.ones(1, 4), "out of bounds"),
        ([1], torch.tensor([[float("nan"), 0, 0, 0]]), "non-finite"),
    ],
)
def test_trainable_token_payload_is_rejected_before_output_mutation(
    tmp_path, indices, payload, message
):
    base = tmp_path / "base"
    base.mkdir()
    safetensors.torch.save_file(
        {"model.embed_tokens.weight": torch.zeros(8, 4)}, base / "model.safetensors"
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 2, "trainable_token_indices": indices})
    )
    safetensors.torch.save_file(
        {
            "base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta": payload
        },
        adapter / "adapter_model.safetensors",
    )

    with pytest.raises(ValueError, match=message):
        merge_lora_sharded_efficient(base, adapter, tmp_path / "merged", device="cpu")
    assert not (tmp_path / "merged").exists()


def _raw_token_adapter(
    tmp_path,
    *,
    indices=None,
    dtype=torch.float32,
    values=None,
    extra=None,
    base_key="model.embed_tokens.weight",
):
    base = tmp_path / "base"
    base.mkdir()
    safetensors.torch.save_file(
        {base_key: torch.zeros(8, 4, dtype=dtype)}, base / "model.safetensors"
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 2,
                "lora_alpha": 2,
                "trainable_token_indices": [1] if indices is None else indices,
            }
        )
    )
    state = {
        "base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta": torch.ones(
            1, 4
        )
        if values is None
        else values
    }
    state.update(extra or {})
    safetensors.torch.save_file(state, adapter / "adapter_model.safetensors")
    return base, adapter


@pytest.mark.parametrize(
    "kind", ["missing", "full_override", "lora", "embedding_lora", "overflow"]
)
def test_invalid_token_merge_is_rejected_before_output_exists(tmp_path, kind):
    kwargs = {}
    if kind == "missing":
        kwargs["indices"] = {"model.embed_tokens": [1], "missing_embedding": [2]}
        message = "no payload"
    elif kind == "full_override":
        kwargs["extra"] = {
            "base_model.model.model.embed_tokens.weight": torch.ones(8, 4)
        }
        message = "full-weight override"
    elif kind in {"lora", "embedding_lora"}:
        suffix = "lora_A.weight" if kind == "lora" else "lora_embedding_A"
        kwargs["extra"] = {
            f"base_model.model.model.embed_tokens.{suffix}": torch.ones(2, 4)
        }
        message = "LoRA weights"
    else:
        kwargs.update(dtype=torch.float16, values=torch.full((1, 4), 1e10))
        message = "non-finite"
    base, adapter = _raw_token_adapter(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=message):
        merge_lora_sharded_efficient(base, adapter, tmp_path / "merged", device="cpu")
    assert not (tmp_path / "merged").exists()


def test_token_rows_follow_checkpoint_weight_renamings(tmp_path, monkeypatch):
    base, adapter = _raw_token_adapter(tmp_path, base_key="legacy.embed.weight")
    monkeypatch.setattr(
        "axolotl.cli.utils.lora_merge._get_conversion_info",
        lambda *args, **kwargs: ({r"^legacy\.embed$": "model.embed_tokens"}, []),
    )
    merge_lora_sharded_efficient(base, adapter, tmp_path / "merged", device="cpu")
    merged = _load_merged_state(tmp_path / "merged")["legacy.embed.weight"]
    assert torch.equal(merged[1], torch.ones(4))
    assert torch.count_nonzero(merged) == 4
