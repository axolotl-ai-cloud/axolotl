"""Nemotron source, objective, generation, and adapter contracts."""

from pathlib import Path

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model

from tests.native_source_fixtures import native_source_fixture_path


def _native_source_or_skip(name: str) -> Path:
    source = native_source_fixture_path(name)
    if source is None:
        pytest.skip("native source fixture is unavailable")
    return source


class _NativeTokenizer:
    def decode(self, ids, **kwargs):
        del kwargs
        return ":".join(map(str, ids))


def test_native_nemotron_generation_dispatches_configurable_steps():
    from types import SimpleNamespace

    from axolotl.model_support.native_generation import generate_for_model

    calls = []

    class Nemotron:
        config = SimpleNamespace(
            model_type="nemotron_labs_diffusion", block_size=4, eos_token_id=1
        )

        def generate_with_denoising_steps(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return torch.tensor([[2, 3, 4, 5, 6]]), 3

    result = generate_for_model(
        Nemotron(),
        _NativeTokenizer(),
        torch.tensor([[2, 3]]),
        7,
        0.0,
        0,
        legacy_generate=lambda *args, **kwargs: pytest.fail("legacy called"),
        mode="completion",
        completion_tokens=3,
    )
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], torch.tensor([[2, 3]]))
    assert calls[0][1] == {
        "max_new_tokens": 4,
        "block_length": 4,
        "denoising_steps": 7,
        "temperature": 0.0,
        "eos_token_id": 1,
    }
    assert result["generated_ids"] == [2, 3, 4, 5, 6]


def test_nemotron_factory_accepts_flex_attention():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from axolotl.model_support.nemotron_diffusion import _model_class

    source = _native_source_or_skip("nemotron")
    config_class = get_class_from_dynamic_module(
        "configuration_nemotron_labs_diffusion.NemotronLabsDiffusionConfig",
        str(source),
        local_files_only=True,
    )
    config = config_class(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        mask_token_id=100,
        dlm_paradigm="bidirectional",
        rope_parameters={
            "llama_4_scaling_beta": 1.0,
            "original_max_position_embeddings": 1,
        },
        use_cache=False,
    )
    config._name_or_path = str(source)
    model = _model_class().from_config(
        config, trust_remote_code=True, attn_implementation="flex_attention"
    )
    assert model.config._attn_implementation == "flex_attention"
    assert model._supports_flex_attn
    assert model.encoder._supports_flex_attn
    attention = model.encoder.layers[0].self_attn
    block_mask = type("BlockMask", (), {})()
    with pytest.raises(ValueError, match="does not support KV caching"):
        attention(None, None, block_mask, past_key_values=object())
    attention.attention_dropout = 0.1
    model.train()
    with pytest.raises(ValueError, match="requires attention_dropout=0"):
        attention(None, None, block_mask)


def test_nemotron_configurable_steps_preserve_source_default_and_allow_large_k():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from axolotl.model_support.nemotron_diffusion.compat import (
        resolve_nemotron_model_class,
    )

    source = _native_source_or_skip("nemotron")
    config_class = get_class_from_dynamic_module(
        "configuration_nemotron_labs_diffusion.NemotronLabsDiffusionConfig",
        str(source),
        local_files_only=True,
    )
    config = config_class(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        mask_token_id=100,
        dlm_paradigm="bidirectional",
        rope_parameters={
            "llama_4_scaling_beta": 1.0,
            "original_max_position_embeddings": 1,
        },
        use_cache=False,
    )
    original_class = get_class_from_dynamic_module(
        "modeling_nemotron_labs_diffusion.NemotronLabsDiffusionModel",
        str(source),
        local_files_only=True,
    )
    torch.manual_seed(9)
    original = original_class(config).eval()
    adapted = resolve_nemotron_model_class(source)(config).eval()
    adapted.load_state_dict(original.state_dict())
    prompt = torch.tensor([[1, 2]])
    expected, expected_nfe = original.generate(
        prompt,
        max_new_tokens=2,
        block_length=2,
        temperature=0.0,
        eos_token_id=127,
    )
    actual, actual_nfe = adapted.generate_with_denoising_steps(
        prompt,
        max_new_tokens=2,
        block_length=2,
        denoising_steps=2,
        temperature=0.0,
        eos_token_id=127,
    )
    torch.testing.assert_close(actual, expected)
    assert actual_nfe == expected_nfe

    large_k, large_k_nfe = adapted.generate_with_denoising_steps(
        prompt,
        max_new_tokens=2,
        block_length=2,
        denoising_steps=4,
        temperature=0.0,
        eos_token_id=127,
    )
    assert large_k.shape == expected.shape
    assert not (large_k[:, -2:] == config.mask_token_id).any()
    assert large_k_nfe <= 5
    with pytest.raises(ValueError, match="at least 1"):
        adapted.generate_with_denoising_steps(
            prompt,
            max_new_tokens=2,
            block_length=2,
            denoising_steps=0,
        )


def test_nemotron_tiny_source_mask_loss_and_generic_peft_contract(tmp_path):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from axolotl.model_support.nemotron_diffusion.compat import (
        resolve_nemotron_model_class,
    )

    source = _native_source_or_skip("nemotron")
    config_class = get_class_from_dynamic_module(
        "configuration_nemotron_labs_diffusion.NemotronLabsDiffusionConfig",
        str(source),
        local_files_only=True,
    )
    config = config_class(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        mask_token_id=100,
        dlm_paradigm="bidirectional",
        rope_parameters={
            "llama_4_scaling_beta": 1.0,
            "original_max_position_embeddings": 1,
        },
        use_cache=False,
    )
    original_class = get_class_from_dynamic_module(
        "modeling_nemotron_labs_diffusion.NemotronLabsDiffusionModel",
        str(source),
        local_files_only=True,
    )
    torch.manual_seed(7)
    original = original_class(config).eval()
    adapted = resolve_nemotron_model_class(source)(config).eval()
    adapted.load_state_dict(original.state_dict())
    ids = torch.tensor([[1, 2, 3, 4]])
    selected = torch.tensor([[False, True, False, True]])
    probability = torch.full((1, 4), 0.5)
    raw = original(
        input_ids=ids, labels=ids, masked_indices=selected, p_mask=probability
    )
    patched = adapted(
        input_ids=ids, labels=ids, masked_indices=selected, p_mask=probability
    )
    torch.testing.assert_close(raw.logits, patched.logits)
    torch.testing.assert_close(raw.loss[0], patched.loss[0])
    assert raw.loss[1].item() == patched.loss[1].item() == 2
    expected = (
        torch.nn.functional.cross_entropy(
            patched.logits[selected], ids[selected], reduction="none"
        )
        .div(probability[selected])
        .sum()
    )
    torch.testing.assert_close(patched.loss[0], expected)

    documents = torch.tensor([[1, 2, 3, 4, 5, 6]])
    visible = torch.zeros((1, 1, 6, 6), dtype=torch.bool)
    visible[:, :, :3, :3] = True
    visible[:, :, 3:, 3:] = True
    packed_mask = torch.where(
        visible, torch.tensor(0.0), torch.tensor(torch.finfo(torch.float32).min)
    )
    packed = adapted(input_ids=documents, attention_mask=packed_mask).logits
    packed_bool = adapted(input_ids=documents, attention_mask=visible).logits
    reset_positions = torch.tensor([[0, 1, 2, 0, 1, 2]])
    packed_reset = adapted(
        input_ids=documents,
        attention_mask=packed_mask,
        position_ids=reset_positions,
    ).logits
    left = adapted(input_ids=documents[:, :3]).logits
    right = adapted(input_ids=documents[:, 3:]).logits
    torch.testing.assert_close(packed_bool, packed, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(packed_reset[:, :3], left, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(packed_reset[:, 3:], right, atol=2e-5, rtol=2e-5)
    changed = documents.clone()
    changed[:, 3:] = torch.tensor([[7, 8, 9]])
    isolated = adapted(input_ids=changed, attention_mask=packed_mask).logits
    torch.testing.assert_close(packed[:, :3], isolated[:, :3], atol=2e-5, rtol=2e-5)

    weight = adapted.encoder.layers[0].self_attn.q_proj.weight
    adapted.zero_grad(set_to_none=True)
    adapted(
        input_ids=documents,
        attention_mask=packed_mask,
        position_ids=reset_positions,
    ).logits.sum().backward()
    packed_gradient = weight.grad.detach().clone()
    adapted.zero_grad(set_to_none=True)
    (
        adapted(input_ids=documents[:, :3]).logits.sum()
        + adapted(input_ids=documents[:, 3:]).logits.sum()
    ).backward()
    torch.testing.assert_close(weight.grad, packed_gradient, atol=2e-5, rtol=2e-5)

    infill_source = torch.tensor([[1, 2, 3, 4]])
    infill_mask = torch.tensor([[False, True, False, True]])
    for steps in (1, 2, 4):
        torch.manual_seed(23)
        infilled, nfe = adapted.infill_with_denoising_steps(
            infill_source, infill_mask, steps
        )
        assert nfe <= steps
        assert infilled.shape == infill_source.shape
        assert torch.equal(infilled[~infill_mask], infill_source[~infill_mask])
        assert not (infilled[infill_mask] == adapted.mask_token_id).any()
    from axolotl.model_support.native_generation import generate_for_model

    native_infill = generate_for_model(
        adapted,
        _NativeTokenizer(),
        infill_source,
        2,
        0.0,
        0,
        legacy_generate=lambda *args, **kwargs: pytest.fail("legacy called"),
        mode="random",
        infill_mask=infill_mask,
    )
    assert native_infill["generated_ids"][0] == 1
    assert native_infill["generated_ids"][2] == 3
    assert native_infill["active_canvas_start"] == 0

    base_state = {
        key: value.detach().clone() for key, value in adapted.state_dict().items()
    }
    peft = get_peft_model(
        adapted,
        LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"], task_type=None),
    )
    assert _get_peft_task_type(peft.get_base_model()) is None
    peft(input_ids=ids, labels=ids, masked_indices=selected, p_mask=probability).loss[
        0
    ].backward()
    assert any(
        parameter.grad is not None
        for parameter in peft.parameters()
        if parameter.requires_grad
    )
    peft.eval()
    with torch.no_grad():
        lora_logits = peft(
            input_ids=documents,
            attention_mask=packed_mask,
            position_ids=reset_positions,
        ).logits
    peft.save_pretrained(tmp_path / "nemotron-adapter")
    reload_base = resolve_nemotron_model_class(source)(config).eval()
    reload_base.load_state_dict(base_state)
    reloaded = PeftModel.from_pretrained(reload_base, tmp_path / "nemotron-adapter")
    with torch.no_grad():
        reloaded_logits = reloaded(
            input_ids=documents,
            attention_mask=packed_mask,
            position_ids=reset_positions,
        ).logits
    torch.testing.assert_close(lora_logits, reloaded_logits, atol=2e-5, rtol=2e-5)


def test_nemotron_trainable_token_rows_survive_loaded_peft_merge(tmp_path):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from axolotl.cli.utils.lora_merge import adapter_has_trainable_token_deltas
    from axolotl.model_support.nemotron_diffusion.compat import (
        resolve_nemotron_model_class,
    )

    source = _native_source_or_skip("nemotron")
    config_class = get_class_from_dynamic_module(
        "configuration_nemotron_labs_diffusion.NemotronLabsDiffusionConfig",
        str(source),
        local_files_only=True,
    )
    config = config_class(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        mask_token_id=100,
        dlm_paradigm="bidirectional",
        rope_parameters={
            "llama_4_scaling_beta": 1.0,
            "original_max_position_embeddings": 1,
        },
        use_cache=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(17)
    base = resolve_nemotron_model_class(source)(config).eval()
    base_state = {
        key: value.detach().clone() for key, value in base.state_dict().items()
    }
    base_embedding = base.get_input_embeddings().weight.detach().clone()
    peft = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["q_proj"],
            trainable_token_indices={"encoder.embed_tokens": [7, 8]},
            task_type=None,
        ),
    ).eval()
    delta = next(
        parameter
        for name, parameter in peft.named_parameters()
        if ".trainable_tokens_delta." in name
    )
    with torch.no_grad():
        delta.copy_(torch.full_like(delta, 0.125))
        peft.get_submodule("base_model.model.encoder.layers.0.self_attn.q_proj").lora_B[
            "default"
        ].weight.fill_(0.03125)
    expected_trainable_rows = delta.detach().clone()
    input_ids = torch.tensor([[7, 8, 9, 10]])
    expected = peft(input_ids=input_ids).logits.detach()
    adapter = tmp_path / "adapter"
    peft.save_pretrained(adapter)
    assert adapter_has_trainable_token_deltas(adapter)

    reloaded_base = resolve_nemotron_model_class(source)(config).eval()
    reloaded_base.load_state_dict(base_state)
    reloaded = PeftModel.from_pretrained(reloaded_base, adapter).eval()
    torch.testing.assert_close(
        expected,
        reloaded(input_ids=input_ids).logits,
        atol=2e-5,
        rtol=2e-5,
    )
    merged = reloaded.merge_and_unload().eval()
    torch.testing.assert_close(
        expected, merged(input_ids=input_ids).logits, atol=2e-5, rtol=2e-5
    )
    merged_embedding = merged.get_input_embeddings().weight.detach()
    torch.testing.assert_close(merged_embedding[7:9], expected_trainable_rows)
    torch.testing.assert_close(merged_embedding[6], base_embedding[6])
