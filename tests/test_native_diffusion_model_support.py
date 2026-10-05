"""Focused contracts for native diffusion model descriptors and tied PEFT layers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import safetensors.torch
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import DiffusionGemmaConfig, DiffusionGemmaForBlockDiffusion

from axolotl.loaders.adapter import _get_peft_task_type
from axolotl.model_support import get_model_support, resolve_model_support
from axolotl.model_support.native_adapters import (
    share_diffusion_gemma_tied_lora,
    validate_native_diffusion_lora,
)
from axolotl.utils.dict import DictDefault

from tests.native_source_fixtures import native_source_fixture_path

_TARGET = r"^(model\.encoder\.language_model\.layers|model\.decoder\.layers)\.0\.self_attn\.q_proj$"
_ENC = "base_model.model.model.encoder.language_model.layers.0.self_attn.q_proj"
_DEC = "base_model.model.model.decoder.layers.0.self_attn.q_proj"


def _native_source_or_skip(name: str) -> Path:
    source = native_source_fixture_path(name)
    if source is None:
        pytest.skip(
            "native source-only fixture is unavailable; run "
            "scripts/diffusion_lm/prepare_native_source_fixtures.py"
        )
    assert source is not None
    return source


def _tiny_model() -> DiffusionGemmaForBlockDiffusion:
    return DiffusionGemmaForBlockDiffusion(
        DiffusionGemmaConfig(
            text_config={
                "vocab_size": 32,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 64,
                "layer_types": ["full_attention"],
                "num_experts": 2,
                "top_k_experts": 1,
                "moe_intermediate_size": 32,
                "pad_token_id": 0,
                "eos_token_id": 1,
                "bos_token_id": 2,
            },
            vision_config={
                "model_type": "gemma4_vision",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": 8,
                "max_position_embeddings": 64,
                "patch_size": 16,
                "position_embedding_size": 16,
            },
            canvas_length=16,
        )
    )


def _native_cfg(**overrides) -> DictDefault:
    cfg = DictDefault(
        {
            "diffusion_lm": {"from_causal_lm": False},
            "adapter": "lora",
            "lora_target_modules": [_TARGET],
            "lora_target_parameters": [],
            "lora_target_linear": False,
            "load_in_4bit": False,
            "load_in_8bit": False,
            "peft_use_dora": False,
            "peft_layer_replication": None,
            "fsdp_config": None,
        }
    )
    cfg.update(overrides)
    return cfg


def test_native_descriptors_publish_immutable_specs_and_generic_peft_wrapper():
    gemma = resolve_model_support(get_model_support("diffusion_gemma"))
    dream = resolve_model_support(get_model_support("Dream"))
    nemotron = resolve_model_support(get_model_support("nemotron_labs_diffusion"))

    assert gemma.family == dream.family == nemotron.family == "diffusion_lm"
    assert gemma.diffusion.to_dict()["layout"] == "encoder_canvas"
    assert dream.diffusion.to_dict()["generation_adapter"] == "dream"
    assert nemotron.diffusion.to_dict() == {
        "noise": "absorbing",
        "layout": "full_sequence",
        "logit_alignment": "aligned",
        "first_position_alignment": "requires_predecessor",
        "self_conditioning": False,
        "max_canvas": None,
        "max_context": 262144,
        "eos_handling": "independent",
        "mask_token_policy": "model",
        "default_time_weighting": "inv_t",
        "objective_reduction": "masked_token_mean",
        "time_floor": 0.001,
        "reduction_scope": "global_window",
        "generation_adapter": "full_sequence",
    }
    assert _get_peft_task_type(_tiny_model()) is None


def test_native_lora_validation_rejects_unsafe_modes_and_unscoped_targets():
    validate_native_diffusion_lora(_native_cfg(), model_name="DiffusionGemma")
    with pytest.raises(ValueError, match="quantized"):
        validate_native_diffusion_lora(
            _native_cfg(adapter="qlora"), model_name="DiffusionGemma"
        )
    with pytest.raises(ValueError, match="vision and router"):
        validate_native_diffusion_lora(
            _native_cfg(lora_target_modules=["q_proj"]), model_name="DiffusionGemma"
        )


def test_diffusion_gemma_self_conditioning_module_is_trainable_and_saved(tmp_path):
    from axolotl.model_support.diffusion_gemma import _validate
    from axolotl.model_support.profile import ModelHookContext

    cfg = _native_cfg(
        diffusion_lm={
            "from_causal_lm": False,
            "self_conditioning": {"train_module": True},
        }
    )
    _validate(ModelHookContext(cfg=cfg))
    assert "model.decoder.self_conditioning" in cfg.lora_modules_to_save

    peft = get_peft_model(
        _tiny_model(),
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=_TARGET,
            modules_to_save=cfg.lora_modules_to_save,
            task_type=None,
        ),
    )
    module = peft.get_submodule(
        "base_model.model.model.decoder.self_conditioning.modules_to_save.default"
    )
    parameter = next(module.parameters())
    assert parameter.requires_grad
    with torch.no_grad():
        parameter.fill_(0.125)
    peft.save_pretrained(tmp_path)

    restored = PeftModel.from_pretrained(_tiny_model(), tmp_path, is_trainable=True)
    restored_module = restored.get_submodule(
        "base_model.model.model.decoder.self_conditioning.modules_to_save.default"
    )
    restored_parameter = next(restored_module.parameters())
    torch.testing.assert_close(restored_parameter, parameter)
    assert restored.config.text_config.vocab_size == 32


def test_tied_text_lora_shares_factors_and_rejects_unequal_loaded_duplicates():
    peft = get_peft_model(
        _tiny_model(),
        LoraConfig(r=2, lora_alpha=2, target_modules=_TARGET, task_type=None),
    )
    enc = peft.get_submodule(_ENC)
    dec = peft.get_submodule(_DEC)
    assert (
        enc.lora_A["default"].weight.data_ptr()
        != dec.lora_A["default"].weight.data_ptr()
    )

    share_diffusion_gemma_tied_lora(peft)
    assert enc.lora_A["default"] is dec.lora_A["default"]
    assert enc.lora_B["default"] is dec.lora_B["default"]
    assert enc.merged_adapters is dec.merged_adapters
    params = [param for param in peft.parameters() if param.requires_grad]
    assert len(params) == 2
    optimizer = torch.optim.SGD(params, lr=0.1)
    assert len({id(param) for param in params}) == len(
        {id(param) for group in optimizer.param_groups for param in group["params"]}
    )
    before = enc.base_layer.weight.detach().clone()
    enc.lora_A["default"].weight.data.fill_(1)
    enc.lora_B["default"].weight.data.fill_(1)
    enc.merge()
    dec.merge()
    assert torch.allclose(enc.base_layer.weight, dec.base_layer.weight)
    enc.unmerge()
    assert torch.allclose(enc.base_layer.weight, before, atol=1e-6, rtol=0)

    reloaded_style = get_peft_model(
        _tiny_model(),
        LoraConfig(r=2, lora_alpha=2, target_modules=_TARGET, task_type=None),
    )
    reloaded_enc = reloaded_style.get_submodule(_ENC)
    reloaded_dec = reloaded_style.get_submodule(_DEC)
    reloaded_dec.lora_A["default"].weight.data.copy_(
        reloaded_enc.lora_A["default"].weight.detach().add(1)
    )
    with pytest.raises(ValueError, match="unequal LoRA factors"):
        share_diffusion_gemma_tied_lora(reloaded_style, validate_saved=True)


def _forward_args():
    def dense_masks(valid, causal, query_length=None):
        query_valid = valid if query_length is None else valid[:, -query_length:]
        visible = query_valid[:, None, :, None] & valid[:, None, None, :]
        if causal:
            visible &= torch.ones(
                query_valid.shape[-1], valid.shape[-1], dtype=torch.bool
            ).tril(diagonal=valid.shape[-1] - query_valid.shape[-1])
        return {"full_attention": visible, "sliding_attention": visible}

    prompt = torch.tensor([[2, 3]])
    canvas = torch.tensor([[4, 5]])
    return {
        "input_ids": prompt,
        "attention_mask": dense_masks(torch.ones_like(prompt, dtype=torch.bool), True),
        "position_ids": torch.tensor([[0, 1]]),
        "decoder_input_ids": canvas,
        "decoder_attention_mask": dense_masks(
            torch.ones((1, 4), dtype=torch.bool), False, query_length=2
        ),
        "decoder_position_ids": torch.tensor([[2, 3]]),
    }


def test_packed_diffusion_gemma_subclass_keeps_encoder_cache_in_one_forward():
    from axolotl.model_support.diffusion_gemma.modeling import (
        AxolotlDiffusionGemmaForBlockDiffusion,
        PackedDiffusionGemmaOutput,
    )

    base = _tiny_model().eval()
    model = AxolotlDiffusionGemmaForBlockDiffusion(base.config).eval()
    model.load_state_dict(base.state_dict())
    args = _forward_args()
    packed = model(
        encoder_input_ids=args["input_ids"],
        encoder_attention_mask=args["attention_mask"],
        encoder_position_ids=args["position_ids"],
        decoder_input_ids=args["decoder_input_ids"],
        decoder_attention_mask=args["decoder_attention_mask"],
        decoder_position_ids=args["decoder_position_ids"],
    )
    assert isinstance(packed, PackedDiffusionGemmaOutput)
    assert packed.encoder_last_hidden_state is not None
    assert packed.past_key_values is not None
    next_packed = model(
        encoder_input_ids=args["input_ids"],
        encoder_attention_mask=args["attention_mask"],
        encoder_position_ids=args["position_ids"],
        decoder_input_ids=args["decoder_input_ids"],
        decoder_attention_mask=args["decoder_attention_mask"],
        decoder_position_ids=args["decoder_position_ids"],
        past_key_values=packed.past_key_values,
        self_conditioning_logits=packed.logits,
        self_conditioning_token_mask=torch.ones((1, 2), dtype=torch.bool),
    )
    assert next_packed.encoder_last_hidden_state is None
    (packed.logits.sum() + next_packed.logits.sum()).backward()
    assert (
        model.model.encoder.language_model.layers[0].self_attn.q_proj.weight.grad
        is not None
    )


def test_packed_diffusion_gemma_subclass_unrolls_inside_ddp_forward():
    from axolotl.model_support.diffusion_gemma.modeling import (
        AxolotlDiffusionGemmaForBlockDiffusion,
        PackedDiffusionGemmaOutput,
    )

    base = _tiny_model().eval()
    model = AxolotlDiffusionGemmaForBlockDiffusion(base.config).eval()
    model.load_state_dict(base.state_dict())
    args = _forward_args()
    canvas = args["decoder_input_ids"]
    output = model(
        encoder_input_ids=args["input_ids"],
        encoder_attention_mask=args["attention_mask"],
        encoder_position_ids=args["position_ids"],
        decoder_input_ids=canvas,
        decoder_attention_mask=args["decoder_attention_mask"],
        decoder_position_ids=args["decoder_position_ids"],
        unroll_steps=2,
        grad_through_steps=False,
        k1_conditioning_mask=torch.ones_like(canvas, dtype=torch.bool),
        recurrent_conditioning_mask=torch.ones_like(canvas, dtype=torch.bool),
        update_mask=torch.ones_like(canvas, dtype=torch.bool),
    )
    assert isinstance(output, PackedDiffusionGemmaOutput)
    assert output.encoder_last_hidden_state is not None
    output.logits.sum().backward()
    assert (
        model.model.encoder.language_model.layers[0].self_attn.q_proj.weight.grad
        is not None
    )


def test_packed_diffusion_gemma_bypasses_native_decoder_mask_creator(monkeypatch):
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaDecoderModel,
    )

    from axolotl.model_support.diffusion_gemma.modeling import (
        AxolotlDiffusionGemmaForBlockDiffusion,
    )

    def mask_creator_must_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("native decoder mask creation must not run")

    monkeypatch.setattr(
        DiffusionGemmaDecoderModel,
        "create_diffusion_decoder_attention_mask",
        mask_creator_must_not_run,
    )
    base = _tiny_model()
    model = AxolotlDiffusionGemmaForBlockDiffusion(base.config)
    model.load_state_dict(base.state_dict())
    args = _forward_args()

    # The normal native decoder path accepts the supplied mapping directly.
    direct = model(**args)
    assert direct.logits.shape == (1, 2, base.config.text_config.vocab_size)

    model.train()
    packed = model(
        encoder_input_ids=args["input_ids"],
        encoder_attention_mask=args["attention_mask"],
        encoder_position_ids=args["position_ids"],
        decoder_input_ids=args["decoder_input_ids"],
        decoder_attention_mask=args["decoder_attention_mask"],
        decoder_position_ids=args["decoder_position_ids"],
        unroll_steps=1,
        pilot_for_single_step=False,
        grad_through_steps=False,
        k1_conditioning_mask=torch.ones((1, 2), dtype=torch.bool),
        recurrent_conditioning_mask=torch.ones((1, 2), dtype=torch.bool),
        update_mask=torch.ones((1, 2), dtype=torch.bool),
    )
    packed.logits.square().mean().backward()
    assert model.model.decoder.layers[0].self_attn.q_proj.weight.grad is not None


def test_tied_lora_full_native_save_reload_merge_and_export(tmp_path, monkeypatch):
    from axolotl.model_support.diffusion_gemma.modeling import (
        AxolotlDiffusionGemmaForBlockDiffusion,
    )

    base = _tiny_model()
    model = AxolotlDiffusionGemmaForBlockDiffusion(base.config)
    model.load_state_dict(base.state_dict())
    base_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    peft = get_peft_model(
        model, LoraConfig(r=2, lora_alpha=2, target_modules=_TARGET, task_type=None)
    )
    share_diffusion_gemma_tied_lora(peft)
    enc, dec = peft.get_submodule(_ENC), peft.get_submodule(_DEC)
    assert enc.lora_A["default"] is dec.lora_A["default"]
    assert enc.lora_B["default"] is dec.lora_B["default"]
    params = [param for param in peft.parameters() if param.requires_grad]
    assert len(params) == 2
    optimizer = torch.optim.SGD(params, lr=0.01)
    assert len({id(param) for param in params}) == len(
        optimizer.param_groups[0]["params"]
    )
    peft.train()
    peft(**_forward_args()).logits.sum().backward()
    before = enc.lora_B["default"].weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, enc.lora_B["default"].weight)
    peft.eval()
    updated = peft(**_forward_args()).logits.detach()
    peft.merge_adapter()
    merged = peft(**_forward_args()).logits.detach()
    peft.unmerge_adapter()
    unmerged = peft(**_forward_args()).logits.detach()
    assert torch.allclose(updated, merged, atol=2e-5, rtol=2e-5)
    assert torch.allclose(updated, unmerged, atol=2e-5, rtol=2e-5)
    peft.save_pretrained(tmp_path / "adapter")
    reload_base = AxolotlDiffusionGemmaForBlockDiffusion(base.config)
    reload_base.load_state_dict(base_state)
    reloaded = PeftModel.from_pretrained(reload_base, tmp_path / "adapter")
    share_diffusion_gemma_tied_lora(reloaded, validate_saved=True)
    assert torch.allclose(
        updated, reloaded(**_forward_args()).logits, atol=2e-5, rtol=2e-5
    )
    exported = reloaded.merge_and_unload()
    exported.eval()
    exported_logits = exported(**_forward_args()).logits.detach()
    assert torch.allclose(updated, exported_logits, atol=2e-5, rtol=2e-5)
    exported.save_pretrained(tmp_path / "export")
    assert json.loads((tmp_path / "export" / "config.json").read_text())[
        "architectures"
    ] == ["DiffusionGemmaForBlockDiffusion"]
    hf = DiffusionGemmaForBlockDiffusion.from_pretrained(tmp_path / "export")
    assert torch.allclose(
        exported_logits, hf(**_forward_args()).logits, atol=2e-5, rtol=2e-5
    )
    assert set(hf.state_dict()) == set(_tiny_model().state_dict())


def test_tied_lora_rejects_partially_targeted_actual_alias_group():
    peft = get_peft_model(
        _tiny_model(),
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=r"^model\.encoder\.language_model\.layers\.0\.self_attn\.q_proj$",
            task_type=None,
        ),
    )
    with pytest.raises(ValueError, match="every projection sharing a text weight"):
        share_diffusion_gemma_tied_lora(peft)


@pytest.mark.parametrize(
    ("source", "revision"),
    [
        ("Dream-org/Dream-v0-Base-7B", "6572adb5535263e4d1a337b56942ba48b6dee2a9"),
        ("Dream-org/Dream-v0-Instruct-7B", "05334cb9faaf763692dcf9d8737c642be2b2a6ae"),
    ],
)
def test_dream_factory_pins_code_and_weights(source, revision, monkeypatch, tmp_path):
    import axolotl.model_support.dream.compat as compat
    from axolotl.model_support.dream import _model_class

    calls = []

    class MockDream:
        reset_calls = 0

        @classmethod
        def from_pretrained(cls, model_source, **kwargs):
            calls.append((model_source, kwargs))
            return cls()

        def reset_rope_parameters(self):
            type(self).reset_calls += 1

        @classmethod
        def _from_config(cls, config, **kwargs):
            instance = cls()
            instance.config = config
            instance.kwargs = kwargs
            return instance

    def resolver(model_source, *, revision=None, local_files_only=False):
        calls.append(("resolver", model_source, revision, local_files_only))
        return MockDream

    monkeypatch.setattr(compat, "resolve_patched_dream_model_class", resolver)
    factory = _model_class()
    assert isinstance(factory.from_pretrained(source), MockDream)
    assert calls == [
        ("resolver", source, revision, False),
        (source, {"revision": revision}),
    ]
    assert MockDream.reset_calls == 1
    calls.clear()
    assert isinstance(factory.from_pretrained(source, revision=revision), MockDream)
    assert calls == [
        ("resolver", source, revision, False),
        (source, {"revision": revision}),
    ]
    assert MockDream.reset_calls == 2
    calls.clear()
    assert isinstance(factory.from_pretrained(source, local_files_only=True), MockDream)
    assert calls == [
        ("resolver", source, revision, True),
        (source, {"local_files_only": True, "revision": revision}),
    ]
    assert MockDream.reset_calls == 3
    local = tmp_path / "export"
    local.mkdir()
    assert isinstance(factory.from_pretrained(local), MockDream)
    assert calls[-2:] == [("resolver", local, None, False), (local, {})]
    assert MockDream.reset_calls == 4


def test_dream_factory_from_config_removes_loader_only_kwargs(monkeypatch):
    from types import SimpleNamespace

    import axolotl.model_support.dream.compat as compat
    from axolotl.model_support.dream import _model_class

    seen = []

    class MockDream:
        @classmethod
        def _from_config(cls, config, **kwargs):
            seen.append((config, kwargs))
            return cls()

    monkeypatch.setattr(
        compat,
        "resolve_patched_dream_model_class",
        lambda source, *, revision=None, local_files_only=False: (
            seen.append((source, revision, local_files_only)) or MockDream
        ),
    )
    config = SimpleNamespace(
        _name_or_path="Dream-org/Dream-v0-Base-7B", _commit_hash="pin"
    )
    result = _model_class().from_config(
        config, trust_remote_code=True, torch_dtype=torch.float32
    )
    assert isinstance(result, MockDream)
    assert seen == [
        ("Dream-org/Dream-v0-Base-7B", "pin", False),
        (config, {"torch_dtype": torch.float32}),
    ]


def test_dream_factory_builds_tiny_cached_audited_source():
    from transformers import AutoConfig

    from axolotl.model_support.dream import _model_class

    source = _native_source_or_skip("dream")
    config = AutoConfig.from_pretrained(
        source, trust_remote_code=True, local_files_only=True
    )
    for name, value in {
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "max_window_layers": 1,
        "bos_token_id": 1,
        "eos_token_id": 1,
        "pad_token_id": 1,
        "mask_token_id": 2,
        "use_cache": False,
    }.items():
        setattr(config, name, value)
    config._name_or_path = str(source)
    model = _model_class().from_config(
        config,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="flex_attention",
    )
    assert type(model).__name__ == "DreamModel"
    assert model.config._attn_implementation == "flex_attention"
    assert model._supports_flex_attn
    assert model.model._supports_flex_attn
    block_mask = type("BlockMask", (), {})()
    with pytest.raises(ValueError, match="does not support KV caching"):
        model.model.layers[0].self_attn(
            attention_mask=block_mask, past_key_value=object()
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dream_fp32_rms_norm_preserves_projection_dtype(dtype):
    from transformers import AutoConfig

    from axolotl.model_support.dream import _model_class

    source = _native_source_or_skip("dream")
    config = AutoConfig.from_pretrained(
        source, trust_remote_code=True, local_files_only=True
    )
    for name, value in {
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "max_window_layers": 1,
        "bos_token_id": 1,
        "eos_token_id": 1,
        "pad_token_id": 1,
        "mask_token_id": 2,
        "use_cache": False,
    }.items():
        setattr(config, name, value)
    config._name_or_path = str(source)
    model = _model_class().from_config(
        config, trust_remote_code=True, torch_dtype=dtype
    )
    model.to(dtype)
    for module in model.modules():
        if type(module).__name__ == "DreamRMSNorm":
            module.float()
            module.weight.data.fill_(1.5)
            reference_input = torch.arange(
                1, module.weight.numel() + 1, dtype=dtype
            ).unsqueeze(0)
            reference_input[:, 1::2].neg_()
            reference = reference_input.float()
            reference = reference * torch.rsqrt(
                reference.pow(2).mean(-1, keepdim=True) + module.variance_epsilon
            )
            torch.testing.assert_close(
                module(reference_input),
                (module.weight * reference.to(dtype)).to(dtype),
                rtol=2e-3,
                atol=2e-3,
            )
            assert module.weight.dtype is torch.float32
    outputs = model(
        input_ids=torch.tensor([[1, 3, 4, 2]]), labels=torch.tensor([[1, 3, 4, 2]])
    )
    assert outputs.logits.dtype is dtype
    assert torch.isfinite(outputs.loss)
    outputs.loss.backward()
    gradient = model.model.layers[0].self_attn.q_proj.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_dream_factory_from_pretrained_resets_nonpersistent_rope_buffers(tmp_path):
    from transformers import AutoConfig

    from axolotl.model_support.dream import _model_class
    from axolotl.model_support.dream.compat import legacy_default_rope_parameters

    source = _native_source_or_skip("dream")
    config = AutoConfig.from_pretrained(
        source, trust_remote_code=True, local_files_only=True
    )
    for name, value in {
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "max_window_layers": 1,
        "bos_token_id": 1,
        "eos_token_id": 1,
        "pad_token_id": 1,
        "mask_token_id": 2,
        "use_cache": False,
    }.items():
        setattr(config, name, value)
    config._name_or_path = str(source)
    factory = _model_class()
    model = factory.from_config(config, trust_remote_code=True)
    config.save_pretrained(tmp_path)
    torch.save(model.state_dict(), tmp_path / "pytorch_model.bin")
    for filename in (
        "configuration_dream.py",
        "generation_config.json",
        "generation_utils.py",
        "modeling_dream.py",
    ):
        shutil.copy2(source / filename, tmp_path / filename)
    loaded = factory.from_pretrained(
        tmp_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    expected, _ = legacy_default_rope_parameters(loaded.config)
    buffers = [
        buffer
        for name, buffer in loaded.named_buffers()
        if name.endswith("rotary_emb.inv_freq")
    ]
    assert buffers
    assert all(torch.isfinite(buffer).all() for buffer in buffers)
    for buffer in buffers:
        torch.testing.assert_close(buffer.cpu(), expected.cpu())


def test_native_diffusion_gemma_merge_dispatches_to_loaded_peft_path(monkeypatch):
    from axolotl.cli.merge_lora import do_merge_lora

    calls = []
    monkeypatch.setattr(
        "axolotl.cli.merge_lora._do_merge_lora_legacy",
        lambda *, cfg: calls.append("legacy"),
    )
    monkeypatch.setattr(
        "axolotl.cli.merge_lora._do_merge_lora_efficient",
        lambda *, cfg: calls.append("memory_efficient"),
    )
    do_merge_lora(
        cfg=DictDefault({"model_config_type": "diffusion_gemma", "merge_method": None})
    )
    assert calls == ["legacy"]


def test_native_diffusion_gemma_rejects_explicit_efficient_merge():
    from axolotl.cli.merge_lora import do_merge_lora

    with pytest.raises(ValueError, match="requires merge_method: legacy"):
        do_merge_lora(
            cfg=DictDefault(
                {
                    "model_config_type": "diffusion_gemma",
                    "merge_method": "memory_efficient",
                }
            )
        )


def test_trainable_token_adapter_routes_to_efficient_merge(monkeypatch, tmp_path):
    from axolotl.cli.merge_lora import do_merge_lora
    from axolotl.cli.utils.lora_merge import adapter_has_trainable_token_deltas

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 2, "trainable_token_indices": [7]}),
        encoding="utf-8",
    )
    safetensors.torch.save_file(
        {
            "base_model.model.encoder.embed_tokens.token_adapter.trainable_tokens_delta": torch.ones(
                1, 8
            )
        },
        adapter / "adapter_model.safetensors",
    )
    calls = []
    monkeypatch.setattr(
        "axolotl.cli.merge_lora._do_merge_lora_legacy",
        lambda *, cfg: calls.append("legacy"),
    )
    monkeypatch.setattr(
        "axolotl.cli.merge_lora._do_merge_lora_efficient",
        lambda *, cfg: calls.append("memory_efficient"),
    )
    cfg = DictDefault(
        {
            "model_config_type": "nemotron_labs_diffusion",
            "merge_method": None,
            "lora_model_dir": str(adapter),
        }
    )
    do_merge_lora(cfg=cfg)
    assert calls == ["memory_efficient"]

    cfg.merge_method = "memory_efficient"
    do_merge_lora(cfg=cfg)
    assert calls == ["memory_efficient", "memory_efficient"]

    missing_header = tmp_path / "missing-header"
    missing_header.mkdir()
    (missing_header / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 2}), encoding="utf-8"
    )
    safetensors.torch.save_file(
        {
            "base_model.model.encoder.embed_tokens.token_adapter.trainable_tokens_delta": torch.ones(
                1, 8
            )
        },
        missing_header / "adapter_model.safetensors",
    )
    with pytest.raises(ValueError, match="disagree"):
        adapter_has_trainable_token_deltas(missing_header)

    missing_delta = tmp_path / "missing-delta"
    missing_delta.mkdir()
    (missing_delta / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 2, "trainable_token_indices": [7]}),
        encoding="utf-8",
    )
    safetensors.torch.save_file(
        {
            "base_model.model.encoder.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(
                2, 8
            )
        },
        missing_delta / "adapter_model.safetensors",
    )
    with pytest.raises(ValueError, match="disagree"):
        adapter_has_trainable_token_deltas(missing_delta)


def test_native_generation_dispatches_and_preserves_legacy_seam():
    from types import SimpleNamespace

    from axolotl.model_support.native_generation import generate_for_model

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return ":".join(map(str, ids))

    class Gemma:
        config = SimpleNamespace(model_type="diffusion_gemma")

        def generate(self, **kwargs):
            assert kwargs["max_denoising_steps"] == 7
            return torch.tensor([[1, 2, 3]])

    result = generate_for_model(
        Gemma(),
        Tokenizer(),
        torch.tensor([[1]]),
        7,
        0.0,
        0,
        legacy_generate=lambda *args, **kwargs: pytest.fail("legacy called"),
        mode="completion",
        completion_tokens=2,
    )
    assert result["generated_ids"] == [1, 2, 3]

    class Unknown:
        config = SimpleNamespace(model_type="unknown")

    legacy = {"generated_ids": [9]}
    assert (
        generate_for_model(
            Unknown(),
            Tokenizer(),
            torch.tensor([[1]]),
            1,
            0.0,
            0,
            legacy_generate=lambda *args, **kwargs: legacy,
        )
        is legacy
    )


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


class _NativeTokenizer:
    def decode(self, ids, **kwargs):
        del kwargs
        return ":".join(map(str, ids))


def test_native_generation_runs_tiny_diffusion_gemma_model_output():
    from axolotl.model_support.native_generation import generate_for_model

    model = _tiny_model().eval()
    model.generation_config.return_dict_in_generate = True
    result = generate_for_model(
        model,
        _NativeTokenizer(),
        torch.tensor([[2, 3]]),
        1,
        0.0,
        0,
        legacy_generate=lambda *args, **kwargs: pytest.fail("legacy called"),
        mode="completion",
        completion_tokens=2,
    )
    assert result["orig_ids"] == [2, 3]
    assert result["generated_ids"][:2] == [2, 3]


@pytest.mark.parametrize("steps", [1, 2])
def test_native_diffusion_gemma_infill_preserves_pinned_canvas_tokens(steps):
    from axolotl.model_support.diffusion_gemma.modeling import (
        AxolotlDiffusionGemmaForBlockDiffusion,
    )
    from axolotl.model_support.native_generation import generate_for_model

    base = _tiny_model().eval()
    model = AxolotlDiffusionGemmaForBlockDiffusion(base.config).eval()
    model.load_state_dict(base.state_dict())
    original = torch.tensor([[2, 3, 4, 5]])
    selected = torch.tensor([[False, True, False, True]])
    torch.manual_seed(19)
    result = generate_for_model(
        model,
        _NativeTokenizer(),
        original,
        steps,
        0.0,
        0,
        legacy_generate=lambda *args, **kwargs: pytest.fail("legacy called"),
        mode="random",
        infill_mask=selected,
    )
    assert result["active_canvas_start"] == 1
    assert result["active_canvas_end"] == 4
    assert result["masked_positions"] == [1, 3]
    assert len(result["generated_ids"]) == original.shape[1]
    assert result["generated_ids"][0] == 2
    assert result["generated_ids"][2] == 4

    with pytest.raises(ValueError, match="outside the active canvas"):
        generate_for_model(
            model,
            _NativeTokenizer(),
            original,
            steps,
            0.0,
            0,
            legacy_generate=lambda *args, **kwargs: pytest.fail("legacy called"),
            mode="random",
            infill_mask=torch.tensor([[True, False, False, False]]),
        )


def test_native_generation_runs_tiny_dream_tensor_output():
    from transformers import AutoConfig

    from axolotl.model_support.dream import _model_class
    from axolotl.model_support.native_generation import generate_for_model

    source = _native_source_or_skip("dream")
    config = AutoConfig.from_pretrained(
        source, trust_remote_code=True, local_files_only=True
    )
    for name, value in {
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "max_window_layers": 1,
        "bos_token_id": 1,
        "eos_token_id": 1,
        "pad_token_id": 1,
        "mask_token_id": 2,
        "use_cache": False,
    }.items():
        setattr(config, name, value)
    config._name_or_path = str(source)
    model = _model_class().from_config(config, trust_remote_code=True).eval()
    result = generate_for_model(
        model,
        _NativeTokenizer(),
        torch.tensor([[1, 3]]),
        1,
        0.0,
        0,
        legacy_generate=lambda *args, **kwargs: pytest.fail("legacy called"),
        mode="completion",
        completion_tokens=1,
    )
    assert result["orig_ids"] == [1, 3]
    assert result["generated_ids"][:2] == [1, 3]
    torch.manual_seed(0)
    infilled = generate_for_model(
        model,
        _NativeTokenizer(),
        torch.tensor([[1, 3]]),
        1,
        0.0,
        0,
        legacy_generate=lambda *args, **kwargs: pytest.fail("legacy called"),
        mode="random",
        target_mask_ratio=0.5,
    )
    assert len(infilled["generated_ids"]) == 2
    assert infilled["masked_tokens"] == 1
    assert len(infilled["masked_positions"]) == 1


def test_native_cli_uses_completion_by_default_and_routes_native_infill(monkeypatch):
    from types import SimpleNamespace

    import axolotl.cli.utils.diffusion as diffusion_utils

    calls = []
    model = SimpleNamespace(config=SimpleNamespace(model_type="diffusion_gemma"))
    tokenizer = type(
        "Tokenizer",
        (),
        {
            "__call__": lambda self, *args, **kwargs: {
                "input_ids": torch.tensor([[2, 3]])
            }
        },
    )()
    cfg = DictDefault({"device": "cpu"})
    monkeypatch.setattr(
        diffusion_utils,
        "get_diffusion_config",
        lambda cfg: SimpleNamespace(
            num_diffusion_steps=3,
            generation_temperature=0.0,
        ),
    )
    monkeypatch.setattr(
        diffusion_utils,
        "generate_for_model",
        lambda *args, **kwargs: (
            calls.append(kwargs) or {"generated_ids": [2, 3, 4], "masked_positions": []}
        ),
    )
    output = diffusion_utils.run_diffusion(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        prompt="hello",
        chat_template_str=None,
    )
    assert output["generated_ids"] == [2, 3, 4]
    assert calls == [
        {
            "legacy_generate": diffusion_utils.generate,
            "mode": "completion",
            "completion_tokens": 0,
            "target_mask_ratio": None,
        }
    ]
    diffusion_utils.run_diffusion(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        prompt="hello",
        chat_template_str=None,
        mode="random",
        target_mask_ratio=0.5,
    )
    assert calls[-1]["mode"] == "random"
    assert calls[-1]["target_mask_ratio"] == 0.5
    dream = SimpleNamespace(config=SimpleNamespace(model_type="Dream"))
    diffusion_utils.run_diffusion(
        model=dream,
        tokenizer=tokenizer,
        cfg=cfg,
        prompt="hello",
        chat_template_str=None,
        mode="random",
        target_mask_ratio=0.5,
    )
    assert calls[-1]["mode"] == "random"
    assert calls[-1]["target_mask_ratio"] == 0.5


def test_native_cli_commands_preserve_completion_and_reject_mask_infill():
    from axolotl.cli.utils.diffusion import _parse_commands

    assert _parse_commands("plain prompt") == (None, 0, None, "plain prompt")
    assert _parse_commands(":complete 7 plain prompt") == (
        "completion",
        7,
        None,
        "plain prompt",
    )
    assert _parse_commands(":mask 0.5 plain prompt") == (
        "random",
        0,
        0.5,
        "plain prompt",
    )


def test_native_generation_callback_samples_sft_completion_suffix():
    from types import SimpleNamespace

    from axolotl.model_support.native_generation import generate_native_samples

    calls = []

    class Gemma(torch.nn.Module):
        config = SimpleNamespace(model_type="diffusion_gemma")

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def generate(self, **kwargs):
            calls.append(kwargs)
            return torch.cat(
                [
                    kwargs["input_ids"],
                    torch.tensor([[8, 9]], device=kwargs["input_ids"].device),
                ],
                dim=1,
            )

    batch = {
        "input_ids": torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]),
        "labels": torch.tensor(
            [[-100, -100, -100, -100, -100, -100, -100, -100, 10, 11]]
        ),
        "attention_mask": torch.ones((1, 10), dtype=torch.long),
    }
    samples = generate_native_samples(
        Gemma(),
        _NativeTokenizer(),
        dataloader=[batch] * 10,
        num_generation_samples=1,
        max_length=10,
        num_diffusion_steps=3,
        temperature=0.0,
    )
    assert len(samples) == 1
    assert samples[0]["generated_ids"] == [2, 3, 4, 5, 6, 7, 8, 9, 8, 9]
    assert len(calls) == 1
    assert calls[0]["max_denoising_steps"] == 3
    assert calls[0]["max_new_tokens"] == 2


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


def test_dream_generation_config_resolves_remote_utils_through_peft(monkeypatch):
    import sys
    import types
    from types import SimpleNamespace

    from axolotl.model_support.native_generation import _dream_generation_config

    generation_utils = types.ModuleType("remote_dream.generation_utils")
    generation_utils.DreamGenerationConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "remote_dream.generation_utils", generation_utils)

    class RemoteDream(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="Dream",
                mask_token_id=5,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
            self.proj = torch.nn.Linear(4, 4)

    RemoteDream.__module__ = "remote_dream.modeling_dream"

    class CompatDream(RemoteDream):
        pass

    model = get_peft_model(
        CompatDream(), LoraConfig(r=2, lora_alpha=4, target_modules=["proj"])
    )
    config = _dream_generation_config(
        model, max_new_tokens=3, num_diffusion_steps=4, temperature=0.0
    )

    assert config.mask_token_id == 5
    assert config.steps == 4
    assert config.max_new_tokens == 3


def test_axolotl_subclass_built_first_keeps_grouped_mm_experts():
    from axolotl.model_support.diffusion_gemma.modeling import (
        AxolotlDiffusionGemmaForBlockDiffusion,
    )

    cls = AxolotlDiffusionGemmaForBlockDiffusion
    for attr in (
        "_can_set_experts_implementation_cached_value",
        "_can_set_attn_implementation_cached_value",
    ):
        if attr in vars(cls):
            delattr(cls, attr)
    fresh_config = DiffusionGemmaConfig(**_tiny_model().config.to_dict())

    first = cls(fresh_config)

    assert first.config._experts_implementation == "grouped_mm"
    assert "_can_set_experts_implementation_cached_value" not in vars(cls)
    cls(_tiny_model().config)


def test_diffusion_gemma_lora_targets_may_be_listed_per_path():
    validate_native_diffusion_lora(
        _native_cfg(
            lora_target_modules=[
                "model.encoder.language_model.layers.0.self_attn.q_proj",
                "model.decoder.layers.0.self_attn.q_proj",
            ]
        ),
        model_name="DiffusionGemma",
    )
    with pytest.raises(ValueError, match="cover both"):
        validate_native_diffusion_lora(
            _native_cfg(
                lora_target_modules=["model.decoder.layers.0.self_attn.q_proj"]
            ),
            model_name="DiffusionGemma",
        )
