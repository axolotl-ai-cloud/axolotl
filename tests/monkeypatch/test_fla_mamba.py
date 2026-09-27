"""FLA Mamba loading and packed-training compatibility."""

import copy

import pytest
import torch
from transformers import Mamba2Config, MambaConfig

from axolotl.model_support.mamba.loading import MambaModelLoader


def _config(family):
    cls = MambaConfig if family == "mamba" else Mamba2Config
    return cls(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        state_size=8,
        expand=2,
        num_heads=2,
        head_dim=32,
        n_groups=1,
        chunk_size=16,
        use_cache=False,
        tie_word_embeddings=True,
        mamba_backend="fla",
        time_step_rank=3,
        layer_norm_epsilon=1e-4,
    )


@pytest.mark.parametrize("family", ["mamba", "mamba2"])
@pytest.mark.parametrize("tied", [False, True])
def test_fla_checkpoint_roundtrip(family, tied, tmp_path, monkeypatch):
    pytest.importorskip("fla")
    from axolotl.loaders.utils import load_model_config
    from axolotl.utils.dict import DictDefault

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = _config(family)
    config.tie_word_embeddings = tied
    model = MambaModelLoader(config)
    assert model.config.norm_eps == 1e-4
    if family == "mamba":
        assert model.backbone.layers[0].mixer.dt_rank == 3
    cloned = copy.deepcopy(model)
    mixer = cloned.backbone.layers[0].mixer
    assert mixer._axolotl_fla_forward.__self__ is mixer
    assert (
        cloned.backbone.norm_f._axolotl_norm_forward.__self__ is cloned.backbone.norm_f
    )
    model.save_pretrained(tmp_path)
    config = load_model_config(DictDefault(base_model=str(tmp_path)))
    restored = MambaModelLoader.from_pretrained(tmp_path, config=config)
    assert restored.config.mamba_backend == "fla"
    assert (restored.lm_head.weight is restored.backbone.embeddings.weight) is tied
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA kernels require CUDA")
@pytest.mark.parametrize("family", ["mamba", "mamba2"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_fla_packing_gradients_and_cache(family, batch_size):
    pytest.importorskip("fla")
    torch.manual_seed(5)
    model = MambaModelLoader(_config(family)).cuda().to(torch.bfloat16)
    ids = torch.randint(0, 64, (batch_size, 48), device="cuda")
    position_ids = (
        torch.cat([torch.arange(17), torch.arange(31)])
        .cuda()[None]
        .expand(batch_size, -1)
    )
    denominator = batch_size * 46
    actual = model(
        input_ids=ids,
        position_ids=position_ids,
        labels=ids,
        num_items_in_batch=denominator,
    )
    tuple_output = model(
        input_ids=ids,
        position_ids=position_ids,
        labels=ids,
        num_items_in_batch=denominator,
        return_dict=False,
    )
    torch.testing.assert_close(tuple_output[0], actual.loss)
    torch.testing.assert_close(tuple_output[1], actual.logits)
    actual.loss.backward()
    gradients = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    pieces = [
        model(input_ids=part, labels=part, num_items_in_batch=denominator)
        for part in (ids[:, :17], ids[:, 17:])
    ]
    torch.testing.assert_close(
        actual.logits, torch.cat([p.logits for p in pieces], 1), atol=0.01, rtol=0.02
    )
    torch.testing.assert_close(
        actual.loss, sum(p.loss for p in pieces), atol=0.01, rtol=0.01
    )
    sum(p.loss for p in pieces).backward()
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(
                parameter.grad, gradients[name], atol=0.002, rtol=0.03
            )
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    checkpointed = model(
        input_ids=ids,
        position_ids=position_ids,
        labels=ids,
        num_items_in_batch=denominator,
    )
    checkpointed.loss.backward()
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(
                parameter.grad, gradients[name], atol=0.002, rtol=0.03
            )
    model.gradient_checkpointing_disable()
    model.eval()
    with torch.no_grad():
        prefix = model(input_ids=ids[:, :17], use_cache=True)
        next_token = model(
            input_ids=ids[:, 17:18],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )
        dense = model(input_ids=ids[:, :18], use_cache=False)
        torch.testing.assert_close(
            next_token.logits, dense.logits[:, -1:], atol=0.01, rtol=0.02
        )
        model.generation_config.eos_token_id = None
        generated = model.generate(
            ids[:, :17], max_new_tokens=2, do_sample=False, use_cache=True
        )
        assert generated.shape == (batch_size, 19)


@pytest.mark.parametrize("family", ["mamba", "mamba2"])
def test_model_support_keeps_native_default(family):
    from axolotl.model_support import get_model_support

    config = _config(family)
    del config.mamba_backend
    loader = get_model_support(family).get_auto_model_cls()
    model = loader(config)
    assert type(model).__module__.startswith("transformers.models.")


def test_invalid_backend_fails_before_fla_loading():
    config = _config("mamba")
    config.mamba_backend = "invalid"
    with pytest.raises(ValueError, match="mamba_backend"):
        MambaModelLoader(config)


def test_fla_rejects_triton_convolution(monkeypatch):
    pytest.importorskip("fla")
    monkeypatch.setenv("FLA_CONV_BACKEND", "triton")
    with pytest.raises(ValueError, match="FLA_CONV_BACKEND=cuda"):
        MambaModelLoader(_config("mamba2"))


@pytest.mark.parametrize("target", ["in_proj", "out_proj", "x_proj"])
def test_fla_adapter_target_validation(target, monkeypatch):
    pytest.importorskip("fla")
    from peft import LoraConfig, get_peft_model

    from axolotl.model_support import get_model_support
    from axolotl.utils.dict import DictDefault

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    base = MambaModelLoader(_config("mamba"))
    support = get_model_support("mamba")
    cfg = DictDefault(
        adapter="lora", overrides_of_model_config={"mamba_backend": "fla"}
    )
    support.validate_cfg(cfg)
    if target == "x_proj":
        with pytest.raises(ValueError, match="targets must be"):
            get_peft_model(
                base, LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=[target])
            )
    else:
        model = get_peft_model(
            base, LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=[target])
        )
        support.post_model_load(cfg, model)
    from transformers import MambaForCausalLM

    native = MambaForCausalLM(_config("mamba"))
    with pytest.raises(ValueError, match="incompatible"):
        get_peft_model(
            native, LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=["out_proj"])
        )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA kernels require CUDA")
@pytest.mark.parametrize("family", ["mamba", "mamba2"])
@pytest.mark.parametrize(
    "targets", [["in_proj"], ["out_proj"], ["in_proj", "out_proj"]]
)
def test_fla_lora_gradients_packing_and_reload(family, targets, tmp_path):
    pytest.importorskip("fla")
    from peft import LoraConfig, PeftModel, get_peft_model

    torch.manual_seed(42)
    base = MambaModelLoader(_config(family)).cuda().to(torch.bfloat16)
    original = copy.deepcopy(base)
    model = get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM", r=4, target_modules=targets, lora_dropout=0.1
        ),
    )
    for module in model.modules():
        if hasattr(module, "lora_dropout"):
            module.lora_dropout["default"].p = 0.0
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.05)
    ids = torch.randint(0, 64, (2, 48), device="cuda")
    positions = (
        torch.cat([torch.arange(17), torch.arange(31)]).cuda()[None].expand(2, -1)
    )
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    actual = model(
        input_ids=ids, position_ids=positions, labels=ids, num_items_in_batch=92
    )
    actual.loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0, (
                name
            )
            gradients[name] = parameter.grad.clone()
        else:
            assert parameter.grad is None, name
    model.zero_grad(set_to_none=True)
    parts = [
        model(input_ids=x, labels=x, num_items_in_batch=92)
        for x in (ids[:, :17], ids[:, 17:])
    ]
    sum(x.loss for x in parts).backward()
    torch.testing.assert_close(
        actual.logits, torch.cat([x.logits for x in parts], 1), atol=0.02, rtol=0.03
    )
    reference_logits = torch.cat([x.logits for x in parts], 1).float()
    assert (
        actual.logits.float() - reference_logits
    ).norm() / reference_logits.norm() < 0.02
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(
                parameter.grad, gradients[name], atol=0.002, rtol=0.04
            )
    model.gradient_checkpointing_disable()
    if len(targets) == 2:
        dropouts = [
            module.lora_dropout["default"]
            for module in model.modules()
            if hasattr(module, "lora_dropout")
        ]
        for dropout in dropouts:
            dropout.p = 0.5
        with torch.no_grad():
            first = model(input_ids=ids).logits
            second = model(input_ids=ids).logits
        assert not torch.equal(first, second)
        for dropout in dropouts:
            dropout.p = 0.0
    model.eval()
    with torch.no_grad():
        expected = model(input_ids=ids).logits
        with model.disable_adapter():
            disabled = model(input_ids=ids).logits
        assert (expected - disabled).float().norm() > 0.01
        prefix = model(input_ids=ids[:, :17], use_cache=True)
        decoded = model(
            input_ids=ids[:, 17:18],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        ).logits
        torch.testing.assert_close(
            decoded, model(input_ids=ids[:, :18]).logits[:, -1:], atol=0.01, rtol=0.03
        )
        model.save_pretrained(tmp_path)
        restored = PeftModel.from_pretrained(original, tmp_path).eval()
        torch.testing.assert_close(restored(input_ids=ids).logits, expected)
        merged = restored.merge_and_unload().eval()
        merged_logits = merged(input_ids=ids).logits
        torch.testing.assert_close(merged_logits, expected, atol=0.02, rtol=0.03)
        assert (
            merged_logits.float() - expected.float()
        ).norm() / expected.float().norm() < 0.02


@pytest.mark.parametrize("family", ["mamba", "mamba2"])
@pytest.mark.parametrize("dtype_key", ["dtype", "torch_dtype"])
def test_fla_from_config_loader_options(family, dtype_key, monkeypatch):
    pytest.importorskip("fla")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous = torch.get_default_dtype()
    model = MambaModelLoader.from_config(
        _config(family),
        trust_remote_code=False,
        attn_implementation="flash_attention_2",
        experts_implementation="eager",
        **{dtype_key: torch.bfloat16},
    )
    assert model.get_input_embeddings().weight.dtype == torch.bfloat16
    assert model.config.dtype == torch.bfloat16
    assert torch.get_default_dtype() == previous


@pytest.mark.parametrize("mapping", [False, True])
@pytest.mark.parametrize(
    "options",
    [
        {"adapter": "qlora"},
        {"adapter": "lora", "lora_target_parameters": ["in_proj.weight"]},
    ],
)
def test_saved_fla_config_validates_before_model_load(mapping, options):
    from axolotl.model_support import get_model_support
    from axolotl.model_support.profile import (
        ModelHookContext,
        ModelHookPhase,
        run_model_support_hooks,
    )
    from axolotl.utils.dict import DictDefault

    config = _config("mamba")
    if mapping:
        config = config.to_dict()
    with pytest.raises(ValueError, match="FLA Mamba"):
        run_model_support_hooks(
            get_model_support("mamba"),
            ModelHookPhase.CONFIGURE_RUN,
            ModelHookContext(cfg=DictDefault(options), model_config=config),
        )


@pytest.mark.parametrize("family", ["mamba", "mamba2"])
@pytest.mark.parametrize("with_labels", [False, True])
@pytest.mark.parametrize("selection", [0, 2, torch.tensor([0, 3])])
def test_fla_logits_selection_and_output_order(
    family, with_labels, selection, monkeypatch
):
    pytest.importorskip("fla")
    from transformers.modeling_outputs import CausalLMOutputWithPast

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = MambaModelLoader(_config(family))
    logits = torch.randn(1, 5, model.config.vocab_size)
    calls = []

    def forward(self, *, logits_to_keep, **kwargs):
        calls.append(logits_to_keep)
        return CausalLMOutputWithPast(logits=logits[:, -logits_to_keep:])

    monkeypatch.setattr(type(model).__bases__[1], "forward", forward)
    ids = torch.randint(0, model.config.vocab_size, (1, 5))
    labels = ids if with_labels else None
    output = model(input_ids=ids, labels=labels, logits_to_keep=selection)
    as_tuple = model(
        input_ids=ids, labels=labels, logits_to_keep=selection, return_dict=False
    )
    expected_keep = selection if not with_labels and isinstance(selection, int) else 0
    assert calls == [expected_keep, expected_keep]
    expected = (
        logits[:, selection]
        if isinstance(selection, torch.Tensor)
        else logits[:, -selection:]
    )
    torch.testing.assert_close(output.logits, expected)
    assert list(output)[0] == ("loss" if with_labels else "logits")
    for value, tuple_value in zip(output.to_tuple(), as_tuple, strict=True):
        torch.testing.assert_close(value, tuple_value)
    if with_labels:
        torch.testing.assert_close(
            output[0],
            model.loss_function(
                logits=logits, labels=ids, vocab_size=model.config.vocab_size
            ),
        )
