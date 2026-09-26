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


@pytest.mark.parametrize("loaded", [False, True])
def test_fla_adapter_training_fails_closed(loaded):
    from types import SimpleNamespace

    from axolotl.model_support import get_model_support
    from axolotl.utils.dict import DictDefault

    support = get_model_support("mamba")
    cfg = DictDefault(
        adapter="lora", overrides_of_model_config={"mamba_backend": "fla"}
    )
    with pytest.raises(ValueError, match="fused projections"):
        if loaded:
            support.post_model_load(cfg, SimpleNamespace(config=_config("mamba")))
        else:
            support.validate_cfg(cfg)
