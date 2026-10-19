"""Full-sequence variable-length metadata routing."""

import pytest
import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from axolotl.core.trainers.diffusion_lm import varlen
from axolotl.core.trainers.diffusion_lm.backends import full_sequence
from axolotl.model_support.nemotron_diffusion.compat import resolve_nemotron_model_class

from tests.native_source_fixtures import native_source_fixture_path


def test_position_reset_uses_no_tensor_scalar_extraction(monkeypatch):
    documents = torch.tensor([[4, 4, -1, 4, 4, 8], [0, 0, 0, 0, 0, 0]])
    validity = torch.tensor(
        [[True, True, False, True, True, True], [True, False, True, True, False, True]]
    )

    def forbidden(*args, **kwargs):
        pytest.fail("position metadata must not extract tensor scalars")

    with monkeypatch.context() as patch:
        for method in ("item", "__int__", "__bool__"):
            patch.setattr(torch.Tensor, method, forbidden)
        positions = full_sequence.reset_position_ids(documents, validity)

    assert positions.tolist() == [[0, 1, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0]]


def test_varlen_pack_never_builds_dense_mask_and_routes_metadata(monkeypatch):
    monkeypatch.setattr(
        full_sequence,
        "create_packed_document_attention_mask",
        lambda *args: pytest.fail("varlen must not build a dense document mask"),
    )
    backend = full_sequence.FullSequenceBackend(
        mask_token_id=100, attention_backend="varlen"
    )
    tokens = torch.tensor([[5, 6, 7, 8, 0]])
    documents = torch.tensor([[0, 0, 1, 1, -1]])
    packed = backend.pack(tokens, documents, documents >= 0)
    assert packed["attention_mask"] is None
    assert packed["input_ids"].shape == tokens.shape
    assert packed["diffusion_varlen"].cu_seqlens.tolist() == [0, 2, 4]
    assert packed["position_ids"].tolist() == [[0, 1, 0, 1, 0]]

    class SupportedModel:
        supports_diffusion_varlen = True

        def __call__(self, **kwargs):
            return kwargs

    forwarded = backend.forward(SupportedModel(), packed, tokens)
    assert forwarded["diffusion_varlen"] is packed["diffusion_varlen"]
    assert forwarded["attention_mask"] is None
    assert forwarded["use_cache"] is False
    assert backend.forward(
        SupportedModel(),
        packed,
        tokens,
        model_kwargs={"output_last_hidden_states_only": True},
    )["output_last_hidden_states_only"]
    with pytest.raises(ValueError, match="routing keys: attention_mask"):
        backend.forward(
            SupportedModel(),
            packed,
            tokens,
            model_kwargs={"attention_mask": torch.ones(1, 1)},
        )

    class WrappedModel:
        def __init__(self):
            self.module = SupportedModel()
            self.called = False

        def __call__(self, **kwargs):
            self.called = True
            return self.module(**kwargs)

    wrapped = WrappedModel()
    assert (
        backend.forward(wrapped, packed, tokens)["diffusion_varlen"]
        is packed["diffusion_varlen"]
    )
    assert wrapped.called
    with pytest.raises(ValueError, match="does not support"):
        backend.forward(lambda **kwargs: kwargs, packed, tokens)


def test_varlen_hidden_only_matches_native_logits_and_gradients(monkeypatch):
    source = native_source_fixture_path("nemotron")
    if source is None:
        pytest.skip("native Nemotron source fixture unavailable")
    config_class = get_class_from_dynamic_module(
        "configuration_nemotron_labs_diffusion.NemotronLabsDiffusionConfig",
        str(source),
        local_files_only=True,
    )
    config = config_class(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        mask_token_id=100,
        dlm_paradigm="bidirectional",
        use_cache=False,
        rope_parameters={
            "llama_4_scaling_beta": 1.0,
            "original_max_position_embeddings": 1,
        },
    )
    config._attn_implementation = "eager"
    torch.manual_seed(42)
    model = resolve_nemotron_model_class(str(source))(config).train()
    backend = full_sequence.FullSequenceBackend(
        mask_token_id=100, attention_backend="varlen"
    )
    input_ids = torch.tensor([[7, 8, 9, 10, 11, 12, 0]])
    document_ids = torch.tensor([[0, 0, 1, 1, 1, 1, -1]])
    packed = backend.pack(input_ids, document_ids, document_ids >= 0)

    def cpu_varlen(q, k, v, cu_q, cu_k, max_q, max_k, **kwargs):
        outputs = []
        for start, end in zip(cu_q[:-1].tolist(), cu_q[1:].tolist(), strict=True):
            query = q[start:end]
            key = k[start:end].repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            value = v[start:end].repeat_interleave(q.shape[1] // v.shape[1], dim=1)
            scores = torch.einsum("qhd,khd->hqk", query, key) * kwargs["scale"]
            outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), value))
        return torch.cat(outputs)

    monkeypatch.setattr(varlen, "varlen_attn", cpu_varlen)
    selected_positions = torch.tensor([1, 4])
    full_logits = backend.forward(model, packed, input_ids).logits
    expected = full_logits[0, selected_positions]
    expected.square().mean().backward()
    expected_grads = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)

    hidden = backend.forward(
        model,
        packed,
        input_ids,
        model_kwargs={"output_last_hidden_states_only": True},
    ).last_hidden_state
    selected_logits = model.get_output_embeddings()(hidden[0, selected_positions])
    torch.testing.assert_close(selected_logits, expected, atol=1e-6, rtol=1e-5)
    selected_logits.square().mean().backward()
    for name, expected_grad in expected_grads.items():
        torch.testing.assert_close(
            model.get_parameter(name).grad, expected_grad, atol=1e-6, rtol=1e-4
        )
