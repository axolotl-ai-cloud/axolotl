"""Native Nemotron packed varlen dispatch against explicit dense attention."""

import importlib

import pytest
import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from axolotl.core.trainers.diffusion_lm import varlen
from axolotl.core.trainers.diffusion_lm.backends.full_sequence import (
    FullSequenceBackend,
)
from axolotl.model_support.nemotron_diffusion.compat import resolve_nemotron_model_class

from tests.native_source_fixtures import native_source_fixture_path


@pytest.fixture
def model():
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
        num_hidden_layers=2,
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
    return resolve_nemotron_model_class(str(source))(config).eval()


def test_varlen_native_logits_gradients_and_mask_bypass(model, monkeypatch):
    ids = torch.tensor([[7, 8, 9, 10, 11, 12, 0]])
    documents = torch.tensor([[0, 0, 1, 1, 1, 1, -1]])
    valid = documents >= 0
    dense = FullSequenceBackend(mask_token_id=100, attention_backend="dense")
    packed = FullSequenceBackend(mask_token_id=100, attention_backend="varlen")
    dense_pack = dense.pack(ids, documents, valid)
    varlen_pack = packed.pack(ids, documents, valid)
    reference = dense.forward(model, dense_pack, ids).logits
    reference[valid].square().mean().backward()
    expected_grads = {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    calls = []

    def cpu_varlen(q, k, v, cu_q, cu_k, max_q, max_k, **kwargs):
        calls.append((cu_q.clone(), cu_k.clone()))
        outputs = []
        offsets = cu_q.tolist()
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            query = q[start:end]
            key = k[start:end].repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            value = v[start:end].repeat_interleave(q.shape[1] // v.shape[1], dim=1)
            scores = torch.einsum("qhd,khd->hqk", query, key) * kwargs["scale"]
            outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), value))
        return torch.cat(outputs)

    def forbidden(*args, **kwargs):
        pytest.fail("varlen must not invoke the native mask builder")

    monkeypatch.setattr(varlen, "varlen_attn", cpu_varlen)
    source = importlib.import_module(type(model.encoder).__module__)
    monkeypatch.setattr(source, "create_causal_mask", forbidden)
    monkeypatch.setattr(source, "create_sliding_window_causal_mask", forbidden)
    output = packed.forward(model, varlen_pack, ids).logits
    torch.testing.assert_close(output[valid], reference[valid], atol=1e-6, rtol=1e-5)
    output[valid].square().mean().backward()
    assert len(calls) == model.config.num_hidden_layers
    assert all(q.tolist() == k.tolist() == [0, 2, 6] for q, k in calls)
    for name, param in model.named_parameters():
        if name in expected_grads:
            torch.testing.assert_close(
                param.grad, expected_grads[name], atol=1e-6, rtol=1e-4
            )
    changed = ids.clone()
    changed[:, 2:6] = torch.tensor([20, 21, 22, 23])
    perturbed = packed.forward(model, varlen_pack, changed).logits
    torch.testing.assert_close(output[:, :2], perturbed[:, :2], atol=0, rtol=0)


@pytest.mark.parametrize(
    "conflict",
    [
        {"attention_mask": torch.ones(1, 4)},
        {"use_causal_mask": True},
        {"use_cache": True},
        {"labels": torch.ones(1, 4, dtype=torch.long)},
    ],
)
def test_varlen_boundary_rejects_conflicts(model, conflict):
    docs = torch.tensor([[0, 0, 1, 1]])
    metadata = varlen.build_varlen_metadata(docs, docs >= 0)
    with pytest.raises(ValueError, match="varlen requires"):
        model(
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            diffusion_varlen=metadata,
            **conflict,
        )


def test_selected_logits_match_dense_projection_and_preserve_default_path(model):
    ids = torch.tensor([[7, 8, 9, 10, 11, 12]])
    rows = torch.tensor([[0, 0, -1], [0, 0, -1]])
    positions = torch.tensor([[1, 4, -1], [2, 5, -1]])

    dense = model(input_ids=ids, use_cache=False).logits
    selected = model(
        input_ids=ids,
        use_cache=False,
        axolotl_selected_logits=(rows, positions),
    )

    assert dense.shape == (1, 6, model.config.vocab_size)
    assert selected.axolotl_selected_logits is True
    torch.testing.assert_close(
        selected.logits, dense[rows.clamp_min(0), positions.clamp_min(0)]
    )

    selected.logits[..., :4].sum().backward()
    selected_grad = model.diffusion_head.weight.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    dense[rows.clamp_min(0), positions.clamp_min(0)][..., :4].sum().backward()
    torch.testing.assert_close(selected_grad, model.diffusion_head.weight.grad)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_encoder_stays_packed_across_layers(model, monkeypatch, checkpointing):
    ids = torch.tensor([[0, 7, 100, 9, 0, 10, 11], [12, 13, 14, 15, 16, 0, 0]])
    docs = torch.tensor([[-1, 0, 0, 0, -1, 1, 1], [0, 0, 1, 1, 1, -1, -1]])
    valid = docs >= 0
    dense = FullSequenceBackend(mask_token_id=100, attention_backend="dense")
    packed = FullSequenceBackend(mask_token_id=100, attention_backend="varlen")
    references = []
    for row in range(2):
        row_ids, row_docs = ids[row : row + 1], docs[row : row + 1]
        row_pack = dense.pack(row_ids, row_docs, row_docs >= 0)
        references.append(dense.forward(model, row_pack, row_ids).logits)
    reference = torch.cat(references)
    reference[valid].square().mean().backward()
    expected_grads = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)

    def cpu_varlen(q, k, v, cu_q, cu_k, max_q, max_k, **kwargs):
        assert q.shape == (10, 4, 16)
        assert k.shape == v.shape == (10, 2, 16)
        assert kwargs["enable_gqa"]
        assert cu_q.tolist() == cu_k.tolist() == [0, 3, 5, 7, 10]
        outputs = []
        offsets = cu_q.tolist()
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            outputs.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q[start:end].transpose(0, 1),
                    k[start:end].transpose(0, 1),
                    v[start:end].transpose(0, 1),
                    scale=kwargs["scale"],
                    enable_gqa=True,
                ).transpose(0, 1)
            )
        return torch.cat(outputs)

    monkeypatch.setattr(varlen, "varlen_attn", cpu_varlen)
    shapes = []

    def record(module, args):
        shapes.append(args[0].shape)

    hooks = [
        module.register_forward_pre_hook(record)
        for layer in model.encoder.layers
        for module in (layer.self_attn.q_proj, layer.mlp)
    ]
    if checkpointing:
        model.train()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    packed_batch = packed.pack(ids, docs, valid)
    output = packed.forward(model, packed_batch, ids).logits
    torch.testing.assert_close(output[valid], reference[valid], atol=1e-6, rtol=1e-5)
    output[valid].square().mean().backward()
    assert len(shapes) >= 2 * model.config.num_hidden_layers
    assert all(shape == (1, 10, model.config.hidden_size) for shape in shapes)
    for hook in hooks:
        hook.remove()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.grad, expected_grads[name], atol=1e-6, rtol=1e-4
        )

    hidden = packed.forward(
        model, packed_batch, ids, model_kwargs={"output_last_hidden_states_only": True}
    )
    assert hidden.last_hidden_state.shape == (2, 7, model.config.hidden_size)
    assert torch.count_nonzero(hidden.last_hidden_state[~valid]) == 0
    encoder_output = model.encoder(
        inputs_embeds=model.get_input_embeddings()(ids),
        position_ids=packed_batch["position_ids"],
        diffusion_varlen=packed_batch["diffusion_varlen"],
        use_cache=False,
        output_hidden_states=True,
    )
    torch.testing.assert_close(
        encoder_output.last_hidden_state, hidden.last_hidden_state
    )
    if encoder_output.hidden_states is not None:
        assert all(
            state.shape == hidden.last_hidden_state.shape
            for state in encoder_output.hidden_states
        )
    rows, positions = torch.tensor([[0, 1]]), torch.tensor([[2, 4]])
    selected = packed.forward(
        model,
        packed_batch,
        ids,
        model_kwargs={"axolotl_selected_logits": (rows, positions)},
    )
    torch.testing.assert_close(
        selected.logits, reference[rows, positions], atol=1e-6, rtol=1e-5
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="varlen_attn needs CUDA")
def test_packed_encoder_native_cuda_parity(model):
    model = model.to(device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[0, 7, 100, 9, 0, 10, 11]], device="cuda")
    docs = torch.tensor([[-1, 0, 0, 0, -1, 1, 1]], device="cuda")
    valid = docs >= 0
    dense = FullSequenceBackend(mask_token_id=100, attention_backend="dense")
    packed = FullSequenceBackend(mask_token_id=100, attention_backend="varlen")
    reference = dense.forward(model, dense.pack(ids, docs, valid), ids).logits
    reference[valid].float().square().mean().backward()
    expected = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    output = packed.forward(model, packed.pack(ids, docs, valid), ids).logits
    torch.testing.assert_close(output[valid], reference[valid], atol=5e-3, rtol=5e-2)
    output[valid].float().square().mean().backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, expected[name], atol=5e-4, rtol=5e-2)
