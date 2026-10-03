import pytest
import torch

from axolotl.integrations.diffusion.lm import varlen


def test_metadata_uses_document_runs_and_distinguishes_rows():
    document_ids = torch.tensor([[7, 7, 8, -1, 9, 9], [7, 7, -1, 8, 8, -1]])
    semantic = document_ids >= 0

    metadata = varlen.build_varlen_metadata(document_ids, semantic)

    assert metadata.flat_indices.tolist() == [0, 1, 2, 4, 5, 6, 7, 9, 10]
    assert metadata.flat_indices.dtype is torch.long
    assert metadata.cu_seqlens.tolist() == [0, 2, 3, 5, 7, 9]
    assert metadata.cu_seqlens.dtype is torch.int32
    assert metadata.max_seqlen == 6
    assert metadata.total_tokens == 9


def test_metadata_rejects_discontiguous_document_within_a_row():
    with pytest.raises(RuntimeError, match="contiguous run"):
        varlen.build_varlen_metadata(
            torch.tensor([[0, 0, 1, 0]]), torch.tensor([[True, True, True, True]])
        )


@pytest.mark.parametrize(
    ("document_ids", "semantic", "message"),
    [
        (torch.tensor([0, 1]), torch.tensor([True, True]), "batch, sequence"),
        (torch.tensor([[0, 1]]), torch.tensor([[True]]), "matching shapes"),
        (torch.tensor([[0.0, 1.0]]), torch.tensor([[True, True]]), "integer dtype"),
        (torch.tensor([[0, -1]]), torch.tensor([[True, True]]), "nonnegative document"),
        (torch.tensor([[-1, -1]]), torch.tensor([[False, False]]), "at least one"),
    ],
)
def test_metadata_shape_and_semantic_validation(document_ids, semantic, message):
    with pytest.raises((TypeError, ValueError, RuntimeError), match=message):
        varlen.build_varlen_metadata(document_ids, semantic)


def test_metadata_uses_width_upperbound_or_explicit_collator_maximum():
    document_ids = torch.tensor([[0, 0, 1, -1]])
    metadata = varlen.build_varlen_metadata(document_ids, document_ids >= 0)
    assert metadata.max_seqlen == 4
    exact = varlen.build_varlen_metadata(document_ids, document_ids >= 0, max_seqlen=2)
    assert exact.max_seqlen == 2
    with pytest.raises(ValueError, match="sequence width"):
        varlen.build_varlen_metadata(document_ids, document_ids >= 0, max_seqlen=5)
    with pytest.raises(RuntimeError, match="cover every document run"):
        varlen.build_varlen_metadata(
            torch.tensor([[0, 0, 0]]),
            torch.ones(1, 3, dtype=torch.bool),
            max_seqlen=2,
        )


@pytest.mark.parametrize(
    ("dtype", "boundary"),
    [
        (torch.uint8, 255),
        (torch.int8, 127),
        (torch.int64, torch.iinfo(torch.int64).max),
    ],
)
def test_metadata_normalizes_integer_boundaries_without_overflow(dtype, boundary):
    document_ids = torch.tensor([[boundary, boundary, boundary - 1]], dtype=dtype)
    metadata = varlen.build_varlen_metadata(
        document_ids, torch.ones_like(document_ids, dtype=torch.bool)
    )

    assert metadata.cu_seqlens.tolist() == [0, 2, 3]


def test_metadata_never_extracts_a_tensor_scalar(monkeypatch):
    document_ids = torch.tensor([[0, 0, -1, 1], [0, -1, 1, 1]])

    def fail_scalar_extraction(*args, **kwargs):
        pytest.fail("metadata construction extracted a tensor scalar")

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "item", fail_scalar_extraction)
        context.setattr(torch.Tensor, "__int__", fail_scalar_extraction)
        context.setattr(torch.Tensor, "__bool__", fail_scalar_extraction)
        metadata = varlen.build_varlen_metadata(document_ids, document_ids >= 0)

    assert metadata.cu_seqlens.tolist() == [0, 2, 3, 4, 6]


def test_gather_scatter_preserves_gradients_and_zeros_padding():
    document_ids = torch.tensor([[0, 0, -1, 1], [0, -1, 1, 1]])
    metadata = varlen.build_varlen_metadata(document_ids, document_ids >= 0)
    value = torch.arange(2 * 3 * 4 * 2, dtype=torch.float32).reshape(2, 3, 4, 2)
    value.requires_grad_()

    gathered = varlen.gather_thd(value, metadata)
    restored = varlen.scatter_thd(gathered, metadata)
    restored.sum().backward()

    assert gathered.shape == (6, 3, 2)
    assert restored.shape == (2, 4, 3, 2)
    assert torch.equal(restored[0, 0], value.detach()[0, :, 0])
    assert torch.equal(restored[0, 2], torch.zeros_like(restored[0, 2]))
    expected_grad = (document_ids >= 0).unsqueeze(1).unsqueeze(-1).expand_as(value)
    assert torch.equal(value.grad.bool(), expected_grad)


def test_varlen_attention_gathers_gqa_and_scatters_without_cpu_kernel(monkeypatch):
    document_ids = torch.tensor([[0, 0, -1, 1], [0, -1, 1, 1]])
    metadata = varlen.build_varlen_metadata(document_ids, document_ids >= 0)
    query = torch.randn(2, 4, 4, 3, requires_grad=True)
    key = torch.randn(2, 2, 4, 3, requires_grad=True)
    value = torch.randn(2, 2, 4, 3, requires_grad=True)
    captured = {}

    def fake_varlen(query_thd, key_thd, value_thd, cu_q, cu_k, max_q, max_k, **kwargs):
        captured.update(
            query_shape=query_thd.shape,
            key_shape=key_thd.shape,
            value_shape=value_thd.shape,
            cu_q=cu_q.clone(),
            cu_k=cu_k.clone(),
            max_q=max_q,
            max_k=max_k,
            kwargs=kwargs,
        )
        return query_thd * 2

    monkeypatch.setattr(varlen, "varlen_attn", fake_varlen)
    output = varlen.varlen_attention(query, key, value, metadata)
    output.sum().backward()

    assert captured["query_shape"] == (6, 4, 3)
    assert captured["key_shape"] == captured["value_shape"] == (6, 2, 3)
    assert captured["cu_q"].tolist() == captured["cu_k"].tolist() == [0, 2, 3, 4, 6]
    assert captured["max_q"] == captured["max_k"] == 4
    assert captured["kwargs"]["enable_gqa"] is True
    assert output.shape == (2, 4, 4, 3)
    assert torch.equal(output[0, 2], torch.zeros_like(output[0, 2]))
    expected_query_grad = (
        (document_ids >= 0).unsqueeze(1).unsqueeze(-1).expand_as(query)
    )
    assert torch.equal(query.grad.bool(), expected_query_grad)


def test_varlen_attention_rejects_incompatible_shapes_before_kernel(monkeypatch):
    metadata = varlen.build_varlen_metadata(
        torch.tensor([[0, 0]]), torch.ones(1, 2, dtype=torch.bool)
    )
    monkeypatch.setattr(
        varlen, "varlen_attn", lambda *args, **kwargs: pytest.fail("kernel")
    )
    with pytest.raises(ValueError, match="key and value"):
        varlen.varlen_attention(
            torch.ones(1, 2, 2, 4),
            torch.ones(1, 1, 2, 4),
            torch.ones(1, 1, 2, 3),
            metadata,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_varlen_attention_moves_cpu_metadata_to_attention_device(monkeypatch):
    document_ids = torch.tensor([[0, 0, -1, 1]])
    metadata = varlen.build_varlen_metadata(document_ids, document_ids >= 0)
    device = torch.device("cuda", torch.cuda.current_device())
    query = torch.randn(1, 1, 4, 2, device=device, requires_grad=True)
    key = torch.randn(1, 1, 4, 2, device=device, requires_grad=True)
    value = torch.randn(1, 1, 4, 2, device=device, requires_grad=True)

    def fake_varlen(query_thd, key_thd, value_thd, cu_q, cu_k, *args, **kwargs):
        assert query_thd.device == cu_q.device == cu_k.device == device
        return query_thd + key_thd + value_thd

    monkeypatch.setattr(varlen, "varlen_attn", fake_varlen)
    output = varlen.varlen_attention(query, key, value, metadata)
    output.sum().backward()
    actual_grads = [tensor.grad.clone() for tensor in (query, key, value)]

    for tensor in (query, key, value):
        tensor.grad = None
    reference = varlen.varlen_attention(query, key, value, metadata.to(device))
    reference.sum().backward()

    torch.testing.assert_close(output, reference)
    for actual, tensor in zip(actual_grads, (query, key, value), strict=True):
        torch.testing.assert_close(actual, tensor.grad)
