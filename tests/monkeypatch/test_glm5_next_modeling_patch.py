"""Tests for the GLM-5.3-Flash (glm5_next) sample-packing monkeypatch."""

import pytest
import torch

glm5_next = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")

from transformers.models.glm5_next.configuration_glm5_next import (  # noqa: E402
    Glm5NextTextConfig,
)


def _torch_reference(dispatcher):
    """Unwrap a kernel dispatcher to its torch fallback, with or without `kernels` installed."""
    forward = getattr(type(dispatcher), "forward", None)
    if forward is not None:
        (cell,) = forward.__closure__
        dispatcher = cell.cell_contents
    return dispatcher.__wrapped__


@pytest.fixture(autouse=True)
def _torch_kda(monkeypatch):
    """fla's kernels need a GPU, but transformers dispatches to them whenever fla is installed."""
    for name in ("chunk_kimi_delta_attention", "causal_conv1d_fn"):
        monkeypatch.setattr(glm5_next, name, _torch_reference(getattr(glm5_next, name)))


ROW_DOC_LENS = [[13, 21, 9], [20, 23]]


def _config():
    config = Glm5NextTextConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_nope_head_dim=8,
        v_head_dim=8,
        index_topk=8,
        index_head_dim=8,
        index_n_heads=2,
        index_kpool=4,
        linear_head_dim=8,
        linear_num_heads=4,
        hc_mult=2,
        layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    return config


def _text_model(seed=0):
    torch.manual_seed(seed)
    model = glm5_next.Glm5NextTextModel(_config()).eval()
    for param in model.parameters():
        if param.dim() >= 2:
            torch.nn.init.normal_(param, std=0.1)
    return model


def _varlen_stubs():
    """Reference varlen kernels standing in for fla: upstream torch ops, per segment."""

    def segments(cu_seqlens):
        return zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True)

    def causal_conv1d(x, weight, bias=None, activation=None, cu_seqlens=None):
        outputs = [
            glm5_next.causal_conv1d_fn(
                x[:, start:end].transpose(1, 2), weight, bias, activation=activation
            ).transpose(1, 2)
            for start, end in segments(cu_seqlens)
        ]
        return torch.cat(outputs, dim=1), None

    def chunk_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=False, cu_seqlens=None):
        outputs = [
            glm5_next.chunk_kimi_delta_attention(
                q[:, start:end],
                k[:, start:end],
                v[:, start:end],
                g=g[:, start:end],
                beta=beta[:, start:end],
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )[0]
            for start, end in segments(cu_seqlens)
        ]
        return torch.cat(outputs, dim=1), None

    return causal_conv1d, chunk_kda


def _packed_inputs(seed=0):
    torch.manual_seed(seed)
    docs = [[torch.randint(1, 250, (1, n)) for n in lens] for lens in ROW_DOC_LENS]
    input_ids = torch.cat([torch.cat(row, dim=1) for row in docs])
    position_ids = torch.stack(
        [torch.cat([torch.arange(n) for n in lens]) for lens in ROW_DOC_LENS]
    )
    return docs, input_ids, position_ids


@pytest.fixture(name="packing_patch")
def fixture_packing_patch(monkeypatch):
    from axolotl.monkeypatch.models.glm5_next import modeling as patch_module

    monkeypatch.setattr(patch_module, "_load_fla", _varlen_stubs)
    unpatch = patch_module.patch_glm5_next_modeling_packing()
    yield patch_module
    unpatch()


class TestGlm5NextPackingPatch:
    def test_patch_and_unpatch(self):
        from axolotl.monkeypatch.models.glm5_next.modeling import (
            patch_glm5_next_modeling_packing,
        )

        classes = (
            glm5_next.Glm5NextTextModel,
            glm5_next.Glm5NextTextLinearAttention,
            glm5_next.Glm5NextTextAttention,
            glm5_next.Glm5NextTextIndexer,
        )
        originals = tuple(cls.forward for cls in classes)

        unpatch = patch_glm5_next_modeling_packing()
        assert all(
            cls.forward != orig for cls, orig in zip(classes, originals, strict=True)
        )
        assert patch_glm5_next_modeling_packing() is None, "patch should be idempotent"

        unpatch()
        assert tuple(cls.forward for cls in classes) == originals

    def test_unpacked_forward_is_unchanged(self, monkeypatch):
        model = _text_model()
        input_ids = torch.randint(1, 250, (2, 24))
        position_ids = torch.arange(24).expand(2, -1)

        with torch.no_grad():
            stock = model(
                input_ids=input_ids, position_ids=position_ids
            ).last_hidden_state

        from axolotl.monkeypatch.models.glm5_next import modeling as patch_module

        monkeypatch.setattr(patch_module, "_load_fla", _varlen_stubs)
        unpatch = patch_module.patch_glm5_next_modeling_packing()
        try:
            with torch.no_grad():
                patched = model(
                    input_ids=input_ids, position_ids=position_ids
                ).last_hidden_state
        finally:
            unpatch()

        assert torch.equal(patched, stock)

    def test_packed_documents_are_isolated(self, packing_patch, monkeypatch):
        """Every packed document reproduces its standalone forward, KDA and DSA alike."""
        # relu leaves tied zero scores whose top-k order depends on the candidate count
        monkeypatch.setattr(torch.nn.functional, "relu", torch.nn.functional.softplus)
        model = _text_model()
        docs, input_ids, position_ids = _packed_inputs()

        with torch.no_grad():
            packed = model(
                input_ids=input_ids, position_ids=position_ids
            ).last_hidden_state
            solo = torch.cat(
                [
                    torch.cat(
                        [
                            model(
                                input_ids=doc,
                                position_ids=torch.arange(doc.shape[1]).view(1, -1),
                            ).last_hidden_state
                            for doc in row
                        ],
                        dim=1,
                    )
                    for row in docs
                ]
            )

        assert torch.allclose(packed, solo, atol=1e-5)

    def test_packed_forward_differs_without_patch(self):
        """Guards the parity test: the stock forward does leak across documents."""
        model = _text_model()
        docs, input_ids, position_ids = _packed_inputs()

        with torch.no_grad():
            packed = model(
                input_ids=input_ids, position_ids=position_ids
            ).last_hidden_state
            first_row_solo = torch.cat(
                [model(input_ids=doc).last_hidden_state for doc in docs[0]], dim=1
            )

        assert not torch.allclose(packed[:1], first_row_solo, atol=1e-5)

    def test_packing_without_fla_raises(self, monkeypatch):
        from axolotl.monkeypatch.models.glm5_next import modeling as patch_module

        monkeypatch.setattr(patch_module, "_load_fla", lambda: (None, None))
        unpatch = patch_module.patch_glm5_next_modeling_packing()
        try:
            _, input_ids, position_ids = _packed_inputs()
            with pytest.raises(RuntimeError, match="flash-linear-attention"):
                _text_model()(input_ids=input_ids, position_ids=position_ids)
        finally:
            unpatch()
