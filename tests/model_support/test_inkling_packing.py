"""Tests that packed Inkling documents match running each one alone."""

import pytest
import torch

modeling = pytest.importorskip("transformers.models.inkling.modeling_inkling")

from axolotl.model_support import (  # noqa: E402
    ModelHookContext,
    ModelHookPhase,
    get_model_support,
    run_model_support_hooks,
)
from axolotl.model_support.inkling import modeling as inkling_modeling  # noqa: E402
from axolotl.utils.dict import DictDefault  # noqa: E402

DOC_LENGTHS = ((7, 5, 4), (3, 13))


def _model(attn_implementation):
    inkling_modeling.patch_inkling_modeling_packing()
    config = modeling.InklingTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        swa_num_attention_heads=2,
        swa_num_key_value_heads=1,
        swa_head_dim=16,
        sliding_window_size=8,
        d_rel=4,
        rel_extent=16,
        log_scaling_n_floor=2,
        layer_types=["hybrid_sliding", "hybrid"],
        mlp_layer_types=["dense", "dense"],
    )
    config._attn_implementation = attn_implementation
    torch.manual_seed(0)
    return modeling.InklingTextModel(config)


def _packed_batch():
    torch.manual_seed(1)
    docs = [[torch.randint(0, 64, (1, n)) for n in row] for row in DOC_LENGTHS]
    input_ids = torch.cat([torch.cat(row, dim=1) for row in docs])
    position_ids = torch.stack(
        [torch.cat([torch.arange(n) for n in row]) for row in DOC_LENGTHS]
    )
    return docs, input_ids, position_ids


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_packed_documents_match_running_alone(attn_implementation):
    model = _model(attn_implementation)
    model.gradient_checkpointing_enable()
    docs, input_ids, position_ids = _packed_batch()

    packed = model(input_ids, position_ids=position_ids).last_hidden_state
    packed_grads = torch.autograd.grad(packed.square().sum(), model.parameters())
    alone = [[model(doc).last_hidden_state for doc in row] for row in docs]
    alone = torch.cat([torch.cat(row, dim=1) for row in alone])
    alone_grads = torch.autograd.grad(alone.square().sum(), model.parameters())

    torch.testing.assert_close(packed, alone)
    torch.testing.assert_close(packed_grads, alone_grads, atol=1e-4, rtol=1e-3)


def test_positional_position_ids_are_packed():
    model = _model("sdpa").eval()
    _, input_ids, position_ids = _packed_batch()

    with torch.no_grad():
        keyword = model(input_ids, position_ids=position_ids).last_hidden_state
        positional = model(input_ids, None, position_ids).last_hidden_state

    torch.testing.assert_close(positional, keyword)


def test_use_cache_rejects_packed_rows_only():
    model = _model("sdpa").eval()
    _, input_ids, position_ids = _packed_batch()

    with pytest.raises(ValueError, match="use_cache=False"):
        model(input_ids, position_ids=position_ids, use_cache=True)
    single = torch.arange(input_ids.shape[1]).expand_as(input_ids)
    assert model(input_ids, position_ids=single, use_cache=True).past_key_values


def test_packed_forward_compiles():
    model = _model("sdpa").eval()
    _, input_ids, position_ids = _packed_batch()

    with torch.no_grad():
        eager = model(input_ids, position_ids=position_ids).last_hidden_state
        compiled = torch.compile(model, backend="aot_eager", fullgraph=True)(
            input_ids, position_ids=position_ids
        ).last_hidden_state

    torch.testing.assert_close(compiled, eager)


@pytest.mark.parametrize("model_type", ["inkling_text", "inkling_mm_model"])
@pytest.mark.parametrize("packing", ["sample_packing", "batch_flattening", None])
def test_model_support_patches_only_packed_runs(monkeypatch, model_type, packing):
    calls = []
    monkeypatch.setattr(
        inkling_modeling,
        "patch_inkling_modeling_packing",
        lambda: calls.append(model_type),
    )
    cfg = DictDefault({packing: True} if packing else {})

    run_model_support_hooks(
        get_model_support(model_type),
        ModelHookPhase.BEFORE_MODEL_BUILD,
        ModelHookContext(cfg=cfg),
    )

    assert calls == ([model_type] if packing else [])
