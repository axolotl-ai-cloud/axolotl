"""`TernaryEmbedding` — the module, the swap, and what each export does with it.

Ternarizing embeddings is a heal-at-scale feature, not a size lever: the damage probe
destroyed a 135M model at every scale structure. These tests cover the mechanism, not
the quality question, which only a healing run at 1.7B+ can answer.
"""

import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary import quant
from axolotl.integrations.ternary.args import TernaryConfig
from axolotl.integrations.ternary.export import bake, hf_bitnet
from axolotl.integrations.ternary.modules import (
    TernaryEmbedding,
    TernaryLinear,
    iter_quantized_modules,
    iter_ternary_modules,
)
from axolotl.integrations.ternary.swap import SwapManifest, convert_model
from axolotl.utils.dict import DictDefault

VOCAB = 128
HIDDEN = 64


def _embedding(structure="per_row", group_size=None, dtype=torch.bfloat16, **kwargs):
    torch.manual_seed(0)
    source = nn.Embedding(VOCAB, HIDDEN, dtype=dtype)
    with torch.no_grad():
        source.weight.normal_(0.0, 0.02)
    return TernaryEmbedding.from_embedding(
        source, scale_structure=structure, group_size=group_size, **kwargs
    )


def _model(tied=False, vocab=VOCAB, hidden=HIDDEN):
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        tie_word_embeddings=tied,
    )
    return LlamaForCausalLM(config).to(torch.bfloat16)


def _cfg(**ternary):
    base = {"embedding_dtype": "ternary", "strict_enumeration": False}
    return DictDefault({"ternary": {**base, **ternary}})


# ------------------------------------------------------------------ the oracle


@pytest.mark.parametrize(
    "structure,group_size",
    [("per_row", None), ("per_tensor", None), ("grouped", 32)],
)
def test_the_forward_gathers_from_the_quantized_weight(structure, group_size):
    """The module is a gather over the same fake-quant the oracle produces."""
    module = _embedding(structure, group_size)
    tokens = torch.tensor([0, 5, 127])

    got = module(tokens)

    expected_weight = quant.fake_quant_weight(
        module.weight.detach(), 1.0, module._scale_group_size(), None
    )
    assert torch.equal(got, expected_weight[tokens])


@pytest.mark.parametrize("structure", ["per_row", "per_tensor"])
def test_a_zero_lambda_is_the_untouched_embedding(structure):
    module = _embedding(structure)
    module.set_lambda(0.0)
    tokens = torch.tensor([1, 2, 3])

    assert torch.equal(module(tokens), module.weight.detach()[tokens])


def test_a_partial_lambda_interpolates():
    module = _embedding()
    tokens = torch.tensor([7])
    module.set_lambda(0.0)
    latent = module(tokens).clone()
    module.set_lambda(1.0)
    quantized = module(tokens).clone()
    module.set_lambda(0.5)

    half = module(tokens)

    assert torch.allclose(
        half.float(), (latent.float() + quantized.float()) / 2, atol=1e-2
    )
    assert not torch.equal(half, latent)


def test_per_row_keeps_every_row_alive_where_per_tensor_zeroes_them():
    """The reason `per_row` is the default, made concrete.

    A per-tensor grid is set by the loud rows, so quiet rows fall inside the rounding
    cell and quantize to all-zero — that is pruning those tokens, not quantizing them.
    """
    torch.manual_seed(0)
    source = nn.Embedding(VOCAB, HIDDEN, dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.normal_(0.0, 0.02)
        source.weight[:8] *= 0.01  # rare tokens, small magnitudes

    per_tensor = TernaryEmbedding.from_embedding(source, scale_structure="per_tensor")
    per_row = TernaryEmbedding.from_embedding(source, scale_structure="per_row")
    rows = torch.arange(8)

    assert float(per_tensor(rows).detach().abs().sum()) == 0.0
    assert float(per_row(rows).detach().abs().sum()) > 0.0


# ------------------------------------------------------------------------- STE


def test_the_gradient_reaches_the_latent_rows():
    module = _embedding()
    module.weight.requires_grad_(True)
    tokens = torch.tensor([3, 9])

    module(tokens).sum().backward()

    assert module.weight.grad is not None
    touched = module.weight.grad.abs().sum(dim=1) > 0
    assert bool(touched[tokens].all()), "the gathered rows got no gradient"
    assert int(touched.sum()) == len(tokens), "a gather must not touch other rows"


def test_the_straight_through_gradient_is_not_the_rounding_derivative():
    """Without an STE the gather's gradient would be zero almost everywhere."""
    module = _embedding()
    module.weight.requires_grad_(True)

    module(torch.tensor([4])).sum().backward()

    assert float(module.weight.grad[4].abs().sum()) > 0.0


# -------------------------------------------------------------- config surface


def test_grouped_rejects_a_group_size_that_does_not_divide_the_width():
    """The 576 % 128 case: a width that is not a power of two."""
    source = nn.Embedding(VOCAB, 576, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="does not divide"):
        TernaryEmbedding.from_embedding(
            source, scale_structure="grouped", group_size=128
        )


def test_the_divisor_error_names_usable_group_sizes():
    source = nn.Embedding(VOCAB, 576, dtype=torch.bfloat16)

    with pytest.raises(ValueError) as excinfo:
        TernaryEmbedding.from_embedding(
            source, scale_structure="grouped", group_size=128
        )

    suggested = str(excinfo.value).rsplit(":", 1)[1]
    assert "64" in suggested
    assert all(576 % int(part) == 0 for part in suggested.split(","))


def test_a_group_size_without_grouped_is_rejected():
    with pytest.raises(ValueError, match="only valid with scale_structure"):
        TernaryEmbedding(VOCAB, HIDDEN, scale_structure="per_row", group_size=32)


def test_an_unknown_structure_is_rejected():
    with pytest.raises(ValueError, match="scale_structure must be"):
        TernaryEmbedding(VOCAB, HIDDEN, scale_structure="per_column")


def test_the_config_warns_loudly_on_enable(caplog):
    with caplog.at_level("WARNING"):
        TernaryConfig(embedding_dtype="ternary")

    warning = caplog.text
    assert "1.7B" in warning
    assert "tied" in warning.lower()


def test_the_config_rejects_a_structure_that_cannot_apply():
    with pytest.raises(ValueError, match="no effect while"):
        TernaryConfig(embedding_dtype="bf16", embedding_scale="per_tensor")


def test_the_config_requires_a_group_size_for_grouped():
    with pytest.raises(ValueError, match="requires"):
        TernaryConfig(embedding_dtype="ternary", embedding_scale="grouped")


# -------------------------------------------------------------------- the swap


def test_the_swap_replaces_the_input_embedding():
    model = _model()

    manifest = convert_model(model, _cfg())

    assert isinstance(model.get_input_embeddings(), TernaryEmbedding)
    assert manifest.ternary_embeddings == ["model.embed_tokens"]
    assert model(torch.randint(0, VOCAB, (2, 8))).logits.shape == (2, 8, VOCAB)


def test_an_untied_lm_head_is_left_full_precision():
    """The field guide's keep_fp for the head stands when nothing forces otherwise."""
    model = _model(tied=False)

    manifest = convert_model(model, _cfg())

    assert type(model.lm_head) is nn.Linear
    assert model.lm_head.weight is not model.get_input_embeddings().weight
    assert manifest.tied_embeddings is False
    assert "lm_head" in manifest.kept_fp


def test_a_tied_head_is_recorded_and_warned_about(caplog):
    model = _model(tied=True)

    with caplog.at_level("WARNING"):
        manifest = convert_model(model, _cfg())

    assert manifest.tied_embeddings is True
    assert model.lm_head.weight is model.get_input_embeddings().weight
    assert "tie_word_embeddings" in caplog.text
    assert "same grid" in caplog.text


def test_a_tied_head_quantizes_on_the_embedding_grid():
    """The head must see the same error the bake will introduce, not the latent.

    Left as a plain Linear it would read the shared latent all through training and
    only meet the quantization error at save time, so a healing run would never train
    against what it ships with.
    """
    model = _model(tied=True)
    convert_model(model, _cfg())
    embedding = model.get_input_embeddings()
    head = model.lm_head
    hidden = torch.randn(1, 2, HIDDEN, dtype=torch.bfloat16)

    with torch.no_grad():
        got = head(hidden)

    assert isinstance(head, TernaryLinear)
    assert torch.equal(got, F.linear(hidden, embedding.baked_weight()))
    assert not torch.equal(got, F.linear(hidden, embedding.weight))


@pytest.mark.parametrize(
    "structure,group_size,expected",
    [
        ("per_row", None, ("group", HIDDEN)),
        ("per_tensor", None, ("absmean", None)),
        ("grouped", 32, ("group", 32)),
    ],
)
def test_the_tied_head_grid_matches_the_embedding_bit_for_bit(
    structure, group_size, expected
):
    """One group per row is what `per_row` means to a Linear; the others map too."""
    model = _model(tied=True)
    convert_model(
        model, _cfg(embedding_scale=structure, embedding_group_size=group_size)
    )
    embedding = model.get_input_embeddings()
    head = model.lm_head

    assert (head.weight_scale, head.group_size) == expected
    with torch.no_grad():
        assert torch.equal(embedding.baked_weight(), head.baked_weight())


def test_the_tied_head_shares_the_parameter_rather_than_copying_it():
    model = _model(tied=True)
    convert_model(model, _cfg())

    embedding = model.get_input_embeddings()
    assert model.lm_head.weight is embedding.weight
    shared = [
        name for name, param in model.named_parameters() if param is embedding.weight
    ]
    assert len(shared) == 1, f"the tensor is duplicated across {shared}"


def test_the_tie_survives_a_later_tie_weights_call():
    model = _model(tied=True)
    convert_model(model, _cfg())

    model.tie_weights()

    assert isinstance(model.get_input_embeddings(), TernaryEmbedding)
    assert isinstance(model.lm_head, TernaryLinear)
    assert model.lm_head.weight is model.get_input_embeddings().weight


def test_the_tie_and_the_grid_survive_resize_token_embeddings():
    model = _model(tied=True)
    convert_model(model, _cfg(embedding_scale="grouped", embedding_group_size=32))

    model.resize_token_embeddings(VOCAB + 32)

    embedding = model.get_input_embeddings()
    assert isinstance(embedding, TernaryEmbedding)
    assert (embedding.scale_structure, embedding.group_size) == ("grouped", 32)
    assert model.lm_head.weight is embedding.weight
    assert embedding.weight.shape[0] == VOCAB + 32


def test_the_shared_latent_is_baked_exactly_once():
    """Baking twice through the two views would re-quantize quantized values."""
    model = _model(tied=True)
    convert_model(model, _cfg())
    embedding = model.get_input_embeddings()

    for name, module in iter_quantized_modules(model):
        module._post_training(model, name)
    once = embedding.weight.detach().clone()
    for name, module in iter_quantized_modules(model):
        module._post_training(model, name)

    assert torch.equal(embedding.weight.detach(), once), "the grid drifted on re-bake"
    assert model.lm_head.weight is embedding.weight


@pytest.mark.parametrize(
    "structure,group_size", [("per_row", None), ("per_tensor", None), ("grouped", 32)]
)
def test_the_tied_pair_upholds_the_requantization_invariant(structure, group_size):
    model = _model(tied=True)
    convert_model(
        model, _cfg(embedding_scale=structure, embedding_group_size=group_size)
    )
    embedding = model.get_input_embeddings()

    for name, module in iter_quantized_modules(model):
        module._post_training(model, name)

    baked = embedding.weight.detach()
    codes, scale = quant.baked_codes_and_scale(baked, embedding._scale_group_size())
    assert torch.equal(quant.dequantize_codes(codes, scale, baked.dtype), baked)
    # and the head agrees the tensor is already on its grid, so it never re-bakes
    assert model.lm_head.is_baked()


def test_the_lambda_schedule_moves_both_views_in_lockstep():
    model = _model(tied=True)
    convert_model(model, _cfg())

    for _, module in iter_quantized_modules(model):
        module.set_lambda(0.3)

    embedding = model.get_input_embeddings()
    assert embedding.lambda_ == model.lm_head.lambda_ == 0.3
    covered = {id(module) for _, module in iter_quantized_modules(model)}
    assert id(model.lm_head) in covered, "the head would drift from the embedding"


def test_an_untied_head_is_not_converted():
    """The new behaviour triggers only on tied + ternary embeddings."""
    model = _model(tied=False)

    convert_model(model, _cfg())

    assert type(model.lm_head) is nn.Linear


def test_a_tied_head_is_untouched_without_ternary_embeddings():
    model = _model(tied=True)

    convert_model(model, DictDefault({"ternary": {"strict_enumeration": False}}))

    assert type(model.lm_head) is nn.Linear
    assert type(model.get_input_embeddings()) is nn.Embedding


def test_the_embedding_is_not_swapped_by_default():
    model = _model()

    manifest = convert_model(
        model, DictDefault({"ternary": {"strict_enumeration": False}})
    )

    assert type(model.get_input_embeddings()) is nn.Embedding
    assert manifest.ternary_embeddings == []
    assert manifest.embedding_entries() == []


def test_the_manifest_entry_carries_the_embedding_grid():
    model = _model()

    manifest = convert_model(model, _cfg())
    entry = manifest.embedding_entries()[0]

    assert entry.kind == "embedding"
    assert entry.name == "model.embed_tokens"
    assert entry.out_features == VOCAB
    assert entry.in_features == HIDDEN
    # per-row is exactly one group per row, so no packer needs a new grid name
    assert (entry.weight_scale, entry.group_size) == ("group", HIDDEN)


def test_the_lambda_schedule_reaches_the_embedding():
    model = _model()
    convert_model(model, _cfg())

    for _, module in iter_quantized_modules(model):
        module.set_lambda(0.25)

    assert model.get_input_embeddings().lambda_ == 0.25
    assert any(
        isinstance(module, TernaryEmbedding)
        for _, module in iter_quantized_modules(model)
    )
    assert all(
        not isinstance(module, TernaryEmbedding)
        for _, module in iter_ternary_modules(model)
    )


# -------------------------------------------------------------------- the bake


@pytest.mark.parametrize(
    "structure,group_size",
    [("per_row", None), ("per_tensor", None), ("grouped", 32)],
)
def test_the_bake_upholds_the_exact_requantization_invariant(structure, group_size):
    """A baked embedding must re-quantize to itself, or every packer disagrees."""
    module = _embedding(structure, group_size)

    module._post_training(None, "model.embed_tokens")

    assert module.is_baked()
    baked = module.weight.detach()
    codes, scale = quant.baked_codes_and_scale(baked, module._scale_group_size())
    assert torch.equal(quant.dequantize_codes(codes, scale, baked.dtype), baked)
    assert torch.equal(module.baked_weight(), baked)


def test_baking_is_idempotent():
    module = _embedding()

    module._post_training(None, "e")
    once = module.weight.detach().clone()
    module._post_training(None, "e")

    assert torch.equal(module.weight.detach(), once)


def test_training_the_latent_lapses_the_baked_flag():
    module = _embedding()
    module._post_training(None, "e")

    with torch.no_grad():
        module.weight.add_(0.01)

    assert not module.is_baked()


def test_a_fused_optimizer_step_lapses_the_baked_flag():
    """`torch._fused_adamw_` moves the latent without bumping its version counter."""
    module = _embedding(dtype=torch.float32)
    module.weight.requires_grad_(True)
    module._post_training(None, "e")
    assert module.is_baked()

    optimizer = torch.optim.AdamW([module.weight], lr=0.1, fused=False)
    module.weight.grad = torch.ones_like(module.weight)
    optimizer.step()

    assert not module.is_baked()


def test_the_swapped_embedding_bakes_through_the_state_dict_path():
    model = _model()
    manifest = convert_model(model, _cfg())
    for _, module in iter_quantized_modules(model):
        module._post_training(model, "")

    state = {k: v for k, v in model.state_dict().items()}
    baked = bake.bake_state_dict(state, manifest)

    key = "model.embed_tokens.weight"
    assert torch.equal(baked[key], state[key]), (
        "an already-baked master must be a no-op"
    )


def test_a_manifest_with_an_embedding_round_trips(tmp_path):
    model = _model(tied=True)
    manifest = convert_model(model, _cfg())

    manifest.save(tmp_path)
    reloaded = SwapManifest.load(tmp_path)

    assert reloaded.ternary_embeddings == manifest.ternary_embeddings
    assert reloaded.tied_embeddings is True
    assert [e.kind for e in reloaded.embedding_entries()] == ["embedding"]
    assert len(reloaded.linear_entries()) == len(manifest.linear_entries())
    written = json.loads((tmp_path / "ternary_manifest.json").read_text())
    assert written["tied_embeddings"] is True


def test_an_old_manifest_without_kind_still_loads(tmp_path):
    """`kind` was added after manifests were already being written."""
    model = _model()
    manifest = convert_model(
        model, DictDefault({"ternary": {"strict_enumeration": False}})
    )
    manifest.save(tmp_path)
    path = tmp_path / "ternary_manifest.json"
    data = json.loads(path.read_text())
    for entry in data["entries"]:
        entry.pop("kind")
    data.pop("ternary_embeddings")
    data.pop("tied_embeddings")
    path.write_text(json.dumps(data))

    reloaded = SwapManifest.load(tmp_path)

    assert reloaded.embedding_entries() == []
    assert len(reloaded.linear_entries()) == len(reloaded.entries)


# ------------------------------------------------------------------ the export


def test_hf_bitnet_keeps_the_embedding_out_of_bitlinear():
    """`BitNetForCausalLM` converts Linears only; the ternary rows ride in as bf16."""
    model = _model()
    manifest = convert_model(model, _cfg())

    patterns = hf_bitnet._modules_to_not_convert(manifest)

    assert any("embed_tokens" in pattern for pattern in patterns)
    packable = {e.name for e in manifest.linear_entries()}
    assert "model.embed_tokens" not in packable


def test_the_gguf_packer_never_sees_an_embedding_entry():
    """TQ1_0/TQ2_0 need `hidden % 256`; embeddings stay on the Q8_0 route instead."""
    model = _model()
    manifest = convert_model(model, _cfg())

    packable = {f"{e.name}.weight" for e in manifest.linear_entries()}

    assert "model.embed_tokens.weight" not in packable
    assert manifest.embedding_entries(), "the entry must still exist, just not for TQ"
