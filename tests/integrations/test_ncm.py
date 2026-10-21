"""CPU tests for the Next-Concept Modeling plugin."""

import pytest
import torch

from axolotl.integrations.ncm.configuration import (
    NCMGeneratorConfig,
    NCMTokenizerConfig,
)
from axolotl.integrations.ncm.loading import NCMAutoModel
from axolotl.integrations.ncm.modeling import NCMGeneratorModel, NCMTokenizerModel
from axolotl.integrations.ncm.plugin import NCMPlugin

LEVELS = [1, 2, 4, 8]
VOCAB = 50


def _tokenizer_config(**kwargs):
    return NCMTokenizerConfig(
        vocab_size=VOCAB,
        levels=LEVELS,
        codebook_size=16,
        dim=16,
        num_heads=2,
        dropout=0.0,
        pad_token_id=0,
        **kwargs,
    )


def _generator(**kwargs):
    torch.manual_seed(0)
    config = NCMGeneratorConfig(
        tokenizer_config=_tokenizer_config(),
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        null_prefix_token_id=1,
        **kwargs,
    )
    return NCMGeneratorModel(config).eval()


def test_quantizer_forward_matches_encode_decode():
    model = NCMTokenizerModel(_tokenizer_config(fine_drop_prob=0.0)).eval()
    x = torch.randint(0, VOCAB, (3, LEVELS[-1]))
    z = model.encode_latents(x)
    z_hat, z_trunc, _ = model.vq(z)
    codes = model.vq.encode(z)
    assert z_trunc is None
    assert codes.shape == (3, sum(LEVELS))
    torch.testing.assert_close(z_hat, model.vq.decode(codes))


@pytest.mark.parametrize("query_level0", [False, True])
def test_greedy_generation_matches_teacher_forcing(query_level0):
    """The parallel training pass and the level-by-level sampler see the same inputs."""
    model = _generator(query_level0=query_level0, head_type="linear")
    level0 = None if query_level0 else torch.randint(0, 16, (3, LEVELS[0]))
    codes = model.generate_codes(batch_size=3, level0_codes=level0, temperature=0)
    logits = model(input_ids=None, codes=codes).logits
    torch.testing.assert_close(
        logits.argmax(-1), codes[:, sum(LEVELS[: model.first]) :]
    )


def test_levels_only_see_coarser_levels():
    model = _generator()
    codes = model.vqvae.encode(torch.randint(0, VOCAB, (2, LEVELS[-1])))
    j = 1
    edited = codes.clone()
    start = sum(LEVELS[:j])
    edited[:, start : start + LEVELS[j]] = (
        edited[:, start : start + LEVELS[j]] + 1
    ) % 16
    a = model(input_ids=None, codes=codes).logits
    b = model(input_ids=None, codes=edited).logits
    # Predictions for levels 1..j condition only on levels < j and must not move.
    unchanged = sum(LEVELS[1 : j + 1])
    torch.testing.assert_close(a[:, :unchanged], b[:, :unchanged])
    assert not torch.allclose(a[:, unchanged:], b[:, unchanged:])


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_level_weights(alpha):
    w = _generator(level_loss_alpha=alpha).level_weights("cpu")
    if alpha == 0:  # every code weighs the same: the plain mean
        torch.testing.assert_close(w, torch.full_like(w, 1 / w.numel()))
    else:  # every level weighs the same
        per_level = torch.stack([c.sum() for c in w.split(LEVELS[1:])])
        torch.testing.assert_close(
            per_level, torch.full_like(per_level, 1 / len(per_level))
        )


def test_corruption_spares_level0_and_finest_level():
    model = _generator(corruption_rate=0.5)
    codes = torch.randint(0, 16, (256, sum(LEVELS)))
    out = model.vq.split_codes(model._corrupt(codes))
    ref = model.vq.split_codes(codes)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[-1], ref[-1])
    rates = [
        (o != r).float().mean().item()
        for o, r in zip(out[1:-1], ref[1:-1], strict=True)
    ]
    assert rates[0] > rates[-1] > 0


def test_tokenizer_trains_and_codebooks_stay_frozen_to_sgd():
    torch.manual_seed(0)
    model = NCMTokenizerModel(_tokenizer_config()).train()
    x = torch.randint(0, VOCAB, (8, LEVELS[-1]))
    before = model.vq.embeddings[0].weight.clone()
    out = model(x, labels=x)
    out.loss.backward()
    assert all(e.weight.grad is None for e in model.vq.embeddings)
    assert model.encoder.wte.weight.grad is not None
    assert not torch.equal(before, model.vq.embeddings[0].weight)  # EMA update


def test_generator_starts_from_stage1_checkpoint(tmp_path):
    tokenizer = NCMTokenizerModel(_tokenizer_config())
    tokenizer.save_pretrained(tmp_path / "tok")
    config = NCMGeneratorConfig.from_pretrained(tmp_path / "tok")
    config.n_layer, config.n_head, config.n_embd = 1, 2, 32
    config.null_prefix_token_id = 1  # bound from the text tokenizer inside axolotl
    model = NCMAutoModel.from_pretrained(str(tmp_path / "tok"), config=config)
    assert isinstance(model, NCMGeneratorModel)
    for k, v in tokenizer.state_dict().items():
        torch.testing.assert_close(model.vqvae.state_dict()[k], v)
    assert not any(p.requires_grad for p in model.vqvae.parameters())

    model.save_pretrained(tmp_path / "gen")
    reloaded = NCMAutoModel.from_pretrained(
        str(tmp_path / "gen"),
        config=NCMGeneratorConfig.from_pretrained(tmp_path / "gen"),
    )
    # Outputs, not just state dicts: tied heads read the codebook by reference.
    codes = model.vqvae.encode(torch.randint(0, VOCAB, (2, LEVELS[-1])))
    torch.testing.assert_close(
        reloaded.eval()(None, codes=codes).logits,
        model.eval()(None, codes=codes).logits,
    )


def test_plugin_routes_stage_configs():
    cfg = {
        "ncm": {"stage": "tokenizer"},
        "tokenizer_config": "/nonexistent/text-tokenizer",
    }
    NCMPlugin().register(cfg)
    assert cfg["base_model"] == "/nonexistent/text-tokenizer"
    assert cfg["reinit_weights"] is True
    assert cfg["cls_model_config"].endswith("NCMScratchTokenizerConfig")

    cfg = {"ncm": {"stage": "generator"}, "base_model": "/nonexistent/stage1"}
    NCMPlugin().register(cfg)
    assert cfg["cls_model_config"].endswith("NCMGeneratorConfig")
