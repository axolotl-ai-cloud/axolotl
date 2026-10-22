"""CPU tests for the Next-Concept Modeling plugin."""

import pytest
import torch

from axolotl.integrations.ncm.configuration import (
    NCMGeneratorConfig,
    NCMTokenizerConfig,
)
from axolotl.integrations.ncm.layers import Segments
from axolotl.integrations.ncm.loading import NCMAutoModel
from axolotl.integrations.ncm.modeling import (
    NCMGeneratorModel,
    NCMTokenizerModel,
    _fit_length,
)
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
    model = NCMGeneratorModel(config).eval()
    # Default c_proj init barely moves the residual stream; scale up so mask/RoPE bugs show.
    with torch.no_grad():
        for name, p in model.transformer.named_parameters():
            if name.endswith("c_proj.weight"):
                p.normal_(std=0.3)
    return model


def _saved_tokenizer(path):
    NCMTokenizerModel(_tokenizer_config()).save_pretrained(path)
    return str(path)


def test_quantizer_forward_matches_encode_decode():
    model = NCMTokenizerModel(_tokenizer_config(fine_drop_prob=0.0)).eval()
    x = torch.randint(0, VOCAB, (3, LEVELS[-1]))
    z = model.encode_latents(x)
    z_hat, z_trunc, _ = model.vq(z)
    codes = model.vq.encode(z)
    assert z_trunc is None
    assert codes.shape == (3, sum(LEVELS))
    torch.testing.assert_close(z_hat, model.vq.decode(codes))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "query_level0": True,
            "head_type": "linear",
            "post_upsample_conv_kernel": 3,
            "corruption_rate": 0.5,  # must stay off outside training
        },
        {"prefix_len": 3, "prefix_pad_token_id": 0, "head_type": "shared"},
    ],
)
def test_generation_logits_match_teacher_forcing(kwargs):
    """The level-by-level sampler and the parallel training pass compute the same logits."""
    model = _generator(**kwargs)
    prefix = None
    if model.config.prefix_len > 1:
        prefix = torch.tensor([[5, 6, 0], [7, 0, 0]])
    level0 = None if model.config.query_level0 else torch.randint(0, 16, (2, 1))
    seen = []
    head = model._head
    model._head = lambda i, x: seen.append(head(i, x)) or seen[-1]
    codes = model.generate_codes(
        batch_size=2, prefix_ids=prefix, level0_codes=level0, temperature=0
    )
    model._head = head
    out = model(None, prefix_ids=prefix, codes=codes)
    torch.testing.assert_close(torch.cat(seen, dim=1), out.logits)


def _strict_cce(e, c, targets, bias=None, reduction="none", **_):
    """torch stand-in that enforces cut_cross_entropy's input/grad contract."""
    assert e.dim() == 2 and targets.dim() == 1 and reduction == "none"
    assert e.dtype in (torch.bfloat16, torch.float16) and c.dtype == e.dtype
    logits = torch.nn.functional.linear(e.float(), c.float(), bias)
    loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    loss.register_hook(
        lambda g: g.view(-1).sum() * 0 + g
    )  # CCE backward views the grad
    return loss


@pytest.mark.parametrize("head_type", ["tied", "linear", "shared", "tokenizer"])
def test_cut_cross_entropy_patch_tracks_dense_loss(head_type, monkeypatch):
    """The CCE patch (kernel swapped for a strict stand-in) tracks the dense loss."""
    pytest.importorskip("cut_cross_entropy")
    from cut_cross_entropy.transformers.patch import PATCH_FNS
    from cut_cross_entropy.transformers.utils import PatchOptions

    from axolotl.integrations.ncm import cut_cross_entropy as ncm_cce, loading

    loading._register_cce_patch(None)
    assert PATCH_FNS["ncm"][1] == PATCH_FNS["ncm_tokenizer"][1] == "patch_ncm"

    torch.manual_seed(0)
    if head_type == "tokenizer":
        model = NCMTokenizerModel(_tokenizer_config(fine_drop_prob=0.0)).eval()
    else:
        model = _generator(head_type=head_type, head_bias=True)
    x = torch.randint(0, VOCAB, (2, LEVELS[-1]))
    if head_type == "tokenizer":
        x[0, -2:] = -100
    ids = x.clamp(min=0)

    def loss_and_grads():
        model.zero_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(ids, labels=x).loss
        loss.backward()
        return loss.detach(), [p.grad for p in model.parameters() if p.grad is not None]

    ref_loss, ref_grads = loss_and_grads()
    monkeypatch.setattr(ncm_cce, "cce", _strict_cce)
    opts = PatchOptions(
        impl="cce",
        reduction="mean",
        filter_eps="auto",
        accum_e_fp32=False,
        accum_c_fp32=False,
        filter_e_grad=True,
        filter_c_grad=True,
        train_only=False,
    )
    ncm_cce.patch_ncm(model, opts)
    loss, grads = loss_and_grads()
    torch.testing.assert_close(loss, ref_loss, rtol=2e-2, atol=2e-2)
    assert len(grads) == len(ref_grads)
    for g, r in zip(grads, ref_grads, strict=True):
        cos = torch.nn.functional.cosine_similarity(g.flatten(), r.flatten(), dim=0)
        assert cos > 0.99 or r.abs().max() < 1e-6


def test_segments_match_dense_prefix_lm_mask():
    torch.manual_seed(0)
    P, sizes = 3, [2, 4]
    q, k, v = (torch.randn(2, 2, P + sum(sizes), 8) for _ in range(3))
    keep = torch.tensor([[True, True, False], [True, False, False]])
    seg = torch.cat(
        [torch.full((P,), -1)] + [torch.full((n,), i) for i, n in enumerate(sizes)]
    )
    allowed = (seg[None, :] == -1) | (seg[None, :] == seg[:, None])
    allowed = (
        allowed[None, None]
        & torch.cat([keep, torch.ones(2, sum(sizes), dtype=torch.bool)], dim=1)[
            :, None, None, :
        ]
    )
    dense = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
    out = Segments(P, sizes, keep).attend(q, k, v, dropout_p=0.0)
    torch.testing.assert_close(out, dense)


def test_level_weights_balance_levels():
    model = _generator(level_loss_alpha=1.0)
    w = model.level_weights("cpu")
    per_level = torch.stack([c.sum() for c in w.split(LEVELS[1:])])
    torch.testing.assert_close(per_level, torch.full_like(per_level, 1 / 3))
    out = model(torch.randint(0, VOCAB, (2, LEVELS[-1])))
    torch.testing.assert_close(out.loss, out.level_losses.mean())


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
    assert rates[0] > 1.5 * rates[-1] > 0  # linear schedule, coarse first


def test_generator_train_step_counts_level0_and_keeps_tokenizer_frozen():
    model = _generator(corruption_rate=0.5).train()
    x = torch.randint(0, VOCAB, (4, LEVELS[-1]))
    out = model(x, labels=x)
    out.loss.backward()
    assert not model.vqvae.training
    assert all(p.grad is None for p in model.vqvae.parameters())
    level0 = out.codes[:, : LEVELS[0]].flatten()
    assert torch.equal(model.level0_counts, torch.bincount(level0, minlength=16))


def test_tokenizer_gradients_and_ema():
    torch.manual_seed(0)
    model = NCMTokenizerModel(_tokenizer_config(fine_drop_prob=0.0)).train()
    x = torch.randint(0, VOCAB, (8, LEVELS[-1]))
    before = model.vq.embeddings[0].weight.clone()
    model(x, labels=x).loss.backward()
    assert all(e.weight.grad is None for e in model.vq.embeddings)
    assert model.encoder.wte.weight.grad is not None
    # Level-0 straight-through estimator trains the learned downsampler.
    assert any(p.grad.abs().sum() > 0 for p in model.vq.learned_down_l1.parameters())
    assert not torch.equal(before, model.vq.embeddings[0].weight)


def test_dead_codes_restart_near_inputs():
    torch.manual_seed(0)
    model = NCMTokenizerModel(_tokenizer_config(fine_drop_prob=0.0)).train()
    vq = model.vq
    with torch.no_grad():
        vq.ema_cluster_size_3.zero_()
        vq.embeddings[3].weight.mul_(1e6)
    model(torch.randint(0, VOCAB, (8, LEVELS[-1])))
    assert vq.embeddings[3].weight.norm(dim=1).max() < 1e3
    assert (vq.ema_cluster_size_3 > 0).all()


@pytest.mark.parametrize("length", [LEVELS[-1] - 3, LEVELS[-1] + 3])
def test_fit_length(length):
    ids = torch.ones(2, length, dtype=torch.long)
    labels = ids.clone()
    labels[:, LEVELS[-1] :] = -100
    out_ids, out_labels = _fit_length(ids, labels, LEVELS[-1], pad_id=0)
    assert out_ids.shape[1] == out_labels.shape[1] == LEVELS[-1]
    assert (out_labels[:, length:] == -100).all()
    with pytest.raises(ValueError):
        _fit_length(ids, torch.ones_like(ids), LEVELS[-1] - 4, pad_id=0)


def test_stage2_loads_fp32_frozen_and_reloads_identically(tmp_path):
    source = _saved_tokenizer(tmp_path / "tok")
    config = NCMGeneratorConfig.from_pretrained(source)
    config.n_layer, config.n_head, config.n_embd = 1, 2, 32
    config.null_prefix_token_id = 1
    model = NCMAutoModel.from_pretrained(source, config=config, dtype=torch.bfloat16)
    reference = NCMTokenizerModel.from_pretrained(source)
    for k, v in reference.state_dict().items():
        torch.testing.assert_close(model.vqvae.state_dict()[k], v)

    model.save_pretrained(tmp_path / "gen")
    gen_config = NCMGeneratorConfig.from_pretrained(tmp_path / "gen")
    # axolotl passes the run dtype; bf16 EMA codebooks would crash the fp32 quantizer.
    reloaded = NCMAutoModel.from_pretrained(
        str(tmp_path / "gen"), config=gen_config, dtype=torch.bfloat16
    )
    assert all(p.dtype == torch.float32 for p in reloaded.parameters())
    assert not any(p.requires_grad for p in reloaded.vqvae.parameters())
    # Outputs, not just state dicts: tied heads read the codebook by reference.
    codes = model.vqvae.encode(torch.randint(0, VOCAB, (2, LEVELS[-1])))
    torch.testing.assert_close(
        reloaded.eval()(None, codes=codes).logits,
        model.eval()(None, codes=codes).logits,
    )

    gen_config.head_bias = True
    with pytest.raises(ValueError, match="lacks"):
        NCMAutoModel.from_pretrained(str(tmp_path / "gen"), config=gen_config)


def test_plugin_routes_stages(tmp_path):
    cfg = {"ncm": {"stage": "tokenizer"}, "tokenizer_config": "/nonexistent/text"}
    NCMPlugin().register(cfg)
    assert cfg["base_model"] == "/nonexistent/text"
    assert cfg["reinit_weights"] is True
    assert cfg["cls_model_config"].endswith("NCMScratchTokenizerConfig")

    stage1 = _saved_tokenizer(tmp_path / "tok")
    cfg = {"ncm": {"stage": "tokenizer"}, "base_model": stage1}
    NCMPlugin().register(cfg)
    assert "reinit_weights" not in cfg and "cls_model_config" not in cfg

    cfg = {"ncm": {"stage": "generator"}, "base_model": stage1}
    NCMPlugin().register(cfg)
    assert cfg["cls_model_config"].endswith("NCMGeneratorConfig")

    for bad in (
        {"ncm": {"stage": "generator"}, "base_model": "/nonexistent/stage1"},
        {
            "ncm": {"stage": "tokenizer"},
            "base_model": "/typo",
            "tokenizer_config": "gpt2",
        },
        {"ncm": {"stage": "generator"}, "base_model": stage1, "sample_packing": True},
    ):
        with pytest.raises(ValueError):
            NCMPlugin().register(bad)
