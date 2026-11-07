"""Load HazyResearch/ncm reference checkpoints (``config.yaml`` + ``tokenizer.pt`` +
``generator.pt``, as released in ``hazyresearch/ncm-checkpoints``) into the HF
models here. Save the result with ``save_pretrained`` to use it as an axolotl
``base_model``."""

from pathlib import Path

import torch
import yaml

from .configuration import NCMGeneratorConfig, NCMTokenizerConfig
from .modeling import NCMGeneratorModel

_DROPPED_GENERATOR_KEYS = ("lvl_1L", "attn_mask_for_masking", "wp_prefix", "wp_1L")


def _require(cond: bool, what: str):
    if not cond:
        raise NotImplementedError(f"reference option not supported: {what}")


def tokenizer_config_from_reference(
    vq_cfg: dict, pad_token_id: int | None = None
) -> NCMTokenizerConfig:
    qc = vq_cfg["vector_quantizer_config"]
    _require(int(vq_cfg.get("enc_layers", 0)) == 0, "enc_layers > 0")
    _require(int(vq_cfg.get("compression_factor", 1)) == 1, "compression_factor != 1")
    _require(bool(qc.get("learned_l1_sampling", False)), "learned_l1_sampling=false")
    _require(not qc.get("learned_all_sampling", False), "learned_all_sampling")
    _require(qc["levels"].get("use_manual_levels", False), "scheduled levels")
    _require(not isinstance(qc.get("decay"), list), "per-level EMA decay")
    qr = qc.get("quant_resi") or {}
    _require(
        int(qr.get("share_mode", 0)) == 0 and not qr.get("learnable_ratio"),
        "shared/learned quant_resi",
    )
    fd = (qc.get("aux") or {}).get("fine_drop") or {}

    groups = vq_cfg.get("pre_quant_groupnorm")
    norm = None
    if groups:
        norm = (
            "layer"
            if str(vq_cfg.get("pre_quant_norm_type", "group")) == "layer"
            else "group"
        )
    return NCMTokenizerConfig(
        vocab_size=vq_cfg["in_vocab_size"],
        levels=list(qc["levels"]["manual_levels"]),
        codebook_size=list(vq_cfg["vqvae_vocab_size"]),
        dim=vq_cfg["embed_dim"],
        num_heads=vq_cfg["n_layers"],
        decoder_layers=int(vq_cfg.get("dec_layers", 1)),
        dropout=vq_cfg.get("dropout", 0.1),
        encoder_positional=bool(vq_cfg.get("enc_positional", True)),
        decoder_positional=bool(vq_cfg.get("dec_positional", False)),
        pre_quant_norm=norm,
        pre_quant_groups=int(groups or 8),
        pre_quant_dropout=float(vq_cfg.get("pre_quant_dropout") or 0.0),
        commitment_cost=qc.get("commitment_cost", 0.25),
        ema_decay=float(qc.get("decay", 0.99)),
        ema_epsilon=float(qc.get("epsilon", 1e-5)),
        quant_resi_ratio=float(qr.get("ratio", 0.5))
        if qr.get("enabled", True)
        else 0.0,
        l0_cascade_base=int(qc.get("learned_l1_cascade_base", 4))
        if qc.get("learned_l1_cascade")
        else None,
        l0_cascade_norm=bool(qc.get("learned_l1_cascade_norm", False)),
        l0_cascade_norm_final=bool(qc.get("learned_l1_cascade_norm_final", False)),
        fine_drop_prob=float(fd.get("prob", 0.0)),
        fine_drop_min_keep=int(fd.get("min_keep", 1)),
        fine_drop_dual=bool(fd.get("dual", False)),
        fine_drop_weights=(float(fd.get("w_full", 0.7)), float(fd.get("w_trunc", 0.3))),
        pad_token_id=pad_token_id,
    )


def generator_config_from_reference(
    cfg: dict, null_prefix_token_id: int | None = None
) -> NCMGeneratorConfig:
    m = cfg["model"]
    _require(m.get("use_rope", False), "learned positions (use_rope=false)")
    _require(m.get("use_level_encoding", True), "use_level_encoding=false")
    _require(not (m.get("ctx_encoder") or {}).get("enabled", False), "ctx_encoder")
    _require(not m.get("predict_level0", False), "predict_level0")
    _require(not m.get("l0_autoregressive", False), "l0_autoregressive")
    _require(
        m.get("attn_pattern", "block_diagonal") == "block_diagonal",
        "block_causal attention",
    )
    _require(m.get("raw_token_prefix", False), "latent prefix (raw_token_prefix=false)")

    if m.get("shared_output_head"):
        _require(
            m.get("shared_head_per_level_bias", False),
            "shared head without per-level bias",
        )
        head_type, head_bias = "shared", False
    elif m.get("tie_heads_to_codebook"):
        _require(not m.get("tie_heads_trainable", False), "trainable tied heads")
        head_type, head_bias = "tied", bool(m.get("tie_heads_bias", False))
    else:
        head_type, head_bias = "linear", bool(m.get("untied_head_bias", True))

    puc = m.get("post_upsample_conv") or {}
    pad = m.get("prefix_pad_token_id")
    return NCMGeneratorConfig(
        tokenizer_config=tokenizer_config_from_reference(
            cfg["tokenizer"]["vqvae"], pad
        ),
        n_layer=m["n_layer"],
        n_head=m["n_head"],
        n_embd=m["n_embd"],
        dropout=m.get("dropout", 0.1),
        bias=bool(m.get("bias", True)),
        rope_base=float(m.get("rope_base", 10000)),
        prefix_len=m["prefix_len"],
        null_prefix_token_id=null_prefix_token_id,
        prefix_pad_token_id=pad,
        query_level0=bool(m.get("query_level0", False)),
        head_type=head_type,
        head_bias=head_bias,
        tied_head_learn_scale=bool(m.get("tie_heads_learn_scale", True)),
        post_upsample_conv_kernel=int(puc.get("kernel_size", 3))
        if puc.get("enabled")
        else 0,
    )


def tokenizer_state_dict_from_reference(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.startswith("decoder.middle_block."):
            k = "decoder.blocks.0." + k[len("decoder.middle_block.") :]
        elif k.startswith("decoder.extra_blocks."):
            idx, rest = k[len("decoder.extra_blocks.") :].split(".", 1)
            k = f"decoder.blocks.{int(idx) + 1}.{rest}"
        out[k] = v
    return out


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def load_reference_checkpoint(
    ckpt_dir: str | Path, null_prefix_token_id: int | None = None
) -> NCMGeneratorModel:
    """Build an :class:`NCMGeneratorModel` from a released reference checkpoint.

    ``null_prefix_token_id`` is the token the reference feeds as the prefix for
    unconditional generation (its text tokenizer's EOS id).
    """
    ckpt_dir = Path(ckpt_dir)
    cfg = yaml.safe_load((ckpt_dir / "config.yaml").read_text())
    model = NCMGeneratorModel(
        generator_config_from_reference(cfg, null_prefix_token_id)
    )

    tok_sd = tokenizer_state_dict_from_reference(
        _load(ckpt_dir / cfg["checkpoints"]["tokenizer"])["model"]
    )
    gen_raw = _load(ckpt_dir / cfg["checkpoints"]["generator"])
    sd = {f"vqvae.{k}": v for k, v in tok_sd.items()}
    sd.update(
        {k: v for k, v in gen_raw["model"].items() if k not in _DROPPED_GENERATOR_KEYS}
    )
    unigram = (gen_raw.get("inference") or {}).get("level0_unigram")
    # The reference stores probabilities; scale to integer counts for the int64 buffer.
    sd["level0_counts"] = (
        (unigram.double() * 2**40).round().long()
        if unigram is not None
        else torch.zeros_like(model.level0_counts)
    )
    model.load_state_dict(sd, strict=True)
    return model.eval()
