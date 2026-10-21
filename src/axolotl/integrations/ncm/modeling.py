"""Next-Concept Modeling (Kumbong et al., 2026, arXiv:2610.14162).

Stage 1, :class:`NCMTokenizerModel`: a multiscale residual VQ-VAE that encodes a
fixed-length block of text tokens into a coarse-to-fine hierarchy of codes.

Stage 2, :class:`NCMGeneratorModel`: a transformer over the frozen tokenizer that
predicts every code of level k in parallel from the cumulative reconstruction of
levels < k, so generation takes one forward pass per level.

Ported from HazyResearch/ncm (MIT). Module names follow the reference so its
released checkpoints convert with a small key map (see ``convert.py``).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration import NCMGeneratorConfig, NCMTokenizerConfig
from .layers import Block, Segments, rope_cos_sin
from .quantizer import MultiscaleResidualQuantizer


@dataclass
class NCMTokenizerOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    recon_loss: torch.Tensor | None = None
    vq_loss: torch.Tensor | None = None


@dataclass
class NCMGeneratorOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    codes: torch.Tensor | None = None
    level_losses: torch.Tensor | None = None


class NCMPreTrainedModel(PreTrainedModel):
    base_model_prefix = "ncm"
    # Strict: transformers applies the non-strict list only to fp16, and EMA codebooks must stay fp32.
    _keep_in_fp32_modules_strict = ["vq"]
    _supports_sdpa = True
    # Losses are already means; stops Trainer from skipping the grad-accumulation divide.
    accepts_loss_kwargs = False

    def _init_weights(self, module):
        # Modules initialize themselves (identity-initialized samplers, EMA codebooks).
        pass

    def _freeze(self):
        """Mark the parameters that must not receive gradients."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        want_info = kwargs.pop("output_loading_info", False)
        model, info = super().from_pretrained(*args, output_loading_info=True, **kwargs)
        if info["missing_keys"]:
            raise ValueError(
                f"{cls.__name__}: checkpoint lacks {sorted(info['missing_keys'])[:8]}. "
                "NCM does not initialize missing weights, so model_config overrides "
                "cannot add parameters to an existing checkpoint."
            )
        # Loading rebuilds parameters as trainable, dropping the freeze set at construction.
        model._freeze()
        return (model, info) if want_info else model


LinearParams = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]


def _fit_length(
    input_ids: torch.Tensor, labels: torch.Tensor | None, length: int, pad_id: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Crop collator padding (or pad) to the tokenizer's fixed block length."""
    T = input_ids.shape[1]
    if T > length:
        if labels is not None and (labels[:, length:] != -100).any():
            raise ValueError(
                f"NCM blocks hold {length} tokens but the batch has {T} unmasked; "
                f"set sequence_len to {length}."
            )
        input_ids = input_ids[:, :length]
        labels = labels[:, :length] if labels is not None else None
    elif T < length:
        input_ids = F.pad(input_ids, (0, length - T), value=pad_id)
        if labels is not None:
            labels = F.pad(labels, (0, length - T), value=-100)
    return input_ids, labels


class NCMEncoder(nn.Module):
    def __init__(self, config: NCMTokenizerConfig):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.dim)
        self.wpe = (
            nn.Embedding(config.context_length, config.dim)
            if config.encoder_positional
            else None
        )
        self.drop = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.wte(input_ids)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(input_ids.shape[1], device=input_ids.device))
        return self.drop(x)


class NCMDecoder(nn.Module):
    def __init__(self, config: NCMTokenizerConfig):
        super().__init__()
        self.wpe = (
            nn.Embedding(config.context_length, config.dim)
            if config.decoder_positional
            else None
        )
        self.blocks = nn.ModuleList(
            Block(config.dim, config.num_heads, bias=True, dropout=config.dropout)
            for _ in range(config.decoder_layers)
        )
        self.ln_f = nn.LayerNorm(config.dim, eps=1e-6, bias=False)
        self.output_layer = nn.Linear(config.dim, config.vocab_size, bias=False)

    def hidden(self, z_hat: torch.Tensor) -> torch.Tensor:
        x = z_hat.to(self.output_layer.weight.dtype)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(x.shape[1], device=x.device))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.hidden(z_hat))


class NCMTokenizerModel(NCMPreTrainedModel):
    config_class = NCMTokenizerConfig

    def __init__(self, config: NCMTokenizerConfig):
        config.validate()
        super().__init__(config)
        self.encoder = NCMEncoder(config)
        if config.pre_quant_norm == "group":
            self.pre_quant_norm = nn.GroupNorm(
                config.pre_quant_groups, config.dim, affine=False
            )
        elif config.pre_quant_norm == "layer":
            self.pre_quant_norm = nn.LayerNorm(config.dim, elementwise_affine=False)
        else:
            self.pre_quant_norm = None
        self.pre_quant_drop = nn.Dropout(config.pre_quant_dropout)
        self.vq = MultiscaleResidualQuantizer(
            levels=config.levels,
            codebook_sizes=config.codebook_sizes,
            dim=config.dim,
            commitment_cost=config.commitment_cost,
            decay=config.ema_decay,
            epsilon=config.ema_epsilon,
            quant_resi_ratio=config.quant_resi_ratio,
            l0_cascade_base=config.l0_cascade_base,
            l0_cascade_norm=config.l0_cascade_norm,
            l0_cascade_norm_final=config.l0_cascade_norm_final,
            fine_drop_prob=config.fine_drop_prob,
            fine_drop_min_keep=config.fine_drop_min_keep,
            fine_drop_dual=config.fine_drop_dual,
            dead_code_threshold=config.dead_code_threshold,
        )
        self.decoder = NCMDecoder(config)
        self.post_init()

    def _freeze(self):
        self.vq.embeddings.requires_grad_(False)

    def get_input_embeddings(self):
        return self.encoder.wte

    def set_input_embeddings(self, value):
        self.encoder.wte = value

    def get_output_embeddings(self):
        return self.decoder.output_layer

    def set_output_embeddings(self, value):
        self.decoder.output_layer = value

    def encode_latents(self, input_ids: torch.Tensor) -> torch.Tensor:
        z = self.encoder(input_ids)
        if isinstance(self.pre_quant_norm, nn.GroupNorm):
            z = self.pre_quant_norm(z.transpose(1, 2)).transpose(1, 2)
        elif self.pre_quant_norm is not None:
            z = self.pre_quant_norm(z)
        return self.pre_quant_drop(z)

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token ids ``(B, L)`` -> codes ``(B, sum(levels))``, coarse to fine."""
        with torch.autocast(input_ids.device.type, enabled=False):
            return self.vq.encode(self.encode_latents(input_ids))

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Codes -> greedy token ids ``(B, L)``."""
        return self.decoder(self.vq.decode(codes)).argmax(-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> NCMTokenizerOutput:
        pad_id = self.config.pad_token_id or 0
        input_ids, labels = _fit_length(
            input_ids, labels, self.config.context_length, pad_id
        )
        with torch.autocast(input_ids.device.type, enabled=False):
            z_hat, z_hat_trunc, vq_loss = self.vq(self.encode_latents(input_ids))
        logits = recon_loss = loss = None
        if labels is None:
            logits = self.decoder(z_hat)
        else:
            recon_loss, logits = self._recon_loss(z_hat, labels)
            if z_hat_trunc is not None:
                w_full, w_trunc = self.config.fine_drop_weights
                trunc, _ = self._recon_loss(z_hat_trunc, labels)
                recon_loss = w_full * recon_loss + w_trunc * trunc
            loss = recon_loss + vq_loss
        return NCMTokenizerOutput(
            loss=loss, logits=logits, recon_loss=recon_loss, vq_loss=vq_loss
        )

    def _recon_loss(self, z_hat: torch.Tensor, labels: torch.Tensor):
        """Token CE of the decoded reconstruction and the logits; the Cut
        Cross-Entropy patch replaces this to skip materializing logits."""
        return self._dense_recon_loss(z_hat, labels)

    def _dense_recon_loss(self, z_hat: torch.Tensor, labels: torch.Tensor):
        logits = self.decoder(z_hat)
        return F.cross_entropy(logits.float().flatten(0, 1), labels.flatten()), logits


class TiedCodebookHead(nn.Module):
    """Logits from negative squared distance to the frozen level codebook, the
    inverse of the tokenizer's nearest-neighbour assignment."""

    def __init__(
        self, n_embd: int, codebook: nn.Embedding, learn_scale: bool, bias: bool
    ):
        super().__init__()
        num_codes, dim = codebook.weight.shape
        self.proj = nn.Linear(n_embd, dim, bias=False)
        self.log_inv_T = nn.Parameter(torch.zeros(())) if learn_scale else None
        self.bias = nn.Parameter(torch.zeros(num_codes)) if bias else None
        # Tuple keeps the tokenizer's codebook out of this module's state dict.
        self._codebook = (codebook,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E = self._codebook[0].weight.float()
        E_sq = E.pow(2).sum(-1)
        z = self.proj(x)
        out = (2.0 * (z @ E.t().to(z.dtype)) - E_sq.to(z.dtype)) / E_sq.mean().clamp(
            min=1e-12
        )
        if self.log_inv_T is not None:
            out = out * self.log_inv_T.exp()
        return out if self.bias is None else out + self.bias

    def linear(self, x: torch.Tensor) -> LinearParams:
        """``forward`` as ``(input, weight, bias)`` of one linear layer, for CCE."""
        E = self._codebook[0].weight.float()
        E_sq = E.pow(2).sum(-1)
        scale = 1.0 / E_sq.mean().clamp(min=1e-12)
        if self.log_inv_T is not None:
            scale = scale * self.log_inv_T.exp()
        bias = -scale * E_sq
        return (
            self.proj(x),
            2 * scale * E,
            bias if self.bias is None else bias + self.bias,
        )


class SharedHeads(nn.Module):
    def __init__(self, n_embd: int, num_codes: int, num_levels: int):
        super().__init__()
        self.head = nn.Linear(n_embd, num_codes, bias=False)
        self.biases = nn.ParameterList(
            nn.Parameter(torch.zeros(num_codes)) for _ in range(num_levels)
        )

    def level(self, k: int, x: torch.Tensor) -> torch.Tensor:
        return self.head(x) + self.biases[k]

    def linear(self, k: int, x: torch.Tensor) -> LinearParams:
        return x, self.head.weight, self.biases[k]


class NCMGeneratorModel(NCMPreTrainedModel):
    config_class = NCMGeneratorConfig
    _no_split_modules = ["Block"]

    def __init__(self, config: NCMGeneratorConfig):
        config.validate()
        super().__init__(config)
        tok_cfg = config.tokenizer_config
        self.vqvae = NCMTokenizerModel(tok_cfg)
        self.vqvae.eval()

        levels = tok_cfg.levels
        self.first = config.first_predicted_level
        self.pred_levels = levels[self.first :]
        self.vq_dim = tok_cfg.dim
        n_embd = config.n_embd

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(self.vq_dim, n_embd, bias=config.bias),
                wle=nn.Embedding(len(self.pred_levels), n_embd),
                blocks=nn.ModuleList(
                    Block(n_embd, config.n_head, config.bias, config.dropout)
                    for _ in range(config.n_layer)
                ),
                ln_f=nn.LayerNorm(n_embd, eps=1e-6, bias=config.bias),
            )
        )
        if config.query_level0:
            self.l0_queries = nn.Parameter(
                nn.init.trunc_normal_(
                    torch.zeros(1, levels[0], self.vq_dim), std=n_embd**-0.5
                )
            )
        k = config.post_upsample_conv_kernel
        if k:
            assert k % 2 == 1, "post_upsample_conv_kernel must be odd"
            self.post_upsample_convs = nn.ModuleList(
                nn.Conv1d(self.vq_dim, self.vq_dim, k, padding=k // 2)
                for _ in self.pred_levels
            )
            for conv in self.post_upsample_convs:
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)
        else:
            self.post_upsample_convs = None

        sizes = tok_cfg.codebook_sizes[self.first :]
        if config.head_type == "tied":
            codebooks = self.vqvae.vq.embeddings
            self.lm_heads = nn.ModuleList(
                TiedCodebookHead(
                    n_embd,
                    codebooks[i + self.first],
                    config.tied_head_learn_scale,
                    config.head_bias,
                )
                for i in range(len(sizes))
            )
        elif config.head_type == "shared":
            assert len(set(sizes)) == 1, "shared heads need equal codebook sizes"
            self.lm_heads = SharedHeads(n_embd, sizes[0], len(sizes))
        else:
            self.lm_heads = nn.ModuleList(
                nn.Linear(n_embd, v, bias=config.head_bias) for v in sizes
            )

        # Level-0 code counts for unconditional sampling; int64 so dtype casts leave them exact.
        self.register_buffer(
            "level0_counts", torch.zeros(tok_cfg.codebook_sizes[0], dtype=torch.long)
        )

        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight") and name.startswith("transformer."):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5)
        self._freeze()
        self.post_init()

    def _freeze(self):
        self.vqvae.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vqvae.eval()
        return self

    def get_input_embeddings(self):
        return self.vqvae.encoder.wte

    @property
    def vq(self) -> MultiscaleResidualQuantizer:
        return self.vqvae.vq

    def _head(self, i: int, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.lm_heads, SharedHeads):
            return self.lm_heads.level(i, x)
        return self.lm_heads[i](x)

    def _head_linear(self, i: int, x: torch.Tensor) -> LinearParams:
        if isinstance(self.lm_heads, SharedHeads):
            return self.lm_heads.linear(i, x)
        head = self.lm_heads[i]
        if isinstance(head, TiedCodebookHead):
            return head.linear(x)
        return x, head.weight, head.bias

    def _embed_prefix(
        self, prefix_ids: torch.Tensor | None, batch_size: int, device
    ) -> torch.Tensor:
        P = self.config.prefix_len
        if prefix_ids is None:
            if self.config.null_prefix_token_id is None:
                raise ValueError("unconditional NCM needs null_prefix_token_id")
            prefix_ids = torch.full(
                (batch_size, P),
                self.config.null_prefix_token_id,
                device=device,
                dtype=torch.long,
            )
        if prefix_ids.shape[1] != P:
            raise ValueError(
                f"prefix must hold prefix_len={P} tokens, got {prefix_ids.shape[1]}"
            )
        emb = self.vqvae.encoder.wte(prefix_ids)
        return self.transformer.wte(emb.to(self.transformer.wte.weight.dtype))

    def _level_inputs(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """Project a predicted level's conditioning vectors to the hidden width."""
        x = x.to(self.transformer.wte.weight.dtype)
        if self.post_upsample_convs is not None:
            x = x + self.post_upsample_convs[i](x.transpose(1, 2)).transpose(1, 2)
        x = self.transformer.wte(x)
        return x + self.transformer.wle.weight[i]

    def _backbone(
        self, x: torch.Tensor, sizes: list[int], prefix_ids: torch.Tensor | None
    ) -> torch.Tensor:
        """Hidden states for prefix + level blocks of ``sizes``; the swappable seam."""
        P = self.config.prefix_len
        pad = self.config.prefix_pad_token_id
        keep = None if prefix_ids is None or pad is None else prefix_ids != pad
        rope = rope_cos_sin(
            self._positions(sizes, x.device),
            x.shape[-1] // self.config.n_head,
            self.config.rope_base,
        )
        segments = Segments(P, sizes, keep)
        for block in self.transformer.blocks:
            x = block(x, rope, segments)
        return self.transformer.ln_f(x)

    def _positions(self, sizes: list[int], device) -> torch.Tensor:
        P = self.config.prefix_len
        return torch.cat(
            [torch.arange(P, device=device)]
            + [torch.arange(n, device=device) for n in sizes]
        )

    def _corrupt(self, codes: torch.Tensor) -> torch.Tensor:
        """Replace a per-level fraction of conditioning codes with uniform codes
        from the same codebook. Coarse levels get the highest rate under the
        linear schedule; rates average to ``corruption_rate``."""
        cfg = self.config
        levels = self.vq.levels
        first = 1 if cfg.corruption_skip_level0 else 0
        corruptible = list(range(first, len(levels) - 1))
        n = len(corruptible)
        if not n or cfg.corruption_rate <= 0:
            return codes
        chunks = list(self.vq.split_codes(codes))
        for j, k in enumerate(corruptible):
            rate = cfg.corruption_rate
            if cfg.corruption_schedule == "linear":
                rate = cfg.corruption_rate * 2 * (n - j) / (n + 1)
            rate = min(rate, 1.0)
            c = chunks[k]
            hit = torch.rand(c.shape, device=c.device) < rate
            rand = torch.randint_like(c, self.vq.codebook_sizes[k])
            chunks[k] = torch.where(hit, rand, c)
        return torch.cat(chunks, dim=1)

    def level_weights(self, device) -> torch.Tensor:
        """Per-position loss weights: level k weighs ell_k^-a / sum_j ell_j^(1-a)."""
        a = self.config.level_loss_alpha
        ells = torch.tensor(self.pred_levels, dtype=torch.float32, device=device)
        w = ells.pow(-a) / ells.pow(1 - a).sum()
        return torch.repeat_interleave(w, ells.long())

    def forward(
        self,
        input_ids: torch.Tensor,
        prefix_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        codes: torch.Tensor | None = None,
        **kwargs,
    ) -> NCMGeneratorOutput:
        """Teacher-forced pass over all levels at once.

        ``input_ids`` is the target text block; it is encoded to codes by the
        frozen tokenizer (or pass precomputed ``codes``). ``labels`` only marks
        that a loss is wanted; the targets are the codes themselves.
        """
        if codes is None:
            pad_id = self.vqvae.config.pad_token_id or 0
            input_ids, _ = _fit_length(
                input_ids, labels, self.vqvae.config.context_length, pad_id
            )
            codes = self.vqvae.encode(input_ids)
        B, device = codes.shape[0], codes.device
        if self.training and not self.config.query_level0:
            self.level0_counts += torch.bincount(
                codes[:, : self.vq.levels[0]].flatten(),
                minlength=self.level0_counts.numel(),
            )

        cond = self._corrupt(codes) if self.training else codes
        with torch.no_grad(), torch.autocast(device.type, enabled=False):
            z_in = self.vq.next_level_inputs(cond)
        if self.config.query_level0:
            z_in = torch.cat(
                [self.l0_queries.expand(B, -1, -1).to(z_in.dtype), z_in], dim=1
            )
        chunks = z_in.split(self.pred_levels, dim=1)
        x = torch.cat([self._level_inputs(c, i) for i, c in enumerate(chunks)], dim=1)

        prefix = self._embed_prefix(prefix_ids, B, device)
        x = torch.cat([prefix.to(x.dtype), x], dim=1)
        h = self._backbone(x, self.pred_levels, prefix_ids)
        h = h[:, self.config.prefix_len :]

        targets = codes[:, sum(self.vq.levels[: self.first]) :]
        ce, logits = self._level_cross_entropy(h, targets)
        loss = (ce * self.level_weights(device)).sum(1).mean()
        level_losses = torch.stack(
            [c.mean() for c in ce.detach().split(self.pred_levels, dim=1)]
        )
        return NCMGeneratorOutput(
            loss=loss, logits=logits, codes=codes, level_losses=level_losses
        )

    def _level_cross_entropy(self, h: torch.Tensor, targets: torch.Tensor):
        """Per-code CE ``(B, codes)`` and the logits; the Cut Cross-Entropy patch
        (``cut_cross_entropy.py``) replaces this to skip materializing logits."""
        return self._dense_level_cross_entropy(h, targets)

    def _dense_level_cross_entropy(self, h: torch.Tensor, targets: torch.Tensor):
        h_levels = h.split(self.pred_levels, dim=1)
        per_level = [self._head(i, hk) for i, hk in enumerate(h_levels)]
        if len({lg.shape[-1] for lg in per_level}) == 1:
            logits = torch.cat(per_level, dim=1)
            ce = F.cross_entropy(
                logits.float().transpose(1, 2), targets, reduction="none"
            )
            return ce, logits
        t_levels = targets.split(self.pred_levels, dim=1)
        ce = torch.cat(
            [
                F.cross_entropy(lg.float().transpose(1, 2), tg, reduction="none")
                for lg, tg in zip(per_level, t_levels, strict=True)
            ],
            dim=1,
        )
        return ce, None

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        generator=None,
    ):
        if temperature == 0:
            return logits.argmax(-1)
        flat = logits.reshape(-1, logits.shape[-1]).float() / temperature
        if top_k > 0:
            kth = torch.topk(flat, top_k, dim=-1).values[:, -1:]
            flat = flat.masked_fill(flat < kth, float("-inf"))
        if top_p < 1.0:
            sorted_logits, sorted_idx = flat.sort(dim=-1, descending=True)
            probs = sorted_logits.softmax(-1)
            remove = probs.cumsum(-1) - probs > top_p
            flat = flat.scatter(
                -1, sorted_idx, sorted_logits.masked_fill(remove, float("-inf"))
            )
        idx = torch.multinomial(flat.softmax(-1), 1, generator=generator)
        return idx.view(logits.shape[:-1])

    @torch.no_grad()
    def generate_codes(
        self,
        batch_size: int | None = None,
        prefix_ids: torch.Tensor | None = None,
        level0_codes: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample codes coarse to fine, one forward pass per predicted level."""
        device = self.level0_counts.device
        B = prefix_ids.shape[0] if prefix_ids is not None else batch_size
        if B is None:
            raise ValueError("pass batch_size or prefix_ids")
        levels, P = self.vq.levels, self.config.prefix_len
        prefix = self._embed_prefix(prefix_ids, B, device)

        def run_level(i: int, x_vq: torch.Tensor) -> torch.Tensor:
            n = self.pred_levels[i]
            x = torch.cat([prefix, self._level_inputs(x_vq, i).to(prefix.dtype)], dim=1)
            h = self._backbone(x, [n], prefix_ids)[:, P:]
            return self._sample(self._head(i, h), temperature, top_k, top_p, generator)

        if self.config.query_level0:
            out = [run_level(0, self.l0_queries.expand(B, -1, -1))]
        elif level0_codes is not None:
            out = [level0_codes.to(device)]
        else:
            if self.level0_counts.sum() <= 0:
                raise ValueError("no level-0 statistics; pass level0_codes")
            out = [
                torch.multinomial(
                    self.level0_counts.float(),
                    B * levels[0],
                    replacement=True,
                    generator=generator,
                ).view(B, levels[0])
            ]

        z_hat = torch.zeros(B, levels[-1], self.vq_dim, device=device)
        for k in range(1, len(levels)):
            z_hat = self.vq.add_level(z_hat, out[-1], k - 1)
            out.append(run_level(k - self.first, self.vq.resize(z_hat, k)))
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def generate(self, *args, **kwargs) -> torch.Tensor:
        """Sample codes and decode them to text token ids ``(B, L)``."""
        return self.vqvae.decode(self.generate_codes(*args, **kwargs))
