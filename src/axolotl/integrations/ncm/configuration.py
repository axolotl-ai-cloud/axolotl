"""Configs for the NCM multiscale tokenizer (stage 1) and generator (stage 2)."""

from itertools import pairwise

from transformers import PretrainedConfig


class NCMTokenizerConfig(PretrainedConfig):
    """Multiscale residual VQ-VAE mapping ``context_length`` text tokens to
    ``len(levels)`` levels of discrete codes (``levels[k]`` codes at level k)."""

    model_type = "ncm_tokenizer"

    def __init__(
        self,
        vocab_size: int = 50304,
        levels: list[int] | None = None,
        codebook_size: int | list[int] = 4096,
        dim: int = 384,
        num_heads: int = 6,
        decoder_layers: int = 1,
        dropout: float = 0.1,
        encoder_positional: bool = True,
        decoder_positional: bool = False,
        pre_quant_norm: str | None = "group",
        pre_quant_groups: int = 8,
        pre_quant_dropout: float = 0.0,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
        quant_resi_ratio: float = 0.5,
        dead_code_threshold: float = 0.05,
        l0_cascade_base: int | None = 4,
        l0_cascade_norm: bool = False,
        l0_cascade_norm_final: bool = False,
        fine_drop_prob: float = 0.5,
        fine_drop_min_keep: int = 1,
        fine_drop_dual: bool = False,
        fine_drop_weights: tuple[float, float] = (0.7, 0.3),
        pad_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.levels = list(levels or [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128])
        self.codebook_size = codebook_size
        self.dim = dim
        self.num_heads = num_heads
        self.decoder_layers = decoder_layers
        self.dropout = dropout
        self.encoder_positional = encoder_positional
        self.decoder_positional = decoder_positional
        self.pre_quant_norm = pre_quant_norm
        self.pre_quant_groups = pre_quant_groups
        self.pre_quant_dropout = pre_quant_dropout
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        self.quant_resi_ratio = quant_resi_ratio
        self.dead_code_threshold = dead_code_threshold
        self.l0_cascade_base = l0_cascade_base
        self.l0_cascade_norm = l0_cascade_norm
        self.l0_cascade_norm_final = l0_cascade_norm_final
        self.fine_drop_prob = fine_drop_prob
        self.fine_drop_min_keep = fine_drop_min_keep
        self.fine_drop_dual = fine_drop_dual
        self.fine_drop_weights = list(fine_drop_weights)
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)

    @property
    def context_length(self) -> int:
        return self.levels[-1]

    @property
    def codebook_sizes(self) -> list[int]:
        if isinstance(self.codebook_size, int):
            return [self.codebook_size] * len(self.levels)
        return list(self.codebook_size)

    def validate(self):
        """Checked at model build, since axolotl applies ``model_config`` overrides
        attribute by attribute after construction."""
        levels = list(self.levels)
        if any(b <= a for a, b in pairwise(levels)):
            raise ValueError(f"levels must be strictly increasing, got {levels}")
        if len(self.codebook_sizes) != len(levels):
            raise ValueError("codebook_size must be an int or one size per level")
        if levels[-1] % levels[0]:
            raise ValueError(
                f"levels[-1]={levels[-1]} must be divisible by levels[0]={levels[0]}"
            )
        if self.dim % self.num_heads:
            raise ValueError(
                f"dim={self.dim} must be divisible by num_heads={self.num_heads}"
            )


class NCMScratchTokenizerConfig(NCMTokenizerConfig):
    """Stage-1 entry point: ignores the source path and starts from defaults, so a
    run's ``model_config`` overrides fully define a from-scratch tokenizer."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return NCMTokenizerConfig()


class NCMGeneratorConfig(PretrainedConfig):
    """Transformer that predicts the tokenizer's code levels coarse to fine, all
    codes of a level in parallel, conditioned on the earlier levels."""

    model_type = "ncm"
    sub_configs = {"tokenizer_config": NCMTokenizerConfig}

    def __init__(
        self,
        tokenizer_config: dict | NCMTokenizerConfig | None = None,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.1,
        bias: bool = True,
        rope_base: float = 100000.0,
        prefix_len: int = 1,
        null_prefix_token_id: int | None = None,
        prefix_pad_token_id: int | None = None,
        query_level0: bool = False,
        head_type: str = "tied",
        head_bias: bool = False,
        tied_head_learn_scale: bool = True,
        post_upsample_conv_kernel: int = 0,
        level_loss_alpha: float = 0.0,
        corruption_rate: float = 0.0,
        corruption_schedule: str = "linear",
        corruption_skip_level0: bool = True,
        **kwargs,
    ):
        if tokenizer_config is None:
            tokenizer_config = NCMTokenizerConfig()
        elif isinstance(tokenizer_config, dict):
            tokenizer_config = NCMTokenizerConfig(**tokenizer_config)
        self.tokenizer_config = tokenizer_config
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.rope_base = rope_base
        self.prefix_len = prefix_len
        self.null_prefix_token_id = null_prefix_token_id
        self.prefix_pad_token_id = prefix_pad_token_id
        self.query_level0 = query_level0
        self.head_type = head_type
        self.head_bias = head_bias
        self.tied_head_learn_scale = tied_head_learn_scale
        self.post_upsample_conv_kernel = post_upsample_conv_kernel
        self.level_loss_alpha = level_loss_alpha
        self.corruption_rate = corruption_rate
        self.corruption_schedule = corruption_schedule
        self.corruption_skip_level0 = corruption_skip_level0
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Also accepts a stage-1 tokenizer checkpoint, wrapping its config to
        start a fresh generator over it."""
        config_dict, _ = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get("model_type") == NCMTokenizerConfig.model_type:
            return cls(tokenizer_config=NCMTokenizerConfig.from_dict(config_dict))
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def validate(self):
        self.tokenizer_config.validate()
        if self.head_type not in ("tied", "linear", "shared"):
            raise ValueError(
                f"head_type must be tied, linear or shared, got {self.head_type}"
            )
        if self.corruption_schedule not in ("linear", "constant"):
            raise ValueError("corruption_schedule must be linear or constant")
        if self.n_embd % self.n_head or (self.n_embd // self.n_head) % 2:
            raise ValueError("n_embd / n_head must be an even integer (RoPE)")
        if self.prefix_len < 1:
            raise ValueError(
                "prefix_len must be >= 1 (unconditional runs use a null prefix)"
            )

    @property
    def levels(self) -> list[int]:
        return self.tokenizer_config.levels

    @property
    def first_predicted_level(self) -> int:
        """Level 0 comes from learned queries, or from its empirical unigram."""
        return 0 if self.query_level0 else 1
