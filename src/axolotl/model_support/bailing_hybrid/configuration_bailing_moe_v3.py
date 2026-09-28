"""Bailing MoE V3 (Ling 3.0) configuration.

Source: https://huggingface.co/inclusionAI/Ling-3.0-flash/blob/main/configuration_bailing_moe_v3.py
Keys the published modeling code never reads are dropped; unknown keys still
round-trip through ``PretrainedConfig``.
"""

from transformers.configuration_utils import PretrainedConfig


class BailingMoeV3Config(PretrainedConfig):
    """Hybrid KDA/MLA MoE decoder used by the Ling 3.0 checkpoints."""

    model_type = "bailing_hybrid"
    keys_to_ignore_at_inference = ["past_key_values"]
    # names the reused deepseek_v3 components read
    attribute_map = {
        "num_local_experts": "num_experts",
        "attention_bias": "use_qkv_bias",
        "n_shared_experts": "num_shared_experts",
    }

    def __init__(
        self,
        vocab_size=157184,
        hidden_size=2560,
        intermediate_size=6144,
        num_hidden_layers=42,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        max_position_embeddings=262144,
        rope_theta=6000000.0,
        rope_scaling=None,
        rope_interleave=True,
        attention_dropout=0.0,
        use_qkv_bias=False,
        use_cache=True,
        tie_word_embeddings=False,
        pad_token_id=156892,
        eos_token_id=156895,
        # multi-latent attention
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        gated_attention_proj_granularity_type="head_wise",
        # kimi delta attention
        layer_group_size=6,
        head_dim=128,
        short_conv_kernel_size=4,
        no_kda_lora=True,
        kda_safe_gate=True,
        kda_lower_bound=-5.0,
        # mixture of experts
        num_experts=512,
        num_experts_per_tok=8,
        num_shared_experts=1,
        moe_intermediate_size=768,
        moe_shared_expert_intermediate_size=768,
        first_k_dense_replace=2,
        n_group=8,
        topk_group=4,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        num_nextn_predict_layers=0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_interleave = rope_interleave
        self.attention_dropout = attention_dropout
        self.use_qkv_bias = use_qkv_bias
        self.use_cache = use_cache

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.gated_attention_proj_granularity_type = (
            gated_attention_proj_granularity_type
        )

        self.layer_group_size = layer_group_size
        self.head_dim = head_dim
        self.short_conv_kernel_size = short_conv_kernel_size
        self.no_kda_lora = no_kda_lora
        self.kda_safe_gate = kda_safe_gate
        self.kda_lower_bound = kda_lower_bound

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.num_nextn_predict_layers = num_nextn_predict_layers

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # after `super().__init__`: a serialized `layer_types` must not win over the
        # layout the decoder layers build from. Sizes the stock hybrid cache too --
        # one linear-attention layer per KDA layer, a conv state per q/k/v projection
        self.layer_types = [
            "linear_attention"
            if self.is_linear_attention_layer(idx)
            else "full_attention"
            for idx in range(num_hidden_layers)
        ]
        self.number_of_conv_states = 3

    def is_linear_attention_layer(self, layer_idx: int) -> bool:
        """Every ``layer_group_size``-th layer is softmax attention; the tail that
        does not fill a whole group is softmax attention too."""
        full_groups = self.num_hidden_layers // self.layer_group_size
        is_group_end = (layer_idx + 1) % self.layer_group_size == 0
        return not (is_group_end or layer_idx >= full_groups * self.layer_group_size)
