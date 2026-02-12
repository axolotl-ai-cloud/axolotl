from __future__ import annotations

from typing import Optional

from transformers import PretrainedConfig


class LaneformerTPConfig(PretrainedConfig):
    model_type = "laneformer_tp"

    def __init__(
        self,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        intermediate_size: int = 4 * 4096,
        max_position_embeddings: int = 131072,
        eos_token_id: int = 0,
        vocab_size: int = 128256,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        multiple_of: int = 32,
        depth_init: bool = True,
        use_flex_attn: bool = False,
        attn_mask_type: str = "causal",
        use_cache: bool = True,
        # laneformer-specific extras (HF is fine with extra fields):
        num_lanes: int = 1,
        broadcast_delay: int = 1,
        use_comm: bool = True,
        use_early_comm: bool = True,
        lm_head_type: str = "replicate",
        pre_norm_lane_agg: bool = False,
        replicated_rmsn_scale: bool = True,
        tie_word_embeddings: bool = False,
        sliding_window: Optional[int] = None,
        sliding_window_n_layers: int = 0,
        **kwargs,
    ):
        super().__init__(
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.multiple_of = multiple_of
        self.depth_init = depth_init
        self.use_flex_attn = use_flex_attn
        self.attn_mask_type = attn_mask_type
        self.use_cache = use_cache
        self.sliding_window = sliding_window
        self.sliding_window_n_layers = sliding_window_n_layers

        # custom fields
        self.num_lanes = num_lanes
        self.use_comm = use_comm
        self.use_early_comm = use_early_comm
        self.broadcast_delay = broadcast_delay
        self.lm_head_type = lm_head_type
        self.pre_norm_lane_agg = pre_norm_lane_agg
        self.replicated_rmsn_scale = replicated_rmsn_scale
