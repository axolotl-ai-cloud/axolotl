"""MiMo-V2 config expressed as transformers' native MiMo-V2-Flash config."""

from transformers.models.mimo_v2_flash.configuration_mimo_v2_flash import (
    MiMoV2FlashConfig,
)


class MiMoV2Config(MiMoV2FlashConfig):
    model_type = "mimo_v2"

    def __post_init__(self, **kwargs):
        # the FP8/MXFP4 checkpoint is dequantized by the load-time weight conversions
        kwargs.pop("quantization_config", None)
        if self.layer_types is None and (pattern := kwargs.get("hybrid_layer_pattern")):
            self.layer_types = [
                "sliding_attention" if swa else "full_attention" for swa in pattern
            ]
        if self.mlp_layer_types is None and (freq := kwargs.get("moe_layer_freq")):
            self.mlp_layer_types = ["sparse" if moe else "dense" for moe in freq]
        self.rms_norm_eps = kwargs.get("layernorm_epsilon", self.rms_norm_eps)
        rope = self.rope_parameters or {}
        if "rope_theta" in rope:
            partial = rope.get("partial_rotary_factor", 0.334)
            thetas = {
                "full_attention": rope["rope_theta"],
                "sliding_attention": kwargs.get("swa_rope_theta", 10_000.0),
            }
            self.rope_parameters = {
                layer_type: {
                    "rope_type": "default",
                    "rope_theta": theta,
                    "partial_rotary_factor": partial,
                }
                for layer_type, theta in thetas.items()
            }
        super().__post_init__(**kwargs)
