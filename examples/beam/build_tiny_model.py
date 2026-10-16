"""Build ./beam-tiny, a tiny random-weight Beam for beam-tiny-smoke.yaml."""

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

SRC = "reflection/Beam-1.0-501B-A23B"
OUT = "./beam-tiny"

cfg = AutoConfig.from_pretrained(
    SRC,
    num_hidden_layers=2,
    # one entry per layer: a sliding layer, then the full-attention layer
    layer_types=["sliding_attention", "full_attention"],
    mlp_layer_types=["dense", "sparse"],
    hidden_size=256,
    intermediate_size=512,
    moe_intermediate_size=256,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    num_experts=8,
    num_experts_per_tok=2,
    n_group=1,
    topk_group=1,
)

AutoModelForCausalLM.from_config(cfg).save_pretrained(OUT)
AutoTokenizer.from_pretrained(SRC).save_pretrained(OUT)
