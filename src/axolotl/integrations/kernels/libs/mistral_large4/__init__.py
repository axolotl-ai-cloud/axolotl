"""Mistral-Large-4 fused training kernels (MLA attention-linear LoRA).

Mistral-Large-4 is a DeepSeek-V3-style MLA + MoE model, but a *clean* one: the query is a
single ``q_proj`` (``q_lora_rank=null``) and there is no gradientless fused indexer, so unlike
DSV4 the MLA attention projections are safe to LoRA-adapt directly. This lib provides a fused,
quantization-dispatching LoRA forward for those projections.
"""

from .lora_attn import patch_mistral_large4_attn_lora

__all__ = ["patch_mistral_large4_attn_lora"]
