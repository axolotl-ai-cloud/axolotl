"""
LoLCATs attention combining sliding window and linear attentions
- Using standard sliding window arrangement
- Training over long sequences with fixed memory with recurrent view
- During attention transfer, use Flash Attention to compute softmax attention outputs

For each layer:
- We first compute (softmax) attention over sliding windows
- We then compute standard linear attention to "fill in" the earlier parts
- We combine to model the entire sequence
"""
from .linear_window_attention_sw import hybrid_attention_quadratic
from .linear_window_attention_tk_long import LolcatsTKWindowLongAttention


class LolcatsSlidingWindowLongAttention(LolcatsTKWindowLongAttention):
    """
    Lolcats attention combining sliding window and linear attention
    """

    def __init__(self, remove_base_attn=True, **kwargs):
        # keep self.base_attn for Flash Attention inference
        super().__init__(remove_base_attn=True, **kwargs)
        self.quadratic_attention = hybrid_attention_quadratic
