"""module to wrap flash_attn.ops.triton.cross_entropy"""
from flash_attn.ops.triton.cross_entropy import CrossEntropyLoss as FACrossEntropyLoss
from torch import Tensor, nn


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Optimized CrossEntropyLoss for Flash Attention.
    """

    def __init__(self, *args, inplace_backward=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.inplace_backward = inplace_backward

    def forward(  # pylint: disable=redefined-builtin
        self, input: Tensor, target: Tensor
    ) -> Tensor:
        return FACrossEntropyLoss.apply(
            input,
            target,
            smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
            inplace_backward=self.inplace_backward,
        )
