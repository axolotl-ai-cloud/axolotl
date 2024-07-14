"""module to wrap flash_attn.ops.triton.cross_entropy"""
from flash_attn.ops.triton.cross_entropy import CrossEntropyLoss as FACrossEntropyLoss
from torch import Tensor, nn


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Optimized CrossEntropyLoss for Flash Attention.
    """

    def __init__(self, *args, inplace_backward=True, process_group=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.logit_scale: float = 1.0
        self.lse_square_scale: float = 0.0
        self.inplace_backward = inplace_backward
        self.process_group = process_group

    def forward(  # pylint: disable=redefined-builtin
        self, input: Tensor, target: Tensor
    ) -> Tensor:
        return FACrossEntropyLoss.apply(
            input,
            target,
            self.label_smoothing,
            self.logit_scale,
            self.lse_square_scale,
            self.ignore_index,
            self.inplace_backward,
            self.process_group,
        )[
            0
        ]  # first element of tuple is the loss, second is the z-loss
