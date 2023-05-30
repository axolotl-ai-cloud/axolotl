"""Module for custom LRScheduler class"""

from torch.optim.lr_scheduler import LRScheduler


class InterpolatingLogScheduler(LRScheduler):
    """
    A scheduler that interpolates learning rates in a logarithmic fashion
    """

    def __init__(self, optimizer, num_steps, min_lr, max_lr, last_epoch=-1):
        """A scheduler that interpolates learning rates in a logarithmic fashion

        Args:
        - optimizer: pytorch optimizer
        - num_steps: int, the number of steps over which to increase from the min_lr to the max_lr
        - min_lr: float, the minimum learning rate
        - max_lr: float, the maximum learning rate

        Usage:
            fc = nn.Linear(1,1)
            optimizer = optim.Adam(fc.parameters())
            lr_scheduler = InterpolatingLogScheduler(optimizer, num_steps=400, min_lr=1e-6, max_lr=1e-4)
        """
        self.num_steps = num_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.q = (max_lr / min_lr) ** (  # pylint: disable=invalid-name
            1 / (num_steps - 1)
        )
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= 0:
            lrs = [self.min_lr for base_lr in self.base_lrs]
        elif self.last_epoch < self.num_steps:
            lrs = [
                self.min_lr * (self.q ** (self.last_epoch - 1))
                for base_lr in self.base_lrs
            ]
        else:
            lrs = [self.max_lr for base_lr in self.base_lrs]

        return lrs
