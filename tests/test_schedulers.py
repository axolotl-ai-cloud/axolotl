"""
test module for the axolotl.utis.data module
"""

import unittest

import torch
from torch.optim import SGD

from axolotl.utils.schedulers import get_cosine_schedule_with_warmup_decay_constant


class TestCosineConstantLr(unittest.TestCase):
    """
    test class for encode pretraining and md5 helper
    """

    def setUp(self):
        self.train_steps = 1000
        self.warmup_steps = 10
        self.min_lr_ratio = 0.1
        self.constant_lr_ratio = 0.8
        self._lr = 0.01
        self.optimizer = SGD([torch.tensor(1)], lr=self._lr)
        self.lr_scheduler = get_cosine_schedule_with_warmup_decay_constant(  # pylint: disable=attribute-defined-outside-init
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.train_steps,
            min_lr_ratio=self.min_lr_ratio,
            constant_lr_ratio=self.constant_lr_ratio,
        )

    def test_schedulers(self):
        self.assertEqual(self.lr_scheduler.get_last_lr()[0], 0)
        for _ in range(self.warmup_steps):
            self.optimizer.step()
            self.lr_scheduler.step()
        self.assertEqual(self.lr_scheduler.get_last_lr()[0], self._lr)
        constant_step = int(self.train_steps * self.constant_lr_ratio)
        remaining_step = self.train_steps - constant_step
        for _ in range(constant_step):
            self.optimizer.step()
            self.lr_scheduler.step()
        self.assertEqual(
            self.lr_scheduler.get_last_lr()[0], self._lr * self.min_lr_ratio
        )
        for _ in range(remaining_step):
            self.optimizer.step()
            self.lr_scheduler.step()
        self.assertEqual(
            self.lr_scheduler.get_last_lr()[0], self._lr * self.min_lr_ratio
        )


if __name__ == "__main__":
    unittest.main()
